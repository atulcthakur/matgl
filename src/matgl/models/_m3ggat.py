"""
Production‐ready implementation of two M3GNet variants:

1. M3GNetWithAttention – a version where a single-head attention mechanism is used for node updates.
2. OriginalM3GNet – the original architecture (without attention).

The main class below (M3GNet) accepts a flag "use_attention". When True, the model
builds its message-passing layers using AttentiveM3GNetBlock (which uses SingleHeadNodeAttention);
otherwise, it uses the standard M3GNetBlock.

Reference:
  Chen, C., Ong, S.P. A universal graph deep learning interatomic potential for the periodic table.
  Nature Computational Science, 2023, 2, 718–728. DOI: 10.1038/s43588-022-00349-3.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Literal

import dgl
import torch
from torch import nn
import dgl.function as fn
from dgl.ops import edge_softmax

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
    ensure_line_graph_compatibility,
)
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
    EmbeddingBlock,
    GatedMLP,
    M3GNetBlock,  # Original block from the published architecture.
    ReduceReadOut,
    Set2SetReadOut,
    SphericalBesselWithHarmonics,
    ThreeBodyInteractions,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from matgl.utils.cutoff import polynomial_cutoff

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__file__)


###########################################################################
# 1. Single-head Attention Module for Node Updates
###########################################################################
class SingleHeadNodeAttention(nn.Module):
    """
    Single-head self-attention for node updates.
    Each node feature is projected to query (Q), key (K), and value (V) spaces.
    Attention scores are computed on edges (using the convention that the message is sent
    from the source node to the destination node) and then aggregated at the destination.
    """
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.query_proj = nn.Linear(in_feats, out_feats, bias=False)
        self.key_proj   = nn.Linear(in_feats, out_feats, bias=False)
        self.value_proj = nn.Linear(in_feats, out_feats, bias=False)
        self.scaling    = out_feats ** 0.5

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            # Project node features
            Q = self.query_proj(node_feats)  # shape: (N, out_feats)
            K = self.key_proj(node_feats)      # shape: (N, out_feats)
            V = self.value_proj(node_feats)    # shape: (N, out_feats)
            g.ndata["Q"] = Q
            g.ndata["K"] = K
            g.ndata["V"] = V

            # Define edge message: destination node gets dot-product between its Q and source's K.
            def edge_message_func(edges):
                # Convention: each edge from src->dst; use dst's Q and src's K.
                attn_score = (edges.dst["Q"] * edges.src["K"]).sum(dim=-1) / self.scaling
                return {"attn_score": attn_score, "msg": edges.src["V"]}

            g.apply_edges(edge_message_func)
            # Normalize attention scores over incoming edges for each destination node.
            g.edata["alpha"] = edge_softmax(g, g.edata["attn_score"])
            # Compute weighted messages.
            g.edata["weighted_msg"] = g.edata["alpha"].unsqueeze(-1) * g.edata["msg"]
            # Aggregate weighted messages at each destination node.
            g.update_all(fn.copy_e("weighted_msg", "m"), fn.sum("m", "h"))
            updated_node_feats = g.ndata["h"]
        return updated_node_feats


###########################################################################
# 2. Attentive M3GNet Block: Replaces standard node update with attention.
###########################################################################
class AttentiveM3GNetBlock(nn.Module):
    """
    A drop-in replacement for M3GNetBlock that uses single-head attention for node updates.
    Here we keep the edge-update (three-body interactions, gating, etc.) unchanged,
    and replace the node aggregation with attention.
    """
    def __init__(
        self,
        degree: int,
        activation: nn.Module,
        conv_hiddens: list[int],
        dim_node_feats: int,
        dim_edge_feats: int,
        dim_state_feats: int = 0,
        include_state: bool = False,
    ):
        super().__init__()
        self.include_state = include_state

        # Edge update: using an MLP on edge features (this mirrors part of the original block)
        self.edge_mlp = MLP(
            dims=[dim_edge_feats, conv_hiddens[0], dim_edge_feats],
            activation=activation,
            activate_last=True,
        )
        # Replace standard node update with single-head attention.
        self.node_attention = SingleHeadNodeAttention(
            in_feats=dim_node_feats, out_feats=dim_node_feats
        )
        # (Optional) State update if needed.
        if include_state and dim_state_feats > 0:
            self.state_mlp = MLP(
                dims=[dim_state_feats, conv_hiddens[0], dim_state_feats],
                activation=activation,
                activate_last=True,
            )
        else:
            self.state_mlp = None

    def forward(
        self,
        g: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor | None,
    ):
        with g.local_scope():
            # Update edges (if your original block does more here, add as needed)
            g.edata["edge_feat"] = edge_feat
            def edge_udf(edges):
                x = edges.data["edge_feat"]
                return {"edge_feat_out": self.edge_mlp(x)}
            g.apply_edges(edge_udf)
            edge_feat_out = g.edata["edge_feat_out"]

            # Update node features via attention aggregator.
            node_feat_out = self.node_attention(g, node_feat)

            # Update state if applicable.
            if self.include_state and self.state_mlp is not None and state_feat is not None:
                state_feat_out = self.state_mlp(state_feat)
            else:
                state_feat_out = state_feat

        return edge_feat_out, node_feat_out, state_feat_out


###########################################################################
# 3. Main M3GNet Model (with switchable attention)
###########################################################################
class M3GNet(MatGLModel, IOMixIn):
    """
    Main M3GNet model with an option to use single-head attention in the node-update step.
    
    When `use_attention=True`, the graph layers are built using AttentiveM3GNetBlock (with neighbor-level
    attention). When False, the standard M3GNetBlock is used (the original architecture with gating).
    """
    __version__ = 2

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        dim_node_embedding: int = 64,
        dim_edge_embedding: int = 64,
        dim_state_embedding: int = 0,
        ntypes_state: int | None = None,
        dim_state_feats: int | None = None,
        max_n: int = 3,
        max_l: int = 3,
        nblocks: int = 3,
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "SphericalBessel",
        is_intensive: bool = True,
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        units: int = 64,
        ntargets: int = 1,
        use_smooth: bool = False,
        use_phi: bool = False,
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        include_state: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        use_attention: bool = False,  # New flag: use attention or not.
        **kwargs,
    ):
        """
        Args:
            ... (all standard arguments as in the published M3GNet)
            use_attention: Whether to insert a single-head attention mechanism for node updates.
                           If False, the original M3GNetBlock is used.
        """
        super().__init__()
        self.save_args(locals(), kwargs)
        self.use_attention = use_attention

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type. Use one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.element_types = element_types or DEFAULT_ELEMENTS

        # Bond expansion from pairwise distances.
        self.bond_expansion = BondExpansion(max_l, max_n, cutoff, rbf_type=rbf_type, smooth=use_smooth)
        degree_rbf = max_n if use_smooth else max_n * max_l

        # Embedding block: embed atomic numbers and initialize node/edge features.
        self.embedding = EmbeddingBlock(
            degree_rbf=degree_rbf,
            dim_node_embedding=dim_node_embedding,
            dim_edge_embedding=dim_edge_embedding,
            ntypes_node=len(element_types),
            ntypes_state=ntypes_state,
            dim_state_feats=dim_state_feats,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        # Three-body basis expansion for angular information.
        self.basis_expansion = SphericalBesselWithHarmonics(
            max_n=max_n,
            max_l=max_l,
            cutoff=cutoff,
            use_phi=use_phi,
            use_smooth=use_smooth,
        )

        # Three-body interactions module.
        self.three_body_interactions = nn.ModuleList(
            {
                ThreeBodyInteractions(
                    update_network_atom=MLP(
                        dims=[dim_node_embedding, degree_rbf],
                        activation=nn.Sigmoid(),
                        activate_last=True,
                    ),
                    update_network_bond=GatedMLP(in_feats=degree_rbf, dims=[dim_edge_embedding], use_bias=False),
                )
                for _ in range(nblocks)
            }
        )

        dim_state_feats = dim_state_embedding  # For convenience

        # Choose which block type to use for graph layers.
        if self.use_attention:
            block_class = AttentiveM3GNetBlock
        else:
            block_class = M3GNetBlock

        self.graph_layers = nn.ModuleList(
            {
                block_class(
                    degree=degree_rbf,
                    activation=activation,
                    conv_hiddens=[units, units],
                    dim_node_feats=dim_node_embedding,
                    dim_edge_feats=dim_edge_embedding,
                    dim_state_feats=dim_state_feats,
                    include_state=include_state,
                )
                for _ in range(nblocks)
            }
        )

        # Readout layers.
        if is_intensive:
            input_feats = dim_node_embedding if field == "node_feat" else dim_edge_embedding
            if readout_type == "set2set":
                self.readout = Set2SetReadOut(
                    in_feats=input_feats, n_iters=niters_set2set, n_layers=nlayers_set2set, field=field
                )
                readout_feats = 2 * input_feats + dim_state_feats if include_state else 2 * input_feats
            elif readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)
                readout_feats = units + dim_state_feats if include_state else units
            else:
                self.readout = ReduceReadOut("mean", field=field)
                readout_feats = input_feats + dim_state_feats if include_state else input_feats

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()
        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=dim_node_embedding, dims=[units, units], num_targets=ntargets
            )

        # Save additional parameters.
        self.max_n = max_n
        self.max_l = max_l
        self.n_blocks = nblocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_state = include_state
        self.task_type = task_type
        self.is_intensive = is_intensive

    def forward(
        self,
        g: dgl.DGLGraph,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
        return_all_layer_output: bool = False,
    ):
        """
        Forward pass: message passing and feature aggregation.
        """
        node_types = g.ndata["node_type"]
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec
        g.edata["bond_dist"] = bond_dist

        expanded_dists = self.bond_expansion(g.edata["bond_dist"])
        if l_g is None:
            l_g = create_line_graph(g, self.threebody_cutoff)
        else:
            l_g = ensure_line_graph_compatibility(g, l_g, self.threebody_cutoff)
        l_g.apply_edges(compute_theta_and_phi)
        g.edata["rbf"] = expanded_dists
        three_body_cutoff = polynomial_cutoff(g.edata["bond_dist"], self.threebody_cutoff)
        node_feat, edge_feat, state_feat = self.embedding(node_types, g.edata["rbf"], state_attr)

        fea_dict = {
            "bond_expansion": expanded_dists,
            "three_body_basis": self.basis_expansion(l_g),
            "embedding": {"node_feat": node_feat, "edge_feat": edge_feat, "state_feat": state_feat},
        }

        # Message-passing through blocks.
        for i in range(self.n_blocks):
            edge_feat = self.three_body_interactions[i](
                g, l_g, fea_dict["three_body_basis"], three_body_cutoff, node_feat, edge_feat
            )
            edge_feat, node_feat, state_feat = self.graph_layers[i](g, edge_feat, node_feat, state_feat)
            fea_dict[f"gc_{i + 1}"] = {"node_feat": node_feat, "edge_feat": edge_feat, "state_feat": state_feat}

        g.ndata["node_feat"] = node_feat
        g.edata["edge_feat"] = edge_feat

        if self.is_intensive:
            field_vec = self.readout(g)
            readout_vec = torch.hstack([field_vec, state_feat]) if self.include_state else field_vec
            fea_dict["readout"] = readout_vec
            output = self.final_layer(readout_vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:
            g.ndata["atomic_properties"] = self.final_layer(g)
            fea_dict["readout"] = g.ndata["atomic_properties"]
            output = dgl.readout_nodes(g, "atomic_properties", op="sum")
        fea_dict["final"] = output

        if return_all_layer_output:
            return fea_dict
        return torch.squeeze(output)

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        output_layers: list | None = None,
        return_features: bool = False,
    ):
        """
        Convenience method to predict properties for a given structure.
        """
        allowed_output_layers = [
            "bond_expansion",
            "embedding",
            "three_body_basis",
            "readout",
            "final",
        ] + [f"gc_{i + 1}" for i in range(self.n_blocks)]

        if not return_features:
            output_layers = ["final"]
        elif output_layers is None:
            output_layers = allowed_output_layers
        elif not isinstance(output_layers, list) or set(output_layers).difference(allowed_output_layers):
            raise ValueError(f"Invalid output_layers, must be a subset of {allowed_output_layers}.")

        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph
            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)

        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]

        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)

        model_output = self(g=g, state_attr=state_feats, return_all_layer_output=True)

        if not return_features:
            return model_output["final"].detach()

        return {k: v for k, v in model_output.items() if k in output_layers}