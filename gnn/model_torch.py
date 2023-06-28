# %%
from typing import Any, Optional, Tuple, Dict, Union
from argparse import Namespace

import torch
from torch_geometric.data import Batch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.io import CartesianTensor

from .blocks import (
    MACELayer,
    Cart_4_to_Mandel,
    GeneralNonLinearReadoutBlock,
    GeneralLinearReadoutBlock,
    OneTPReadoutBlock,
    GlobalSumHistoryPooling,
    TensorProductInteractionBlock,
    EquivariantProductBlock
)
from .mace import get_edge_vectors_and_lengths, reshape_irreps
# %%

class PositiveLiteGNN(torch.nn.Module):
    def __init__(self, params: Namespace, *args: Any, **kwargs: Any) -> "PositiveLiteGNN":
        super().__init__(*args, **kwargs)

        self.params = params
        hidden_irreps = o3.Irreps(params.hidden_irreps)

        self.node_ft_embedding = torch.nn.Linear(in_features=1, out_features=hidden_irreps.count(o3.Irrep(0,1)))
        self.number_of_edge_basis = 6
        node_ft_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        edge_feats_irreps = o3.Irreps(f"{self.number_of_edge_basis*2}x0e")
        edge_attr_irreps = o3.Irreps.spherical_harmonics(params.lmax)
        self.spherical_harmonics = o3.SphericalHarmonics(
            edge_attr_irreps,
            normalize=True, normalization='component'
        )
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (edge_attr_irreps * num_features).sort()[0].simplify()
        readout_irreps = o3.Irreps(params.readout_irreps)

        self.linear_skip = o3.Linear(
                irreps_in=hidden_irreps,
                irreps_out=hidden_irreps,
                internal_weights=True,
                shared_weights=True
            )
        
        self.interactions = torch.nn.ModuleList([
            TensorProductInteractionBlock(
                node_feats_irreps=node_ft_irreps,
                edge_attrs_irreps=edge_attr_irreps,
                edge_feats_irreps=edge_feats_irreps,
                irreps_out=interaction_irreps,
                agg_norm_const=params.agg_norm_const,
                reduce=params.interaction_reduction,
                bias=True
            ),
            TensorProductInteractionBlock(
                node_feats_irreps=interaction_irreps,
                edge_attrs_irreps=edge_attr_irreps,
                edge_feats_irreps=edge_feats_irreps,
                irreps_out=hidden_irreps,
                agg_norm_const=params.agg_norm_const,
                reduce=params.interaction_reduction,
                bias=True
            )
        ])
        self.product = EquivariantProductBlock(
            node_feats_irreps=readout_irreps,
            target_irreps=readout_irreps,
            correlation=params.correlation,
            use_sc=False
        )
        # self.gnn_layers = torch.nn.ModuleList([
        #     MACELayer(node_ft_irreps, edge_attr_irreps, edge_feats_irreps, interaction_irreps, hidden_irreps, params.agg_norm_const, params.interaction_reduction, True, params.correlation),
        #     MACELayer(hidden_irreps, edge_attr_irreps, edge_feats_irreps, interaction_irreps, hidden_irreps, params.agg_norm_const, params.interaction_reduction, True, params.correlation),
        # ])

        # self.lin_readout = GeneralLinearReadoutBlock(
        #         irreps_in=hidden_irreps,
        #         irreps_out=readout_irreps,
        #         hidden_irreps=readout_irreps,
        #     )
        self.nonlin_readout = GeneralNonLinearReadoutBlock(
                irreps_in=hidden_irreps,
                irreps_out=hidden_irreps,
                hidden_irreps=hidden_irreps,
                gate=torch.nn.functional.silu,
            )       
        
        self.pooling = GlobalSumHistoryPooling(reduce=params.global_reduction)
        self.linear = o3.Linear(hidden_irreps, readout_irreps, 
            internal_weights=True, 
            shared_weights=True,
            biases=True
        )
        self.fourth_order_expansion = OneTPReadoutBlock(
            irreps_in=readout_irreps,
            irreps_out=o3.Irreps('2x0e+2x2e+1x4e')
        )

        self.el_tens = CartesianTensor('ijkl=ijlk=jikl=klij')
        self.cart_to_Mandel = Cart_4_to_Mandel()
        if params.matrix_func.lower()=='exp':
            self.matrix_func = torch.linalg.matrix_exp
        elif params.matrix_func.lower()=='square':
            self.matrix_func = lambda x: torch.linalg.matrix_power(x, 2)


    def forward(self, batch: Batch) -> Dict:
        
        num_graphs = batch.num_graphs
        edge_index = batch.edge_index
        node_ft = batch.node_attrs
        # _,cnt = torch.unique(edge_index[1], return_counts=True)
        # node_ft = cnt.unsqueeze(-1).float()
        node_ft = self.node_ft_embedding(node_ft)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch.positions, edge_index=edge_index, shifts=batch.shifts
        )
        edge_length_embedding = soft_one_hot_linspace(
            lengths.squeeze(-1), start=0, end=0.6, number=self.number_of_edge_basis, basis='gaussian', cutoff=False
        )
        edge_radii = batch.edge_attr
        edge_radius_embedding = soft_one_hot_linspace(
            edge_radii.squeeze(-1), 0, 0.03, self.number_of_edge_basis, 'gaussian', False
        )
        edge_feats = torch.cat(
            (edge_length_embedding, edge_radius_embedding), 
            dim=1
        )
        edge_sh = self.spherical_harmonics(vectors)
        
        outputs = []

        node_ft, _ = self.interactions[0](node_ft, edge_sh, edge_feats, edge_index)
        node_ft, _ = self.interactions[1](node_ft, edge_sh, edge_feats, edge_index)
        # self.product(node_ft, sc)

        # node_ft = self.gnn_layers[0](node_ft, edge_index, edge_sh, edge_feats)

        # for i_mp in range(self.params.message_passes-1):
        #     node_ft = self.gnn_layers[1](node_ft, edge_index, edge_sh, edge_feats) + self.linear_skip(node_ft)
        
        # outputs.append(self.lin_readout(node_ft))
        outputs.append(self.nonlin_readout(node_ft))
       
        outputs = torch.stack(outputs, dim=-1)

        graph_ft = self.pooling(outputs, batch.batch, num_graphs)
        graph_ft = self.linear(graph_ft)

        graph_ft = self.product(graph_ft, None)

        # graph_ft = self.linear(graph_ft)
        graph_ft = self.fourth_order_expansion(graph_ft) 

        stiffness = self.el_tens.to_cartesian(graph_ft)
        C = self.cart_to_Mandel(stiffness)
        C_exp = self.matrix_func(C)

        return {'stiffness': C_exp}