from typing import Tuple, Optional, Union, Dict, Callable

import numpy as np
import torch
from torch_scatter import scatter
from e3nn.nn import FullyConnectedNet, Gate
from e3nn import o3
from e3nn.util.jit import compile_mode

from .mace import (
    BesselBasis, 
    PolynomialCutoff, 
    SymmetricContraction,
    tp_out_irreps_with_instructions,
    reshape_irreps
)

###################
# Embedding blocks
###################
class RepeatNodeEmbedding(torch.nn.Module):
    def __init__(self, num_repeats: int) -> None:
        super().__init__()
        self.num_repeats = num_repeats

    def forward(
        self, 
        x: torch.Tensor # [n_nodes, @irreps]
    ) -> torch.Tensor:
        return x.tile((1, self.num_repeats)) # [n_nodes, @irreps*num_repeats]


class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(self, r_max: float, num_bessel: int, num_polynomial_cutoff: int):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel, trainable=False)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
    ) -> torch.Tensor:
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]


class FourierBasisEmbeddingBlock(torch.nn.Module):
    def __init__(self, n_max: int) -> None:
        super().__init__()
        
        fourier_weights = (
            np.pi
            * torch.linspace(
                start=0.0,
                end=n_max,
                steps=n_max+1,
                dtype=torch.get_default_dtype(),
            )
        )
        self.register_buffer("fourier_weights", fourier_weights)
        self.out_dim = 2*(n_max+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        cos_comp = torch.cos(self.fourier_weights * x)  # [n_edges, n_max+1]
        sin_comp = torch.sin(self.fourier_weights * x)  # [n_edges, n_max+1]
        return torch.cat((cos_comp, sin_comp), dim=1)   # [n_edges,2*(n_max+1)]


class PolynomialBasisEmbeddingBlock(torch.nn.Module):
    def __init__(self, max_exp: int) -> None:
        super().__init__()

        powers = torch.linspace(
            start=-max_exp,
            end=max_exp,
            steps=2*max_exp+1,
            dtype=torch.get_default_dtype()
        )
        factors = torch.tensor(3).pow(powers+1)
        self.register_buffer('powers', powers)
        self.register_buffer('factors', factors)
        self.out_dim = 2*max_exp+1

    def forward(
        self, 
        x: torch.Tensor # [n_edges, 1]
    ) -> torch.Tensor:
        return self.factors*x.pow(self.powers) # [n_edges, 2*max_exp+1]

class WaveletEmbeddingBlock(torch.nn.Module):
    def __init__(self, num_freq: int) -> None:
        super().__init__()

        fourier_weights = (
            np.pi
            * torch.linspace(
                start=1.0,
                end=num_freq,
                steps=num_freq,
                dtype=torch.get_default_dtype(),
            )
        )

        shifts = torch.zeros(num_freq, dtype=torch.get_default_dtype())

        self.register_buffer('fourier_weights', fourier_weights)
        self.shifts = torch.nn.Parameter(shifts)

        self.out_dim = 2*num_freq + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero_component = torch.exp(-0.5*x.pow(2))
        cos_components = (
            torch.cos(self.fourier_weights*(x - self.shifts))
            * torch.exp(-0.5*(x - self.shifts).pow(2))
        )
        sin_components = (
            torch.sin(self.fourier_weights*(x - self.shifts))
            * torch.exp(-0.5*(x - self.shifts).pow(2))
        )
        return torch.cat((zero_component, cos_components, sin_components), dim=1)

#################
# Readout blocks
#################

class GeneralLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self, 
        irreps_in: o3.Irreps, 
        hidden_irreps: o3.Irreps,
        irreps_out: o3.Irreps
    ):
        super().__init__()
        self.linear1 = o3.Linear(irreps_in=irreps_in, irreps_out=hidden_irreps)
        self.linear2 = o3.Linear(irreps_in=hidden_irreps, irreps_out=irreps_out)

    def forward(
        self, 
        x: torch.Tensor # [n_nodes, @irreps_in]
    ) -> torch.Tensor:  
        x = self.linear1(x)
        return self.linear2(x)  # [n_nodes, @irreps_out]


class GeneralNonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        hidden_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
        gate: Callable
    ):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.irreps_out = irreps_out
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_nonlin)
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=self.irreps_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode('script')
class half_irreps(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        out_irreps = []
        columns0 = []
        columns1 = []
        ix = 0
        for mul, ir in irreps_in:
            assert mul%2==0
            n_half = int(mul/2)
            out_irreps.append(
                (n_half, (ir.l, ir.p))
            )
            columns0.extend([i+ix for i in range(n_half*ir.dim)])
            columns1.extend([i+ix+n_half*ir.dim for i in range(n_half*ir.dim)])
            ix += mul*ir.dim

        self.irreps_out = o3.Irreps(out_irreps)
        self.columns_0 = torch.tensor(columns0, dtype=torch.long)
        self.columns_1 = torch.tensor(columns1, dtype=torch.long)

    def forward(
        self, 
        x: torch.Tensor # [n_nodes, @irreps]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = (x[:, self.columns_0], x[:, self.columns_1])
        return out # [n_nodes, irreps/2], # [n_nodes, @irreps/2]

class OneTPReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        
        self.split_irreps = half_irreps(irreps_in)

        irrep_half = self.split_irreps.irreps_out

        self.tp3 = o3.FullyConnectedTensorProduct(
            irreps_in1=irrep_half,
            irreps_in2=irrep_half,
            irreps_out=irreps_out,
            # path_normalization='element',
            # irrep_normalization='norm',
            internal_weights=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [num_graphs, @irreps_in]
        x0, x1 = self.split_irreps(x)
        x = self.tp3(x0,x1)
        return x # [num_graphs, @irreps_out]


#################
# Product blocks
#################
class EquivariantProductBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        use_sc: bool = True,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            element_dependent=False,
            num_elements=None,
        )
        # Update linear
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self, 
        node_feats: torch.Tensor, 
        sc: torch.Tensor, 
    ) -> torch.Tensor:
        node_feats = self.symmetric_contractions(x=node_feats, y=None)
        if self.use_sc:
            return self.linear(node_feats) + sc

        return self.linear(node_feats)

#####################
# Interaction blocks
#####################
class TensorProductInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
        agg_norm_const: float,
        reduce: str = 'sum'
    ) -> None:
        super().__init__()
        self._node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self._irreps_out = irreps_out
        self.agg_norm_const = agg_norm_const
        self.reduce = reduce

        self.linear_up = o3.Linear(
            irreps_in=self._node_feats_irreps,
            irreps_out=self._node_feats_irreps,
            internal_weights=True,
            shared_weights=True
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self._node_feats_irreps,
            self.edge_attrs_irreps,
            self._irreps_out,
        )
        self.conv_tp = o3.TensorProduct(
            self._node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.SiLU(),
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.linear = o3.Linear(
            irreps_in=irreps_mid, 
            irreps_out=self._irreps_out, 
            internal_weights=True, 
            shared_weights=True
        )

        # CHANGED
        # self.skip_tp = o3.FullyConnectedTensorProduct(
        #     self._irreps_out, '1x0e', self._irreps_out
        # )

        self.reshape = reshape_irreps(self._irreps_out)

    @property
    def irreps_in(self):
        return self._node_feats_irreps
    
    @property
    def irreps_out(self):
        return self._irreps_out

    def forward(
        self,
        node_feats: torch.Tensor, # [num_nodes, @node_feats_irreps]
        edge_attrs: torch.Tensor, # [num_edges, @edge_attrs_irreps]
        edge_feats: torch.Tensor, # [num_edges, @edge_feats_irreps]
        edge_index: torch.Tensor, # [2, num_edges]
        node_attrs: Optional[torch.Tensor] = None
    ) -> torch.Tensor: 
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(node_feats[sender], edge_attrs, tp_weights)  
        message = scatter(
            src=mji, index=receiver, dim=0, dim_size=num_nodes, reduce=self.reduce
        ) / self.agg_norm_const # [n_nodes, irreps]
        # TODO: try batchnorm
        message = self.linear(message)
        # message = self.skip_tp(message, node_attrs) # CHANGED
        return (
            self.reshape(message),
            None, # no skip connection
        )  # [n_nodes, channels, (lmax + 1)**2]

class TensorProductResidualInteractionBlock(TensorProductInteractionBlock):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
        sc_irreps_out: o3.Irreps,
        agg_norm_const: float,
        reduce: str = 'sum'
    ) -> None:
        super().__init__(
            node_feats_irreps,
            edge_attrs_irreps,
            edge_feats_irreps,
            irreps_out,
            agg_norm_const,
            reduce
        )
        self.sc_irreps_out = sc_irreps_out

        # add skip connection # CHANGED
        self.linear_skip = o3.Linear(
            irreps_in=self._node_feats_irreps,
            irreps_out=self.sc_irreps_out,
            internal_weights=True,
            shared_weights=True
        )

        # self.skip_tp = o3.FullyConnectedTensorProduct(
        #     self._node_feats_irreps, '1x0e', self.sc_irreps_out
        # )

    def forward(
        self,
        node_feats: torch.Tensor, # [num_nodes, @node_feats_irreps]
        edge_attrs: torch.Tensor, # [num_edges, @edge_attrs_irreps]
        edge_feats: torch.Tensor, # [num_edges, @edge_feats_irreps]
        edge_index: torch.Tensor, # [2, num_edges]
        node_attrs: Optional[torch.Tensor] = None
    ) -> torch.Tensor: 
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        sc = self.linear_skip(node_feats) # skip connection # CHANGED
        # sc = self.skip_tp(node_feats, node_attrs) # skip connection # CHANGED
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(node_feats[sender], edge_attrs, tp_weights)  
        message = scatter(
            src=mji, index=receiver, dim=0, dim_size=num_nodes, reduce=self.reduce
        ) / self.agg_norm_const # [n_nodes, irreps]
        # TODO: try batchnorm
        message = self.linear(message)
        return (
            self.reshape(message),
            sc, 
        )  # [n_nodes, channels, (lmax + 1)**2]

#################
# Pooling blocks
#################

class GlobalSumHistoryPooling(torch.nn.Module):
    def __init__(self, reduce='sum') -> None:
        super().__init__()
        self.reduce = reduce

    def forward(
        self,
        node_ft_history: torch.Tensor, # [num_nodes, irreps, num_message_passes]
        batch_index: torch.Tensor, # [num_nodes]
        num_graphs: int,
    ) -> torch.Tensor:
        x = torch.sum(node_ft_history, dim=-1) # [num_nodes, irreps]
        graph_ft = scatter(
            src=x,
            index=batch_index.view(-1,1),
            dim=0,
            dim_size=num_graphs,
            reduce=self.reduce
        )
        return graph_ft # [irreps,]