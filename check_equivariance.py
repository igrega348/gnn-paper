# %%
import sys
from argparse import Namespace

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from e3nn import o3
from e3nn.io import CartesianTensor

from data import GLAMM_rhotens_Dataset as GLAMM_Dataset
from data import elasticity_func
from gnn.models import CrystGraphConv, LatticeGNN, SpectralGNN, PositiveGNN
# %%
def rotate_lat_sh(lat: Data):
    Q = o3.rand_matrix()
    irreps = o3.Irreps('2x0e+2x2e+1x4e')
    R = irreps.D_from_matrix(Q)
    transformed = Data(
        node_attrs=lat.node_attrs,
        edge_attr=lat.edge_attr,
        edge_index=lat.edge_index,
        positions = torch.einsum('ij,pj->pi', Q, lat.positions),
        unit_shifts = lat.unit_shifts,
        shifts = torch.einsum('ij,pj->pi', Q, lat.shifts),
        rel_dens=lat.rel_dens,
        stiffness=torch.einsum('pj,ij->pi', lat.stiffness, R),
        name = lat.name
    )
    return transformed, Q
def rotate_lat_cart(lat: Data):
    Q = o3.rand_matrix()
    transformed = Data(
        node_attrs=lat.node_attrs,
        edge_attr=lat.edge_attr,
        edge_index=lat.edge_index,
        positions = torch.einsum('ij,pj->pi', Q, lat.positions),
        shifts = torch.einsum('ij,pj->pi', Q, lat.shifts),
        unit_shifts = lat.unit_shifts,
        rel_dens=lat.rel_dens,
        stiffness=torch.einsum('pijkl,ai,bj,ck,dl->pabcd', lat.stiffness, Q,Q,Q,Q),
        name = lat.name
    )
    return transformed, Q
indices = [
    (0,0,0,0),(0,0,1,1),(0,0,2,2),(0,0,0,1),(0,0,0,2),(0,0,1,2),
    (1,1,1,1),(1,1,2,2),(1,1,0,1),(1,1,0,2),(1,1,1,2),
    (2,2,2,2),(2,2,0,1),(2,2,0,2),(2,2,1,2),
    (0,1,0,1),(0,1,0,2),(0,1,1,2),
    (0,2,0,2),(0,2,1,2),
    (1,2,1,2)
]
# %%
def get_C_21(C):
    if C.ndim == 4:
        C = C.unsqueeze(0)
    C_21 = C.new_zeros(C.shape[0], 21)
    for i,j,k,l in indices:
        C_21[:,indices.index((i,j,k,l))] = C[:,i,j,k,l]
    return C_21
# %%
def assemble_C(C_21):
    if C_21.ndim == 1:
        C_21 = C_21.unsqueeze(0)
    C = C_21.new_zeros(C_21.shape[0], 3, 3, 3, 3)
    for i,j,k,l in indices:
        C[:,i,j,k,l] = C_21[:,indices.index((i,j,k,l))]
        C[:,i,j,l,k] = C_21[:,indices.index((i,j,k,l))]
        C[:,j,i,k,l] = C_21[:,indices.index((i,j,k,l))]
        C[:,j,i,l,k] = C_21[:,indices.index((i,j,k,l))]
        C[:,k,l,i,j] = C_21[:,indices.index((i,j,k,l))]
        C[:,k,l,j,i] = C_21[:,indices.index((i,j,k,l))]
        C[:,l,k,i,j] = C_21[:,indices.index((i,j,k,l))]
        C[:,l,k,j,i] = C_21[:,indices.index((i,j,k,l))]
    return C
# %%
def main(model: str):
    # %%
    dataset = GLAMM_Dataset(
        root='./GLAMMDsetT',
        catalogue_path='./GLAMMDsetT/raw/tiny_dset_1298_val.lat',
        dset_fname='validation.pt',
        n_reldens=5,
        choose_reldens='half',
        graph_ft_format='cartesian_4',
    )
    #     root=f'./GLAMMDsetO',
    #     catalogue_path=f'./GLAMMDsetO/raw/overfit_dset_60_train.lat',
    #     dset_fname='train.pt',
    #     n_reldens=5,
    #     choose_reldens='half',
    #     graph_ft_format='cartesian_4',
    # )

    if model.lower() == 'mace':
        # Prediction with LatticeGNN
        el_tens = CartesianTensor('ijkl=jikl=ijlk=klij')
        graph = dataset[0]
        delattr(graph, 'compliance')
        graph.stiffness = el_tens.from_cartesian(graph.stiffness.float())
        rot_graph, Q = rotate_lat_sh(graph)
        data = Batch.from_data_list([graph, rot_graph])
        params = Namespace(
                lmax=3, # spherical harmonics
                hidden_irreps='32x0e+32x1o+32x2e',
                readout_irreps='32x0e+32x1o+32x2e',
                interaction_reduction='sum',
                agg_norm_const=1,
                correlation=3,
                global_reduction='mean',
                message_passes=2,
                interaction_bias=True,
                optimizer='adamw',
                lr=0.001,
                amsgrad=True,
                weight_decay=1e-8,
                beta1=0.99,
                epsilon=1e-8,
                scheduler=None,
            )
        model = LatticeGNN(params)
        with torch.no_grad():
            model.eval()
            out = model(data)
        # equivariance of input data:
        C = el_tens.to_cartesian(graph.stiffness[0])
        C_rot = el_tens.to_cartesian(rot_graph.stiffness[0])
        print('Equivariance of input data:')
        print(torch.allclose(torch.einsum('ijkl,ai,bj,ck,dl->abcd',C,Q,Q,Q,Q), C_rot, rtol=1e-3))
        # equivariance of output data:
        C = el_tens.to_cartesian(out['stiffness'][0])
        C_rot = el_tens.to_cartesian(out['stiffness'][1])
      

    elif model.lower() == 'cgc':
        # Prediction with CrystGraphConv
        graph = dataset[0]
        delattr(graph, 'compliance')
        graph.stiffness = graph.stiffness.float()
        rot_graph, Q = rotate_lat_cart(graph)
        data = Batch.from_data_list([graph, rot_graph])
        data.stiffness = get_C_21(data.stiffness)

        params = Namespace(
            hidden_dim=32,
            interaction_reduction='sum',
            global_reduction='min',
            message_passes=5,
            optimizer='radam',
            lr_milestones=[50,100],
            lr=1e-3,
            amsgrad=True,
            weight_decay=1e-8,
            beta1=0.9,
            epsilon=1e-8,
            scheduler='multisteplr',
        )
        model = CrystGraphConv(params)
        with torch.no_grad():
            model.eval()
            out = model(data)
        # equivariance of input data:
        C = graph.stiffness[0]
        C_rot = rot_graph.stiffness[0]
        print('Equivariance of input data:')
        print(torch.allclose(torch.einsum('ijkl,ai,bj,ck,dl->abcd',C,Q,Q,Q,Q), C_rot, rtol=1e-3))
        # equivariance of output data:
        out = assemble_C(out['stiffness'])
        C = out[0]
        C_rot = out[1]
        

    elif model.lower() == 'spectral':
        graph = dataset[0]
        delattr(graph, 'compliance')
        graph.stiffness = graph.stiffness.float()
        rot_graph, Q = rotate_lat_cart(graph)
        data = Batch.from_data_list([graph, rot_graph])
        data.stiffness = get_C_21(data.stiffness)

        params = Namespace(
            lmax=3, # spherical harmonics
            # hidden_irreps='64x0e+64x1o+64x2e',
            # hidden_irreps='32x0e+32x1o',
            hidden_irreps='32x0e+32x1o+32x2e',
            # readout_irreps='21x0e+8x1o',
            readout_irreps='22x0e+1x2e',
            interaction_reduction='sum',
            agg_norm_const=1.0,
            correlation=3,
            global_reduction='mean',
            message_passes=2,
            interaction_bias=True,
            optimizer='adamw',
            lr=1e-3,
            amsgrad=True,
            weight_decay=1e-8,
            beta1=0.9,
            epsilon=1e-8,
            scheduler=None,
        )
        model = SpectralGNN(params)
        with torch.no_grad():
            model.eval()
            out = model(data)
       
        # equivariance of input data:
        C = graph.stiffness[0]
        C_rot = rot_graph.stiffness[0]
        print('Equivariance of input data:')
        print(torch.allclose(torch.einsum('ijkl,ai,bj,ck,dl->abcd',C,Q,Q,Q,Q), C_rot, rtol=1e-3))
        
        # equivariance of output data:
        out = out['stiffness']
        # print(out.shape)
        C = out[0]
        C_rot = out[1]

        # # equivariance of positions
        # with np.printoptions(precision=3, suppress=True):
        #     print(torch.einsum('ij,pj->pi',Q,graph.positions).numpy())
        #     print(rot_graph.positions.numpy())

        # equivariance of vectors
        # print('Equivariance of vectors')
        # print(torch.allclose(torch.einsum('ij,j->i',Q,C), C_rot, rtol=1e-3))
        # with np.printoptions(precision=3, suppress=False):
        #     print(C.numpy())
        #     print('rot')
        #     print(torch.einsum('ij,j->i',Q,C).numpy())
        #     print(C_rot.numpy())

    elif model.lower() == 'positive':
        graph = dataset[0]
        delattr(graph, 'compliance')
        graph.stiffness = graph.stiffness.float()
        rot_graph, Q = rotate_lat_cart(graph)
        data = Batch.from_data_list([graph, rot_graph])

        params = Namespace(
            lmax=3, # spherical harmonics
            hidden_irreps='32x0e+32x1o+32x2e',
            readout_irreps='16x0e+16x2e',
            interaction_reduction='sum',
            agg_norm_const=3.0,
            correlation=3,
            global_reduction='mean',
            message_passes=2,
            interaction_bias=True,
            optimizer='adamw',
            lr=1e-3,
            amsgrad=True,
            weight_decay=1e-8,
            beta1=0.9,
            epsilon=1e-8,
            scheduler=None,
            func='square'
        )
        model = PositiveGNN(params)
        with torch.no_grad():
            model.eval()
            out = model(data)
       
        # equivariance of input data:
        C = graph.stiffness[0]
        C_rot = rot_graph.stiffness[0]
        print('Equivariance of input data:')
        print(torch.allclose(torch.einsum('ijkl,ai,bj,ck,dl->abcd',C,Q,Q,Q,Q), C_rot, rtol=1e-3))
        
        # equivariance of output data:
        out = out['stiffness']
        C = out[0]
        C_rot = out[1]
        C4 = elasticity_func.stiffness_Mandel_to_cart_4(C)
        C4_rot = elasticity_func.stiffness_Mandel_to_cart_4(C_rot)
        print('Equivariance of output data:')
        print(torch.allclose(torch.einsum('ijkl,ai,bj,ck,dl->abcd',C4,Q,Q,Q,Q), C4_rot, rtol=1e-3))
        with np.printoptions(precision=3, suppress=True):
            print(elasticity_func.stiffness_cart_4_to_Mandel(C4).numpy())
            print(elasticity_func.stiffness_cart_4_to_Mandel(C4_rot).numpy())

    # with np.printoptions(precision=3, suppress=True):
    #     print(elasticity_func.stiffness_4th_order_to_Voigt(C.numpy()))
    # C = torch.einsum('ijkl,ai,bj,ck,dl->abcd',C,Q,Q,Q,Q)
    # print('Equivariance of output data:')
    # print(torch.allclose(C, C_rot, rtol=1e-2))
    # with np.printoptions(precision=3, suppress=True):
    #     print(elasticity_func.stiffness_4th_order_to_Voigt(C.numpy()))
    #     print(elasticity_func.stiffness_4th_order_to_Voigt(C_rot.numpy()))

if __name__=='__main__':
    main(sys.argv[1])