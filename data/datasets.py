import os
import os.path as osp
import sys
from typing import Callable, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)

from .lattice import Lattice
from .catalogue import Catalogue
from . import elasticity_func

def sample_uniform(
    compliance_tensors: dict,
    Npoints: int
    ):
    directions = directions_Gaussian(Npoints, 'cartesian')
    youngs_moduli = dict()
    for rel_dens in sorted(compliance_tensors):
        S = compliance_tensors[rel_dens]
        S = elasticity_func.compliance_Voigt_to_4th_order(S)
        youngs_moduli[rel_dens] = elasticity_func.Youngs_modulus(
            S, directions
            )
    exponents, constants = elasticity_func.scaling_law_fit(youngs_moduli)
    return directions, exponents, constants

def directions_Gaussian(Npoints: int, repr='angles'):
    directions = np.random.randn(Npoints,3)
    directions = directions/np.linalg.norm(directions, axis=1, keepdims=True)
    if repr=='cartesian': 
        return directions
    elif repr=='angles':
        x = directions[:,0]
        y = directions[:,1]
        z = directions[:,2]
        r = np.sqrt(np.einsum('ij,ij->i',directions,directions) - z**2)
        phi = np.arccos(z)
        th = np.arctan2(y,x)+np.pi
        return phi, th
    else:
        raise ValueError("Wrong 'repr' option")

def periodic_edge_adjacency(
    edge_adjacency: npt.NDArray, pp_list: list, 
    node_pos: Optional[npt.NDArray] = None
) -> npt.NDArray:
    periodic_partners = [list(s) for s in pp_list]
    periodic_partners = np.row_stack(periodic_partners)
    edge_map = np.sort(periodic_partners, axis=1)
    new_edge_adjacency = relabel_edges(edge_adjacency, edge_map)
    if isinstance(node_pos, np.ndarray):
        new_edge_adjacency, new_nodal_pos = get_uq_node_nums(new_edge_adjacency, node_pos)
        return new_edge_adjacency, new_nodal_pos
    else:
        new_edge_adjacency = get_uq_node_nums(new_edge_adjacency)
        return new_edge_adjacency

def get_uq_node_nums(
    edge_adjacency: npt.NDArray,
    nodal_pos: Optional[npt.NDArray] = None
) -> npt.NDArray:
    uq_node_nums = np.unique(edge_adjacency)
    edge_map = np.column_stack(
        (uq_node_nums, np.arange(len(uq_node_nums)))
        )
    new_edge_adjacency = relabel_edges(edge_adjacency, edge_map)
    if isinstance(nodal_pos, np.ndarray):
        new_nodal_pos = nodal_pos[edge_map[:,0], :]
        return new_edge_adjacency, new_nodal_pos
    else:
        return new_edge_adjacency

def relabel_edges(edges: npt.NDArray, edge_map: npt.NDArray):
    new_edges = []
    mask_to_edit = (
        (np.isin(edges[:,0], edge_map[:,0])) 
        | (np.isin(edges[:,1], edge_map[:,0]))
    )
    new_edges.extend(edges[~mask_to_edit,:])
    for i in np.flatnonzero(mask_to_edit):
        n0, n1 = edges[i]
        if n0 in edge_map[:,0]:
            n0 = edge_map[edge_map[:,0]==n0, 1]
            assert len(n0)==1
            n0 = n0[0]
        if n1 in edge_map[:,0]:
            n1 = edge_map[edge_map[:,0]==n1, 1]
            assert len(n1)==1
            n1 = n1[0]
        new_edges.append([n0,n1])
    new_edges = np.row_stack(new_edges)
    return new_edges

class GLAMMDataset(InMemoryDataset):
    r"""Lattice dataset.
    Work in progress.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`b`                        | Scaling exponent                                                                  | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`B`                        | Proportionality constant                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    """  # noqa: E501


    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 stage: str = 'train',
                 nlat: int = 20, 
                 nimperf: int = 8,
                 imperf_level: str = '0.0',
                 rot_per_lat: int = 2000, 
                 representation: str = 'pbc',
                 node_pos: bool = False,
                 node_ft: str = 'ones',
                 edge_ft: str = 'L,r,e_vec',
                 ):
        
        self.nlat = nlat
        self.imperf_level = imperf_level
        self.nimperf = nimperf
        self.ROT_PER_LAT = rot_per_lat
        self.node_pos = node_pos
        if representation in ['orig','pbc','fund_tess','fund_inner']:
            self.repr = representation
        else:
            raise ValueError(f'Representation {representation} does not exist')
        if node_ft in ['ones','exponent']:
            self.node_ft_format = node_ft
        else:
            raise ValueError(f'Node ft format `{node_ft}` not recognised')
        for key in edge_ft.split(','):
            if key not in ['L','r','e_vec','euler']:
                raise ValueError(f'Edge feature format key `{key}` not recognised')
        self.edge_ft_format = edge_ft
            

        repo_url = 'https://github.com/igrega348/lattices/raw/main/'
                
        if stage=='train':
            self.raw_compressed_name = f'cat_{nlat}lat_p{imperf_level}.lat.gz'
            self.raw_url0 = repo_url + self.raw_compressed_name
        elif stage=='test':
            self.raw_compressed_name = f'cat_test_{nlat}lat.lat.gz'
            self.raw_url0 = repo_url + self.raw_compressed_name
            assert nimperf==1
        else:
            ValueError(f'Stage `{stage}` not supported')
        self.stage = stage

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        if self.stage=='train':
            return [f'cat_{self.nlat}lat_p{self.imperf_level}.lat']
        else:
            return [f'cat_test_{self.nlat}lat.lat']

    @property
    def processed_file_names(self) -> str:
        pos = '_pos' if self.node_pos else ''
        if self.stage=='train':
            return f'data_{self.nlat}lat_{self.nimperf}imp_p{self.imperf_level}_{self.ROT_PER_LAT}rot_{self.repr}_nft-{self.node_ft_format}_eft-{self.edge_ft_format}{pos}.pt'
        else:
            return f'test_data_{self.nlat}lat_{self.nimperf}imp_{self.ROT_PER_LAT}rot_{self.repr}_nft-{self.node_ft_format}_eft-{self.edge_ft_format}{pos}.pt'
        

    def download(self):
        for url in [self.raw_url0]:
            file_path = download_url(url, self.raw_dir)
            extract_gz(osp.join(self.raw_dir, self.raw_compressed_name), self.raw_dir)

    def process(self):
        cat = Catalogue.from_file(self.raw_paths[0], 0)
        print(f'Processing catalogue of {self.nlat} lattices.'
        f' Stage {self.stage},'
        f' imperfection level {self.imperf_level},'
        f' {self.nimperf} imperfection realizations,'
        f' and {self.ROT_PER_LAT} orientations per lattice.'
        f' Representation {self.repr}.'
        f' Exporting nodal positions: {str(self.node_pos)}.'
        f' Nodal features: {self.node_ft_format}.'
        f' Edge features: {self.edge_ft_format}.'
        f' In total {self.nlat*self.nimperf*self.ROT_PER_LAT}.')

        names = cat.names
    
        data_list = []
        base_name_counts = dict()

        for i, lattice in enumerate(tqdm(names)):
            base_name = '_'.join(lattice.split('_')[:3])
            if base_name in base_name_counts:
                if base_name_counts[base_name] >= self.nimperf:
                    continue
                else:
                    base_name_counts[base_name] += 1
            else:
                base_name_counts[base_name] = 1

            lat = Lattice(**cat.get_unit_cell(lattice))
            lat.set_edge_radii(0.01, repr='transformed') # will calculate transformed coordinates
            edge_rad = lat.edge_radii.mean()

            if self.repr in ['fund_tess','fund_inner']:
                lat.calculate_fundamental_representation()
                edge_coords = lat._node_adj_to_ec(
                    lat.reduced_node_coordinates, lat.fundamental_edge_adjacency
                )
                edge_coords += lat.fundamental_tesselation_vecs
                edge_coords[:,:3] = lat.transform_coordinates(
                    edge_coords[:,:3], lat.get_transform_matrix()
                    )
                edge_coords[:,3:] = lat.transform_coordinates(
                    edge_coords[:,3:], lat.get_transform_matrix()
                    )
                if self.repr=='fund_tess':
                    nodal_pos, edge_adjacency = lat._ec_to_node_adj(edge_coords)
                else: # fund_inner
                    edge_adjacency = lat.fundamental_edge_adjacency
                    if self.node_pos:
                        edge_adjacency, nodal_pos = get_uq_node_nums(
                            edge_adjacency, lat.reduced_node_coordinates
                        )
                    else:
                        edge_adjacency = get_uq_node_nums(edge_adjacency)

                if self.node_pos:
                    nodal_pos = lat.transform_coordinates(
                        nodal_pos, lat.get_transform_matrix()
                    )

                edge_lengths = lat._edge_lengths_from_coords(edge_coords)

            else:
                pp_list = lat.get_periodic_partners()
                lat.update_representations()
                nodal_pos = lat.transformed_node_coordinates
                edge_coords = lat.transformed_edge_coordinates
                edge_lengths = lat.transformed_edge_lengths
                edge_adjacency = lat.edge_adjacency
                if self.repr=='pbc':
                    # merge periodic partners and relabel edges
                    if self.node_pos:
                        edge_adjacency, nodal_pos = periodic_edge_adjacency(
                            edge_adjacency, pp_list, nodal_pos
                        )
                    else:
                        edge_adjacency = periodic_edge_adjacency(
                            edge_adjacency, pp_list
                        )
                else:
                    pass

            edge_radii = edge_rad*np.ones(edge_coords.shape[0])
            edge_vecs = (edge_coords[:,3:] - edge_coords[:,:3])
            edge_unit_vecs = edge_vecs / np.linalg.norm(edge_vecs, axis=1, keepdims=True)


            edge_adjacency = np.row_stack(
                (edge_adjacency, edge_adjacency[:,::-1])
                ) # reverse connections

            num_uq_nodes = len(np.unique(edge_adjacency))

            if len(lat.compliance_tensors)<2: 
                print(f'Lattice {lat.name} does not have enough data')
                raise ValueError

            directions, exponents, constants = sample_uniform(
                lat.compliance_tensors, self.ROT_PER_LAT)

            x = directions[:,0]
            y = directions[:,1]
            z = directions[:,2]
            phi = np.arccos(z)
            th = np.arctan2(y,x)+np.pi
            psi = 2*np.pi*np.random.rand(self.ROT_PER_LAT)   

            for i_rot in range(self.ROT_PER_LAT):
                
                rotated_edge_vecs = lat._rotate_coordinates(
                    edge_unit_vecs, th[i_rot], phi[i_rot], psi[i_rot]
                    )
                

                edge_ft_list = []
                edge_ft_list_rev = []
                for key in self.edge_ft_format.split(','):
                    if key=='L':
                        edge_ft_list.append(edge_lengths)
                        edge_ft_list_rev.append(edge_lengths)
                    elif key=='r':
                        edge_ft_list.append(edge_radii)
                        edge_ft_list_rev.append(edge_radii)
                    elif key=='e_vec':
                        edge_ft_list.append(rotated_edge_vecs)
                        edge_ft_list_rev.append(-rotated_edge_vecs)
                    elif key=='euler':
                        v = rotated_edge_vecs
                        _phi = np.arccos(v[:,2])
                        _th = np.arctan2(v[:,1],v[:,0])+np.pi
                        edge_ft_list.append(np.column_stack(_phi, _th))
                        v = -rotated_edge_vecs
                        _phi = np.arccos(v[:,2])
                        _th = np.arctan2(v[:,1],v[:,0])+np.pi
                        edge_ft_list_rev.append(np.column_stack(_phi, _th))
                    else:
                        raise ValueError(f'Unrecognised edge format string `{key}`')

                edge_features = np.column_stack(edge_ft_list)
                # reverse connections
                edge_features_rev = np.column_stack(edge_ft_list_rev)
                edge_features = np.row_stack((edge_features, edge_features_rev))
                    
                exponent = exponents[i_rot]
                constant = constants[i_rot]

                if self.node_ft_format=='ones':
                    nodal_features = np.ones(num_uq_nodes)
                elif self.node_ft_format=='exponent':
                    nodal_features = exponent*np.ones(num_uq_nodes)

                if self.node_pos:
                    rotated_nodal_pos = lat._rotate_coordinates(
                        nodal_pos, th[i_rot], phi[i_rot], psi[i_rot]
                    )
                    pos = torch.tensor(rotated_nodal_pos, dtype=torch.float)
                else:
                    pos = None
                nodal_ft = torch.tensor(
                    nodal_features.reshape((-1,1)), dtype=torch.float
                    )
                edge_ft = torch.tensor(edge_features, dtype=torch.float)
                edge_adj = torch.tensor(edge_adjacency.T, dtype=torch.long)
                graph_ft = torch.tensor([exponent, constant]).unsqueeze(0)
                

                data = Data(
                    x=nodal_ft, 
                    edge_index=edge_adj, 
                    edge_attr=edge_ft, 
                    y=graph_ft,
                    pos=pos,
                    th=th[i_rot],
                    phi=phi[i_rot],
                    psi=psi[i_rot],
                    name=lat.name
                    )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
      
        if len(data_list)<1:
            raise RuntimeError('Empty data list')
        else:
            torch.save(self.collate(data_list), self.processed_paths[0])




class GLAMM_tens_Dataset(InMemoryDataset):
    r"""Lattice dataset.
    Work in progress.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`S`                        | Compliance tensor                                                                  | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
  
    """  # noqa: E501


    def __init__(self, root: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 stage: str = 'train',
                 nlat: int = 20, 
                 nimperf: int = 8,
                 imperf_level: str = '0.0',
                 representation: str = 'pbc',
                 node_pos: bool = False,
                 node_ft: str = 'ones',
                 edge_ft: str = 'L,r,e_vec',
                 graph_ft_format: str = 'Voigt',
                 rel_dens: float = 0.01,
                 ):
        
        self.nlat = nlat
        self.imperf_level = imperf_level
        self.nimperf = nimperf
        self.node_pos = node_pos
        self.graph_ft_format = graph_ft_format
        self.rel_dens = rel_dens
       
        if representation in ['orig','pbc','fund_tess','fund_inner']:
            self.repr = representation
        else:
            raise ValueError(f'Representation {representation} does not exist')
        if node_ft in ['ones','exponent']:
            self.node_ft_format = node_ft
        else:
            raise ValueError(f'Node ft format `{node_ft}` not recognised')
        for key in edge_ft.split(','):
            if key not in ['L','r','e_vec','euler']:
                raise ValueError(f'Edge feature format key `{key}` not recognised')
        self.edge_ft_format = edge_ft
            

        repo_url = 'https://github.com/igrega348/lattices/raw/main/'
                
        if stage=='train':
            self.raw_compressed_name = f'cat_{nlat}lat_p{imperf_level}.lat.gz'
            self.raw_url0 = repo_url + self.raw_compressed_name
        elif stage=='test':
            self.raw_compressed_name = f'cat_test_{nlat}lat.lat.gz'
            self.raw_url0 = repo_url + self.raw_compressed_name
            assert nimperf==1
        else:
            ValueError(f'Stage `{stage}` not supported')
        self.stage = stage

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        if self.stage=='train':
            return [f'cat_{self.nlat}lat_p{self.imperf_level}.lat']
        else:
            return [f'cat_test_{self.nlat}lat.lat']

    @property
    def processed_file_names(self) -> str:
        pos = '_pos' if self.node_pos else ''
        if self.stage=='train':
            return f'data_{self.nlat}lat_{self.nimperf}imp_p{self.imperf_level}_{self.repr}_nft-{self.node_ft_format}_eft-{self.edge_ft_format}{pos}_{self.graph_ft_format}_rho-{self.rel_dens}.pt'
        else:
            return f'test_data_{self.nlat}lat_{self.nimperf}imp_{self.repr}_nft-{self.node_ft_format}_eft-{self.edge_ft_format}{pos}_{self.graph_ft_format}_rho-{self.rel_dens}.pt'
        

    def download(self):
        for url in [self.raw_url0]:
            file_path = download_url(url, self.raw_dir)
            extract_gz(osp.join(self.raw_dir, self.raw_compressed_name), self.raw_dir)

    def process(self):
        cat = Catalogue.from_file(self.raw_paths[0], 0)
        print(f'Processing catalogue of {self.nlat} lattices.'
        f' Stage {self.stage},'
        f' imperfection level {self.imperf_level},'
        f' {self.nimperf} imperfection realizations,'
        f' Representation {self.repr}.'
        f' Exporting nodal positions: {str(self.node_pos)}.'
        f' Nodal features: {self.node_ft_format}.'
        f' Edge features: {self.edge_ft_format}.'
        f' In total {self.nlat*self.nimperf}.'
        f' Graph feature format {self.graph_ft_format}.'
        f' Relative density {self.rel_dens}')

        names = cat.names
    
        data_list = []
        base_name_counts = dict()

        for i, lattice in enumerate(tqdm(names)):
            base_name = '_'.join(lattice.split('_')[:3])
            if base_name in base_name_counts:
                if base_name_counts[base_name] >= self.nimperf:
                    continue
                else:
                    base_name_counts[base_name] += 1
            else:
                base_name_counts[base_name] = 1

            lat = Lattice(**cat.get_unit_cell(lattice))
            lat.set_edge_radii(self.rel_dens, repr='transformed') # will calculate transformed coordinates
            edge_rad = lat.edge_radii.mean()
            tessellation_vecs = np.zeros((lat.edge_radii.shape[0], 3))
            unit_shifts = np.zeros_like(tessellation_vecs).astype(int)

            if self.repr in ['fund_tess','fund_inner']:
                lat.calculate_fundamental_representation()
                edge_coords = lat._node_adj_to_ec(
                    lat.reduced_node_coordinates, lat.fundamental_edge_adjacency
                )
                assert lat.fundamental_tesselation_vecs[:,:3].sum()==0
                tessellation_vecs = lat.fundamental_tesselation_vecs[:,3:]
                unit_shifts = np.zeros_like(tessellation_vecs).astype(int)
                unit_shifts[tessellation_vecs!=0] = np.sign(tessellation_vecs[tessellation_vecs!=0])
                tessellation_vecs = lat.transform_coordinates(
                    tessellation_vecs, lat.get_transform_matrix()
                )
                edge_coords += lat.fundamental_tesselation_vecs
                edge_coords[:,:3] = lat.transform_coordinates(
                    edge_coords[:,:3], lat.get_transform_matrix()
                    )
                edge_coords[:,3:] = lat.transform_coordinates(
                    edge_coords[:,3:], lat.get_transform_matrix()
                    )
                if self.repr=='fund_tess':
                    nodal_pos, edge_adjacency = lat._ec_to_node_adj(edge_coords)
                else: # fund_inner
                    edge_adjacency = lat.fundamental_edge_adjacency
                    if self.node_pos:
                        edge_adjacency, nodal_pos = get_uq_node_nums(
                            edge_adjacency, lat.reduced_node_coordinates
                        )
                    else:
                        edge_adjacency = get_uq_node_nums(edge_adjacency)

                if self.node_pos:
                    nodal_pos = lat.transform_coordinates(
                        nodal_pos, lat.get_transform_matrix()
                    )

                edge_lengths = lat._edge_lengths_from_coords(edge_coords)

            else:
                pp_list = lat.get_periodic_partners()
                lat.update_representations()
                nodal_pos = lat.transformed_node_coordinates
                edge_coords = lat.transformed_edge_coordinates
                edge_lengths = lat.transformed_edge_lengths
                edge_adjacency = lat.edge_adjacency
                if self.repr=='pbc':
                    # merge periodic partners and relabel edges
                    if self.node_pos:
                        edge_adjacency, nodal_pos = periodic_edge_adjacency(
                            edge_adjacency, pp_list, nodal_pos
                        )
                    else:
                        edge_adjacency = periodic_edge_adjacency(
                            edge_adjacency, pp_list
                        )
                else:
                    pass

            edge_radii = edge_rad*np.ones(edge_coords.shape[0])
            edge_vecs = (edge_coords[:,3:] - edge_coords[:,:3])
            edge_unit_vecs = edge_vecs / np.linalg.norm(edge_vecs, axis=1, keepdims=True)


            edge_adjacency = np.row_stack(
                (edge_adjacency, edge_adjacency[:,::-1])
                ) # reverse connections
            unit_shifts = np.row_stack(
                (unit_shifts, -unit_shifts)
            )
            tessellation_vecs = np.row_stack(
                (tessellation_vecs, -tessellation_vecs)
            )

            num_uq_nodes = len(np.unique(edge_adjacency))

            if len(lat.compliance_tensors)<2: 
                print(f'Lattice {lat.name} does not have enough data')
                raise ValueError

            compliance = lat.compliance_tensors[self.rel_dens]
            stiffness = np.linalg.inv(compliance)
            if self.graph_ft_format=='Voigt':
                stiffness = torch.from_numpy(stiffness)
                compliance = torch.from_numpy(compliance)
            elif (self.graph_ft_format=='cartesian_4'):    
                compliance = elasticity_func.compliance_Voigt_to_4th_order(compliance)
                stiffness = elasticity_func.stiffness_Voigt_to_4th_order(stiffness)
                compliance = torch.from_numpy(compliance)
                stiffness = torch.from_numpy(stiffness)

                
            edge_ft_list = []
            edge_ft_list_rev = []
            for key in self.edge_ft_format.split(','):
                if key=='L':
                    edge_ft_list.append(edge_lengths)
                    edge_ft_list_rev.append(edge_lengths)
                elif key=='r':
                    edge_ft_list.append(edge_radii)
                    edge_ft_list_rev.append(edge_radii)
                elif key=='e_vec':
                    edge_ft_list.append(edge_unit_vecs)
                    edge_ft_list_rev.append(-edge_unit_vecs)
                elif key=='euler':
                    v = edge_unit_vecs
                    _phi = np.arccos(v[:,2])
                    _th = np.arctan2(v[:,1],v[:,0])+np.pi
                    edge_ft_list.append(np.column_stack(_phi, _th))
                    v = -edge_unit_vecs
                    _phi = np.arccos(v[:,2])
                    _th = np.arctan2(v[:,1],v[:,0])+np.pi
                    edge_ft_list_rev.append(np.column_stack(_phi, _th))
                else:
                    raise ValueError(f'Unrecognised edge format string `{key}`')

            edge_features = np.column_stack(edge_ft_list)
            # reverse connections
            edge_features_rev = np.column_stack(edge_ft_list_rev)
            edge_features = np.row_stack((edge_features, edge_features_rev))

            if self.node_ft_format=='ones':
                nodal_features = np.ones(num_uq_nodes)
           
            if self.node_pos:
                pos = torch.tensor(nodal_pos, dtype=torch.float)
            else:
                pos = None
            nodal_ft = torch.tensor(
                nodal_features.reshape((-1,1)), dtype=torch.float
                )
            edge_ft = torch.tensor(edge_features, dtype=torch.float)
            edge_adj = torch.tensor(edge_adjacency.T, dtype=torch.long)
            # graph_ft = torch.from_numpy(compliance)
            

            data = Data(
                node_attrs=nodal_ft, 
                edge_index=edge_adj, 
                edge_attr=edge_ft, 
                shifts = torch.tensor(tessellation_vecs, dtype=torch.float),
                unit_shifts = torch.tensor(unit_shifts, dtype=torch.long),
                stiffness=stiffness.unsqueeze(0),
                compliance=compliance.unsqueeze(0),
                rel_dens=self.rel_dens,
                positions=pos,
                name=lat.name
                )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
    
        if len(data_list)<1:
            raise RuntimeError('Empty data list')
        else:
            torch.save(self.collate(data_list), self.processed_paths[0])


class GLAMM_rhotens_Dataset(InMemoryDataset):
    r"""Lattice dataset.
    Work in progress.

    """  # noqa: E501


    def __init__(self, root: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 stage: str = 'train',
                 nlat: int = 20, 
                 nimperf: int = 8,
                 imperf_level: str = '0.0',
                 representation: str = 'pbc',
                 node_pos: bool = False,
                 node_ft: str = 'ones',
                 edge_ft: str = 'r',
                 graph_ft_format: str = 'Voigt',
                 n_reldens: Union[int, slice] = 1,
                 ):
        
        self.nlat = nlat
        self.imperf_level = imperf_level
        self.nimperf = nimperf
        self.node_pos = node_pos
        self.graph_ft_format = graph_ft_format
        # self.rel_dens = rel_dens
        if isinstance(n_reldens, int):
            self.reldens_slice = slice(n_reldens+1)
        elif isinstance(n_reldens, slice):
            self.reldens_slice = n_reldens
       
        if representation in ['orig','pbc','fund_tess','fund_inner']:
            self.repr = representation
        else:
            raise ValueError(f'Representation {representation} does not exist')
        if node_ft in ['ones']:
            self.node_ft_format = node_ft
        else:
            raise ValueError(f'Node ft format `{node_ft}` not recognised')
        for key in edge_ft.split(','):
            if key not in ['L','r','e_vec','euler']:
                raise ValueError(f'Edge feature format key `{key}` not recognised')
        self.edge_ft_format = edge_ft
            

        repo_url = 'https://github.com/igrega348/lattices/raw/main/'
                
        if stage=='train':
            self.raw_compressed_name = f'cat_{nlat}lat_reldens_p{imperf_level}.lat.gz'
            self.raw_url0 = repo_url + self.raw_compressed_name
        elif stage=='test':
            self.raw_compressed_name = f'cat_test_{nlat}lat.lat.gz'
            self.raw_url0 = repo_url + self.raw_compressed_name
            assert nimperf==1
        else:
            ValueError(f'Stage `{stage}` not supported')
        self.stage = stage

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        if self.stage=='train':
            return [f'cat_{self.nlat}lat_reldens_p{self.imperf_level}.lat']
        else:
            return [f'cat_test_{self.nlat}lat.lat']

    @property
    def processed_file_names(self) -> str:
        pos = '_pos' if self.node_pos else ''
        if self.stage=='train':
            return f'data_{self.nlat}lat_{self.nimperf}imp_p{self.imperf_level}_{self.repr}_nft-{self.node_ft_format}_eft-{self.edge_ft_format}{pos}_{self.graph_ft_format}_nreldens-{self.reldens_slice}.pt'
        else:
            return f'test_data_{self.nlat}lat_{self.nimperf}imp_{self.repr}_nft-{self.node_ft_format}_eft-{self.edge_ft_format}{pos}_{self.graph_ft_format}_nreldens-{self.reldens_slice}.pt'
        

    def download(self):
        for url in [self.raw_url0]:
            file_path = download_url(url, self.raw_dir)
            extract_gz(osp.join(self.raw_dir, self.raw_compressed_name), self.raw_dir)

    def process(self):
        cat = Catalogue.from_file(self.raw_paths[0], 0)
        print(f'Processing catalogue of {self.nlat} lattices.'
        f' Stage {self.stage},'
        f' imperfection level {self.imperf_level},'
        f' {self.nimperf} imperfection realizations,'
        f' Representation {self.repr}.'
        f' Exporting nodal positions: {str(self.node_pos)}.'
        f' Nodal features: {self.node_ft_format}.'
        f' Edge features: {self.edge_ft_format}.'
        f' In total {self.nlat*self.nimperf}.'
        f' Graph feature format {self.graph_ft_format}.'
        )

        names = cat.names
    
        data_list = []
        base_name_counts = dict()

        for i, lattice in enumerate(tqdm(names)):
            base_name = '_'.join(lattice.split('_')[:3])
            if base_name in base_name_counts:
                if base_name_counts[base_name] >= self.nimperf:
                    continue
                else:
                    base_name_counts[base_name] += 1
            else:
                base_name_counts[base_name] = 1

            lat = Lattice(**cat.get_unit_cell(lattice))
            
            tessellation_vecs = np.zeros((lat.edge_adjacency.shape[0], 3))
            unit_shifts = np.zeros_like(tessellation_vecs).astype(int)

            if self.repr in ['fund_tess','fund_inner']:
                lat.calculate_fundamental_representation()
                edge_coords = lat._node_adj_to_ec(
                    lat.reduced_node_coordinates, lat.fundamental_edge_adjacency
                )
                assert lat.fundamental_tesselation_vecs[:,:3].sum()==0
                tessellation_vecs = lat.fundamental_tesselation_vecs[:,3:]
                unit_shifts = np.zeros_like(tessellation_vecs).astype(int)
                unit_shifts[tessellation_vecs!=0] = np.sign(tessellation_vecs[tessellation_vecs!=0])
                tessellation_vecs = lat.transform_coordinates(
                    tessellation_vecs, lat.get_transform_matrix()
                )
                edge_coords += lat.fundamental_tesselation_vecs
                edge_coords[:,:3] = lat.transform_coordinates(
                    edge_coords[:,:3], lat.get_transform_matrix()
                    )
                edge_coords[:,3:] = lat.transform_coordinates(
                    edge_coords[:,3:], lat.get_transform_matrix()
                    )
                if self.repr=='fund_tess':
                    nodal_pos, edge_adjacency = lat._ec_to_node_adj(edge_coords)
                else: # fund_inner
                    edge_adjacency = lat.fundamental_edge_adjacency
                    if self.node_pos:
                        edge_adjacency, nodal_pos = get_uq_node_nums(
                            edge_adjacency, lat.reduced_node_coordinates
                        )
                    else:
                        edge_adjacency = get_uq_node_nums(edge_adjacency)

                if self.node_pos:
                    nodal_pos = lat.transform_coordinates(
                        nodal_pos, lat.get_transform_matrix()
                    )

                edge_lengths = lat._edge_lengths_from_coords(edge_coords)

            else:
                pp_list = lat.get_periodic_partners()
                lat.update_representations()
                nodal_pos = lat.transformed_node_coordinates
                edge_coords = lat.transformed_edge_coordinates
                edge_lengths = lat.transformed_edge_lengths
                edge_adjacency = lat.edge_adjacency
                if self.repr=='pbc':
                    # merge periodic partners and relabel edges
                    if self.node_pos:
                        edge_adjacency, nodal_pos = periodic_edge_adjacency(
                            edge_adjacency, pp_list, nodal_pos
                        )
                    else:
                        edge_adjacency = periodic_edge_adjacency(
                            edge_adjacency, pp_list
                        )
                else:
                    pass

           
            edge_vecs = (edge_coords[:,3:] - edge_coords[:,:3])
            edge_unit_vecs = edge_vecs / np.linalg.norm(edge_vecs, axis=1, keepdims=True)


            edge_adjacency = np.row_stack(
                (edge_adjacency, edge_adjacency[:,::-1])
                ) # reverse connections
            unit_shifts = np.row_stack(
                (unit_shifts, -unit_shifts)
            )
            tessellation_vecs = np.row_stack(
                (tessellation_vecs, -tessellation_vecs)
            )

            num_uq_nodes = len(np.unique(edge_adjacency))

            if len(lat.compliance_tensors)<2: 
                print(f'Lattice {lat.name} does not have enough data')
                raise ValueError

            avail_reldens = list(lat.compliance_tensors.keys())
            for rel_dens in avail_reldens[self.reldens_slice]:

                lat.set_edge_radii(rel_dens, repr='transformed') # will calculate transformed coordinates
                edge_rad = lat.edge_radii.mean()
                edge_radii = edge_rad*np.ones(edge_coords.shape[0])

                compliance = lat.compliance_tensors[rel_dens]
                stiffness = np.linalg.inv(compliance)
                if self.graph_ft_format=='Voigt':
                    stiffness = torch.from_numpy(stiffness).unsqueeze(0)
                    compliance = torch.from_numpy(compliance).unsqueeze(0)
                elif (self.graph_ft_format=='cartesian_4'):    
                    compliance = elasticity_func.compliance_Voigt_to_4th_order(compliance)
                    stiffness = elasticity_func.stiffness_Voigt_to_4th_order(stiffness)
                    compliance = torch.from_numpy(compliance).unsqueeze(0)
                    stiffness = torch.from_numpy(stiffness).unsqueeze(0)

                
                edge_ft_list = []
                edge_ft_list_rev = []
                for key in self.edge_ft_format.split(','):
                    if key=='L':
                        edge_ft_list.append(edge_lengths)
                        edge_ft_list_rev.append(edge_lengths)
                    elif key=='r':
                        edge_ft_list.append(edge_radii)
                        edge_ft_list_rev.append(edge_radii)
                    elif key=='e_vec':
                        edge_ft_list.append(edge_unit_vecs)
                        edge_ft_list_rev.append(-edge_unit_vecs)
                    elif key=='euler':
                        v = edge_unit_vecs
                        _phi = np.arccos(v[:,2])
                        _th = np.arctan2(v[:,1],v[:,0])+np.pi
                        edge_ft_list.append(np.column_stack(_phi, _th))
                        v = -edge_unit_vecs
                        _phi = np.arccos(v[:,2])
                        _th = np.arctan2(v[:,1],v[:,0])+np.pi
                        edge_ft_list_rev.append(np.column_stack(_phi, _th))
                    else:
                        raise ValueError(f'Unrecognised edge format string `{key}`')

                edge_features = np.column_stack(edge_ft_list)
                # reverse connections
                edge_features_rev = np.column_stack(edge_ft_list_rev)
                edge_features = np.row_stack((edge_features, edge_features_rev))

            
                if self.node_pos:
                    pos = torch.tensor(nodal_pos, dtype=torch.float)
                else:
                    pos = None
                
                edge_ft = torch.tensor(edge_features, dtype=torch.float)
                edge_adj = torch.tensor(edge_adjacency.T, dtype=torch.long)
                # graph_ft = torch.from_numpy(compliance)
            

            
                if self.node_ft_format=='ones':
                    nodal_features = np.ones(num_uq_nodes)
                    nodal_ft = torch.tensor(
                        nodal_features.reshape((-1,1)), dtype=torch.float
                        )

                data = Data(
                    node_attrs=nodal_ft, 
                    edge_index=edge_adj, 
                    edge_attr=edge_ft, 
                    shifts = torch.tensor(tessellation_vecs, dtype=torch.float),
                    unit_shifts = torch.tensor(unit_shifts, dtype=torch.long),
                    rel_dens=rel_dens,
                    stiffness=stiffness,
                    compliance=compliance,
                    positions=pos,
                    name=lat.name
                    )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
    
        if len(data_list)<1:
            raise RuntimeError('Empty data list')
        else:
            torch.save(self.collate(data_list), self.processed_paths[0])