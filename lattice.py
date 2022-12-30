import copy
from collections.abc import Sequence, Iterable
import numpy as np
import numpy.typing as npt
from math import ceil
from scipy.spatial import transform

class Lattice:
    TOL_DIST: float = 1e-5
    TOL_ANGLE: float = 1e-5
    # book-keeping variables
    name: str
    code: str
    # topological representations
    reduced_node_coordinates: npt.NDArray[np.float_]
    edge_adjacency: npt.NDArray[np.int_]
    reduced_edge_coordinates: npt.NDArray[np.float_]
    transformed_node_coordinates: npt.NDArray[np.float_]
    transformed_edge_coordinates: npt.NDArray[np.float_]
    fundamental_edge_adjacency: npt.NDArray[np.int_]
    fundamental_tesselation_vecs: npt.NDArray[np.float_]
    num_nodes: int
    num_edges: int
    num_fundamental_edges: int
    nodal_connectivity: npt.NDArray[np.int_]
    node_types: dict[str, set[int]]
    periodic_partners: list[set[int]]
    reduced_edge_lengths: npt.NDArray[np.float_]
    transformed_edge_lengths: npt.NDArray[np.float_]
    fundamental_edge_lengths: npt.NDArray[np.float_]
    # elasticity properties
    S_tens: npt.NDArray[np.float_]
    compliance_tensors: dict[float, npt.NDArray[np.float_]]
    Youngs_moduli: dict[str, float]
    scaling_exponents: dict[str, float]
    # other properties
    lattice_constants: npt.NDArray[np.float_]
    uc_volume: float
    rel_dens: float
    edge_radii: npt.NDArray[np.float_]    

    def __init__(
            self,*, name=None, code=None, lattice_constants=None,
            nodal_positions=None, edge_adjacency=None, 
            edge_coordinates=None, **kwargs
            ) -> None:
        """
        Construct lattice unit cell

        Takes in keyword-only arguments. 
        Intended for three ways of initialisation:
        - by unpacking the catalogue dictionary
        - by manually specifying node coordinates and edge adjacency
        - by manually specifying edge coordinates
        """
        if isinstance(name, str):
            self.name = name
        if isinstance(code, str):
            self.code = code

        if 'reduced_node_coordinates' in kwargs:
            nodal_positions = kwargs['reduced_node_coordinates']
        if isinstance(nodal_positions, Sequence):
            nodal_positions = np.array(nodal_positions)
        if isinstance(edge_adjacency, Sequence):
            edge_adjacency = np.array(edge_adjacency)
        if isinstance(edge_coordinates, Sequence):
            edge_coordinates = np.array(edge_coordinates)
        
        if (isinstance(lattice_constants, Sequence)
            or isinstance(lattice_constants, np.ndarray)):
            self.lattice_constants = np.array(lattice_constants, dtype=float)
        
        if 'compliance_tensors' in kwargs:
            compliance_tensors_flat = kwargs['compliance_tensors']
            compliance_tensors = dict()
            for rel_dens in compliance_tensors_flat:
                nums = compliance_tensors_flat[rel_dens]
                assert len(nums)==21
                S = np.zeros((6,6))
                S[np.triu_indices(6)] = nums
                S[np.triu_indices(6)[::-1]] = nums
                compliance_tensors[rel_dens] = S
            self.compliance_tensors = compliance_tensors

        if 'Youngs_moduli' in kwargs:
            self.Youngs_moduli = kwargs['Youngs_moduli']
        if 'scaling_exponents' in kwargs:
            self.scaling_exponents = kwargs['scaling_exponents']


        if (isinstance(nodal_positions, np.ndarray)
            and isinstance(edge_adjacency, np.ndarray) 
            and not isinstance(edge_coordinates, np.ndarray)):
            # either node coordinates and edge adjacency are specified            
            nodes = np.atleast_2d(nodal_positions).astype(np.float_)
            assert nodes.shape[1]==3
            self.num_nodes = nodes.shape[0]
            self.reduced_node_coordinates = nodes
            edges = np.atleast_2d(edge_adjacency)
            assert edges.shape[1]==2
            assert edges.min()>=0 # ensure 0-based indexing
            assert edges.max()<nodes.shape[0]
            edges = np.sort(edges, axis=1)
            edges = edges[np.argsort(edges[:,0]), :]
            self.num_edges = edges.shape[0]
            assert edges.max()<self.num_nodes
            self.edge_adjacency = edges
        elif (isinstance(edge_coordinates, np.ndarray) 
            and not isinstance(nodal_positions, np.ndarray) 
            and not isinstance(edge_adjacency, np.ndarray)):
            edge_coordinates = np.atleast_2d(edge_coordinates)
            assert edge_coordinates.shape[1]==6
            self.reduced_edge_coordinates = edge_coordinates
            self.num_edges = edge_coordinates.shape[0]
        else:
            raise NotImplementedError(
                'Lattice cell can be initialised from either '\
                'nodal positions and edge adjacency, or ' \
                'from edge coordinates'
            )

    def __repr__(self) -> str:
        repr_dict = {}
        if hasattr(self, 'name'):
            repr_dict['name'] = self.name
        if hasattr(self, 'num_nodes'):
            repr_dict['num_nodes'] = self.num_nodes
        if hasattr(self, 'num_edges'):
            repr_dict['num_edges'] = self.num_edges
        return repr_dict.__repr__()


    def get_UC_volume(self) -> float:
        """
        Calculate unit cell volume from internally-stored crystal data
        """
        crys_data = self.lattice_constants
        
        a = crys_data[0]
        b = crys_data[1]
        c = crys_data[2]
        alpha = crys_data[3] * np.pi/180 # in radians
        beta = crys_data[4] * np.pi/180
        gamma = crys_data[5] * np.pi/180
        
        term = (1.0 
                - (np.cos(alpha))**2.0 
                - (np.cos(beta))**2.0 
                - (np.cos(gamma))**2.0)
        term = term + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        omega = a*b*c* np.sqrt( term )  
        self.uc_volume = omega
        return omega

    def get_transform_matrix(self) -> npt.NDArray[np.float_]:
        """
        Assemble transformation matrix from crystal data.

        Formula is in the Appendix to the PNAS paper:
        Lumpe, T. S. and Stankovic, T. (2020)
        https://www.pnas.org/doi/10.1073/pnas.2003504118.
        """
        crys_data = self.lattice_constants
        a = crys_data[0]
        b = crys_data[1]
        c = crys_data[2]
        alpha = crys_data[3] * np.pi/180 # in radians
        beta = crys_data[4] * np.pi/180
        gamma = crys_data[5] * np.pi/180
        
        if hasattr(self, 'uc_volume'):
            omega = self.uc_volume
        else:
            omega = self.get_UC_volume()
        
        transform_mat = np.zeros((3,3))
        transform_mat[0][0] = a
        transform_mat[0][1] = b * np.cos(gamma)
        transform_mat[0][2] = c * np.cos(beta)
        transform_mat[1][0] = 0
        transform_mat[1][1] = b * np.sin(gamma)
        transform_mat[1][2] = c * ((np.cos(alpha) 
                                - (np.cos(beta)*np.cos(gamma)))
                                /(np.sin(gamma)))
        transform_mat[2][0] = 0
        transform_mat[2][1] = 0
        transform_mat[2][2] = ( omega / ( a*b*np.sin(gamma) ) )

        return transform_mat

    def calculate_transformed_coordinates(self) -> None:
        """Transform the reduced unit cell coordinates."""
        nodes_in = self.reduced_node_coordinates
        transform_mat = self.get_transform_matrix()

        nodes_out = np.matmul( transform_mat, np.transpose(nodes_in) )
        
        self.transformed_node_coordinates = np.transpose( nodes_out )

    def _rotate_coordinates(
        self, coordinates: npt.NDArray[np.float_], 
        th: float, phi: float, psi: float
        ) -> npt.NDArray[np.float_]:
        """
        Rotate nodal coordinates or edge vectors.

        Parameters:
            - coordinates: (N,3) array of coordinates
            - th: azimuth = angle around Z-axis
            - phi: inclination = angle from Z-axis
            - psi: spin = angle about loading axis
        """

        Q = transform.Rotation.from_euler(
            'ZYX', np.array([th, phi-np.pi/2, psi])
            )
    
        transformed_coords = Q.apply(coordinates, inverse=True)
        
        return transformed_coords



    def set_edge_radii(self, rel_dens: float, repr: str = 'transformed'):
        """
        Set edge radii according to relative density.

        Params:
            - rel_dens: relative density
            - repr: 'reduced' or 'transformed'
        """
        self.rel_dens = rel_dens
        if repr=='reduced':
            edge_lengths = self.calculate_edge_lengths(repr='reduced')
        elif repr=='transformed':
            edge_lengths = self.calculate_edge_lengths(repr='transformed')
        else:
            raise NotImplementedError
        sum_edge_lengths = edge_lengths.sum()
        if not hasattr(self, 'uc_volume'):
            uc_vol = self.get_UC_volume()
        else:
            uc_vol = self.uc_volume

        edge_radius = np.sqrt(rel_dens*uc_vol/(sum_edge_lengths * np.pi))
        edge_radii = edge_radius*np.ones_like(edge_lengths)
        self.edge_radii = edge_radii
        return edge_radii
        
    def refine_mesh(self, min_length: float, min_div: int) -> None:
        """
        Split edges into at least 'min_div' segments per edge
        and with each segment having length at least 'min_length'
        """
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        new_nodes = []
        new_edges = []
        new_edge_radii = []
        new_nodes.extend(nodes)
        for i_edge, e in enumerate(edges):
            n0 = e[0]
            x0 = nodes[n0]
            e_vec = nodes[e[1]] - nodes[e[0]]
            e_norm = np.linalg.norm(e_vec)
            e_unit = e_vec/e_norm
            num_div = max(min_div, ceil(e_norm/min_length))
            L_step = e_norm/num_div
            for i in range(num_div-1):
                x1 = x0 + e_unit*L_step
                n1 = len(new_nodes)
                new_nodes.append(x1)
                new_edges.append([n0,n1])
                if hasattr(self, 'edge_radii'):
                    new_edge_radii.append(self.edge_radii[i_edge])
                n0 = n1
                x0 = x1
            x1 = nodes[e[1]]
            n1 = e[1]
            new_edges.append([n0,n1])
            if hasattr(self, 'edge_radii'):
                new_edge_radii.append(self.edge_radii[i_edge])
        
        self.reduced_node_coordinates = np.row_stack(new_nodes)
        self.edge_adjacency = np.row_stack(new_edges)
        if hasattr(self, 'edge_radii'):
            self.edge_radii = np.array(new_edge_radii)
        self.update_representations()

    def merge_nonunique_nodes(self) -> None:
        """
        Merge nodes with identical coordinates
        and collapse self-incident edges
        """
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        uq_nodes, uq_inv = np.unique(nodes, axis=0, return_inverse=True)
        edges = uq_inv[edges]

        mask_self_edge = edges[:,0]==edges[:,1]
        num_self = np.count_nonzero(mask_self_edge)
        if num_self:
            print(f'Merging {num_self} self-incident edges')
        edges = edges[~mask_self_edge]


        self.reduced_node_coordinates = uq_nodes
        self.edge_adjacency = edges
        self.update_representations()

    def remove_duplicate_edges_adjacency(self) -> None:
        """
        Remove duplicate edges.

        Operates on the adjacency representation.
        Sort edges first
        """
        edges = self.edge_adjacency
        edges = np.sort(edges, axis=1)
        uq_edges = np.unique(edges, axis=0)        
        self.edge_adjacency = uq_edges
        self.update_representations()

    def remove_duplicate_edges_nodes(self) -> None:
        """
        Remove edges which are on top of each other.

        Operates on edge coordinate representation.
        To deal with machine precision, round all nodal coordinates
        to a specific number of decimal places.
        """
        NDECIMALS = 5
        if not hasattr(self, 'reduced_edge_coordinates'):
            self.node_adjacency_to_edge_coordinates()
        edge_coords = self.reduced_edge_coordinates
        edge_coords = np.around(edge_coords, decimals=NDECIMALS)
        nodes, edges = self._ec_to_node_adj(edge_coords)
        # sort each row and remove duplicates
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        
        ### Remove self-connecting edges
        mask_self_connecting = edges[:,0]==edges[:,1]
        edges = edges[~mask_self_connecting]
        # arrived at unique node-adjacency representation

        self.reduced_node_coordinates = nodes
        self.edge_adjacency = edges
        self.update_representations()

    def update_representations(
        self, basis: str = 'reduced_adjacency'
        ) -> None:
        """
        Propagate one format of representation 
        to all other available formats.

        Implemented options for basis:
        - 'reduced_adjacency'
        - 'reduced_edge_coords'

        Propagates to nodal coordinates, edge adjacency,
        edge coordinates, number of nodes/edges.
        """
        if basis=='reduced_adjacency':
            nodes = self.reduced_node_coordinates
            edges = self.edge_adjacency
            edges.sort(axis=1)
            sorting_inds = np.argsort(edges[:,0])
            edges = edges[sorting_inds,:]
            if hasattr(self, 'edge_radii'):
                self.edge_radii = self.edge_radii[sorting_inds]
            self.edge_adjacency = edges
            self.num_nodes = nodes.shape[0]
            self.num_edges = edges.shape[0]
            edge_coords = self._node_adj_to_ec(nodes, edges)
            self.reduced_edge_coordinates = edge_coords
        elif basis=='reduced_edge_coords':
            edge_coords = self.reduced_edge_coordinates
            nodes, edges = self._ec_to_node_adj(edge_coords)
            edges.sort(axis=1)
            sorting_inds = np.argsort(edges[:,0])
            edges = edges[sorting_inds,:]
            if hasattr(self, 'edge_radii'):
                self.edge_radii = self.edge_radii[sorting_inds]
            self.edge_adjacency = edges
            self.reduced_node_coordinates = nodes
            self.edge_adjacency = edges
            self.num_nodes = nodes.shape[0]
            self.num_edges = edges.shape[0]
        else:
            raise NotImplementedError('Wrong basis')

        if hasattr(self, 'transformed_node_coordinates'):
            self.calculate_transformed_coordinates()
            nodes = self.transformed_node_coordinates
            edge_coords = self._node_adj_to_ec(nodes, edges)
            self.transformed_edge_coordinates = edge_coords
        if hasattr(self, 'fundamental_edge_adjacency'):
            self.calculate_fundamental_representation()

    def node_adjacency_to_edge_coordinates(
            self, which : str = 'reduced'
            ) -> None:
        """
        Calcualate the edge coordinate representation of the lattice
        and propagate the information to all available representations

        which: representation to use as basis
        - 'reduced' 
        - 'transformed': NotImplemented
        """
        edges = self.edge_adjacency

        if which=='reduced':    
            self.reduced_edge_coordinates = self._node_adj_to_ec(
                self.reduced_node_coordinates, edges
            )
            if hasattr(self, 'transformed_node_coordinates'):
                # make sure the transformation is up-to-date
                self.calculate_transformed_coordinates()
                self.transformed_edge_coordinates = self._node_adj_to_ec(
                    self.transformed_node_coordinates, edges
                )
            self.num_edges = edges.shape[0]
            self.num_nodes = self.reduced_node_coordinates.shape[0]
        elif which=='transformed':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def edge_coordinates_to_node_adjacency(
            self, which : str = 'reduced'
            ) -> None:
        """
        Calcualate the node coordinate - edge adjacency representation
        and propagate the information to all available representations.
        
        which: representation to use as basis
        - 'reduced' 
        - 'transformed': NotImplemented
        """
        if which=='reduced':    
            edge_coords = self.reduced_edge_coordinates
            nodes, edges = self._ec_to_node_adj(edge_coords)
            self.reduced_node_coordinates = nodes
            self.edge_adjacency = edges
            self.num_nodes = self.reduced_node_coordinates.shape[0]
            self.num_edges = self.edge_adjacency.shape[0]
            
            if hasattr(self, 'transformed_node_coordinates'):
                self.calculate_transformed_coordinates()
            if hasattr(self, 'transformed_edge_coordinates'):
                self.transformed_edge_coordinates = self._node_adj_to_ec(
                    self.transformed_node_coordinates, self.edge_adjacency
                )
        elif which=='transformed':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _node_adj_to_ec(self, nodes, edges) -> npt.NDArray[np.float_]:
        ec = np.zeros((edges.shape[0], 6))
        ec[:,:3] = nodes[edges[:,0]]
        ec[:,3:] = nodes[edges[:,1]]
        return ec
    def _ec_to_node_adj(self, edge_coords) -> tuple:
        node_coords = np.row_stack((edge_coords[:,:3], edge_coords[:,3:]))
        nodes, inds = np.unique(node_coords, axis=0, return_inverse=True)
        numbered_edges = np.reshape(inds, (2, -1)).T
        return nodes, numbered_edges

    def verify_num_nodes_edges(self) -> None:
        assert self.num_nodes==self.reduced_node_coordinates.shape[0]
        assert self.num_edges==self.edge_adjacency.shape[0]
        if hasattr(self, 'transformed_node_coordinates'):
            assert self.transformed_node_coordinates.shape[0]==self.num_nodes
        if hasattr(self, 'reduced_edge_coordinates'):
            assert self.reduced_edge_coordinates.shape[0]==self.num_edges
        if hasattr(self, 'transformed_edge_coordinates'):
            assert self.transformed_edge_coordinates.shape[0]==self.num_edges

    def crop_lattice(self) -> None:
        """
        Crop lattice to fit within unit cell
    
        Operates on reduced edge coordinates
        """
        UC_L = 1.0

        if not hasattr(self, 'reduced_edge_coordinates'):
            self.node_adjacency_to_edge_coordinates('reduced')
        edge_coords = self.reduced_edge_coordinates

        for dim in range(3):
            new_edges = []

            plane_dims = [0,1,2]
            plane_dims.remove(dim)

            for e in edge_coords:
                # p0 and p1 are coordinates of edge endpoints
                # order such that dim-coord of p1 is greater than of p0
                if e[dim] < e[3+dim]: p0 = e[:3]; p1 = e[3:]
                else: p0 = e[3:]; p1 = e[:3]
                
                # remap edge to fit its 'left' point within window
                t = UC_L * np.floor(p0[dim]/UC_L)
                p0[dim] -= t
                p1[dim] -= t
                niter = 0
                # see if 'right' point is sticking out
                while p1[dim]>UC_L:
                    end_pt = p0 + (p1-p0)*(UC_L-p0[dim])/(p1[dim]-p0[dim])
                    new_edges.append(np.concatenate((p0, end_pt)))

                    p0 = end_pt
                    p1 = p1
                    p0[dim] -= UC_L
                    p1[dim] -= UC_L
                    niter += 1
                    assert niter<=10 # avoid hanging while loop

                # if it is not, append edge to new edges
                new_edges.append(np.concatenate((p0, p1)))

            edge_coords = np.row_stack(new_edges)
        
        self.reduced_edge_coordinates = edge_coords
        self.update_representations(basis='reduced_edge_coords')
        self.remove_duplicate_edges_nodes()

    def _node_conn_edge_colin(
            self, nodes: npt.ArrayLike, edges: npt.ArrayLike
            ) -> tuple:
        """
        Calculate connectivity of all nodes and the dot product
        between edge unit vectors connected to nodes which have 
        connectivity 2. This represents colinearity of edges.
        For nodes with connectivity other than 2, return np.nan.

        Inputs:
            nodes (N,3): nodal coordinates
            edges (E,2): 0-indexed edge adjacency
        Returns:
            node_connectivity (N,)
            dev_colin (N,)
        """
        node_connectivity = np.zeros(nodes.shape[0], dtype=np.int_)
        connected_node_nums, connectivity = np.unique(
            edges, return_counts=True
            )
        node_connectivity[connected_node_nums] = connectivity
        
        # Find nodes with connectivity 2
        num_conn_2 = connected_node_nums[connectivity==2]
        # Calculate deviation from colinearity
        dev_colin = np.full(node_connectivity.shape, np.nan)
        for n in num_conn_2:
            # choose edges connected to node n
            lines = edges[np.any(edges==n, axis=1),:]
            assert len(lines)==2
            unit_vecs = []
            other_nodes = []
            p0 = n
            for line in lines:
                p1 = (set(line)-{p0}).pop()
                other_nodes.append(p1)
                v = nodes[p1, :] - nodes[p0, :]
                unit_vecs.append(v/np.linalg.norm(v, keepdims=True))
            dev = np.abs(np.dot(unit_vecs[0], unit_vecs[1]) + 1)
            dev_colin[n] = dev
        
        return node_connectivity, dev_colin

    def calculate_edge_lengths(
            self, repr: str = 'reduced'
            ) -> npt.NDArray[np.float_]:
        """
        Use reduced nodal coordinates - edge adjacency representation
        as the basis for calculation.

        Parameters:
            repr: 'reduced' or 'transformed' or 'fundamental'

        Returns: 
            edge_lengths: array of shape (num_edges,) with edge lengths
                in the selected representation
        """
        assert self.edge_adjacency.shape==(self.num_edges, 2)
        if repr=='reduced':
            self.node_adjacency_to_edge_coordinates('reduced')
            edge_coords = self.reduced_edge_coordinates
        elif repr=='transformed':
            if not hasattr(self, 'transformed_node_coordinates'):
                self.calculate_transformed_coordinates()
            self.node_adjacency_to_edge_coordinates('reduced')
            edge_coords = self.transformed_edge_coordinates
        elif repr=='fundamental':
            self.calculate_fundamental_representation()
            edge_coords = self._node_adj_to_ec(
                        self.reduced_node_coordinates, 
                        self.fundamental_edge_adjacency
                        )
            edge_coords += self.fundamental_tesselation_vecs
            assert self.num_fundamental_edges==edge_coords.shape[0]
        else:
            raise ValueError('Unsupported representation')

        assert edge_coords.shape[1]==6
        p0 = edge_coords[:,:3]
        p1 = edge_coords[:,3:]
        v = p1 - p0
        edge_lengths = np.linalg.norm(v, axis=1)

        if repr=='reduced':
            self.reduced_edge_lengths = edge_lengths
        elif repr=='transformed':
            self.transformed_edge_lengths = edge_lengths
        elif repr=='fundamental':
            self.fundamental_edge_lengths = edge_lengths

        return edge_lengths


    def merge_colinear_edges(self) -> None:
        """
        Merge colinear edges based on nodal connectivity 2 
        and similar angle between two unit vectors along the edges.

        Operates on reduced node coordinates and edge adjacency relations.
        """
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        
        node_connectivity, dev_colin = self._node_conn_edge_colin(nodes, edges)
        num_conn_2 = np.flatnonzero(node_connectivity==2)
        num_nodes_conn_2 = len(num_conn_2)
        if num_nodes_conn_2>0:
            min_dev_num = np.nanargmin(dev_colin)
            min_dev = dev_colin[min_dev_num]
        else:
            min_dev = np.inf
        nodes_deleted = 0
        while (num_nodes_conn_2>0) and (min_dev<self.TOL_ANGLE):
            # Choose edges connected to node n
            n_edge_indices = np.any(edges==min_dev_num, axis=1)
            assert n_edge_indices.sum()==2
            lines = edges[n_edge_indices,:]
            other_nodes = []
            for line in lines:
                p1 = (set(line)-{min_dev_num}).pop()
                other_nodes.append(p1)
            new_edges = []
            new_edges.append(other_nodes)
            new_edges.extend(edges[~n_edge_indices,:])
            edges = np.row_stack(new_edges)
            nodes_deleted += 1

            node_connectivity, dev_colin = self._node_conn_edge_colin(
                                            nodes, edges
                                            )
            num_conn_2 = np.flatnonzero(node_connectivity==2)
            num_nodes_conn_2 = len(num_conn_2)
            if (~np.isnan(dev_colin)).sum()>0:
                min_dev_num = np.nanargmin(dev_colin)
                min_dev = dev_colin[min_dev_num]
            else:
                min_dev = np.inf

            assert nodes_deleted<1000  # Avoid hanging loop
        
        # print(f'Deleted {nodes_deleted} nodes because of colinear edges')    
            
        # Delete disconnected nodes
        self.edge_adjacency = edges
        self.num_edges = self.edge_adjacency.shape[0]
        self.node_adjacency_to_edge_coordinates()
        self.edge_coordinates_to_node_adjacency()

    def calculate_node_types(self):
        """
        Calculate types of nodes from reduced representation

        Returns a dictionary with sets of 
        - corner nodes
        - edge nodes
        - face nodes
        - inner nodes
        """
        UC_L = 1.0
        self.verify_num_nodes_edges()
        nodes = self.reduced_node_coordinates

        coords_on_bnds = (np.sum(np.abs(nodes)<self.TOL_DIST, axis=1)
                        + np.sum(np.abs(nodes-UC_L)<self.TOL_DIST, axis=1))

        corner_nodes = np.flatnonzero(coords_on_bnds==3)
        edge_nodes = np.flatnonzero(coords_on_bnds==2)
        face_nodes = np.flatnonzero(coords_on_bnds==1)
        inner_nodes = np.flatnonzero(coords_on_bnds==0)

        node_types = dict()
        node_types['corner_nodes'] = set(corner_nodes)
        node_types['edge_nodes'] = set(edge_nodes)
        node_types['face_nodes'] = set(face_nodes)
        node_types['inner_nodes'] = set(inner_nodes)
        self.node_types = node_types
        return node_types

    def calculate_nodal_connectivity(self) -> npt.NDArray[np.int_]:
        """
        Operate on reduced edge adjacency representation
        """
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        node_connectivity = np.zeros(nodes.shape[0], dtype=np.int_)
        connected_node_nums, connectivity = np.unique(
            edges, return_counts=True
            )
        node_connectivity[connected_node_nums] = connectivity
        self.nodal_connectivity = node_connectivity
        return self.nodal_connectivity
        
    def check_window_conditions(self) -> bool:
        """
        Operate on reduced edge adjacency representation.

        Check that 
        - reduced node coordinates lie between 0 and 1
        - the only boundary nodes are face nodes
        - the connectivity of these nodes is 1
        """
        nodes = self.reduced_node_coordinates
        if (np.abs(nodes.min())>self.TOL_DIST
            or np.abs(nodes.max()-1)>self.TOL_DIST):
            return False
        node_types = self.calculate_node_types()
        if len(node_types['corner_nodes'])>0: 
            return False
        if len(node_types['edge_nodes'])>0:
            return False
        face_node_numbers = list(node_types['face_nodes'])
        connectivity = self.calculate_nodal_connectivity()
        if not np.all(connectivity[face_node_numbers]==1):
            return False
        # TODO any other conditions - e.g. edge length
        return True

    def get_periodic_partners(self) -> list[set[int]]:
        """
        Calculate periodic partners.

        Check is done whether lattice is in a valid window condition.
        Returns:
            periodic_partners: list of 2-element sets of node numbers
                of periodic partners
        """
        assert self.check_window_conditions()
        # node types and nodal connectivity 
        # have been calculated in the window conditions call
        TOL_PARTNER = 5e-4
        node_types = self.node_types
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        face_node_nums = node_types['face_nodes']
        periodic_partners = []
        nodes_already_in = []
        for n in list(face_node_nums):
            if n in nodes_already_in: 
                continue
            x = nodes[n, :]
            dim = np.flatnonzero((np.abs(x)<self.TOL_DIST) | (np.abs(x-1)<self.TOL_DIST))
            assert len(dim)==1
            dim = dim[0]
            if abs(x[dim])<self.TOL_DIST: 
                dim_partner = 1
            elif abs(x[dim]-1)<self.TOL_DIST:
                dim_partner = 0
            else:
                RuntimeError
            comp_dims = [0,1,2]
            comp_dims.remove(dim)
            ind_partner = np.flatnonzero(
                            (np.abs(nodes[:,dim] - dim_partner)<self.TOL_DIST)
                            & (np.all(
                                np.abs(nodes[:, comp_dims] - x[comp_dims])<TOL_PARTNER, 
                                axis=1))
                            )
            assert len(ind_partner)==1
            ind_partner = ind_partner[0]
            periodic_partners.append({n, ind_partner})
            nodes_already_in.extend([n, ind_partner])
            # Vectors of edges connecting to the periodic partners
            # have to be colinear - check that
            unit_vectors = []
            for node_num in [n, ind_partner]:
                e = edges[np.any(edges==node_num, axis=1), :]
                assert e.shape==(1,2)
                e = e[0,:]
                v = nodes[e[0], :] - nodes[e[1], :]
                v = v/np.linalg.norm(v, keepdims=True)
                unit_vectors.append(v)
            assert (np.abs(np.dot(unit_vectors[0], unit_vectors[1])) - 1 
                    < self.TOL_ANGLE)
            
        assert len(nodes_already_in)==len(face_node_nums)
        self.periodic_partners = periodic_partners
        return self.periodic_partners

    def _pp_list_to_dict(self, pp_list : list) -> dict[int,int]:
        """Create a dictionary map for periodic partners"""
        pp_dict = dict()
        for pp in pp_list:
            pp_l = list(pp)
            pp_dict[pp_l[0]] = pp_l[1]
            pp_dict[pp_l[1]] = pp_l[0]
        return pp_dict

    def _obtain_best_shift_vector(self):
        MIN_EDGE_LENGTH = 5e-4
        IMPROVE_THRESHOLD = 0.05
        ITER_IMPROVE = 5
        ITER_FAIL = 20
        temp_lattice = Lattice(**self.to_dict())
        window_satisfied = temp_lattice.check_window_conditions()
        if window_satisfied:
            try:
                _ = temp_lattice.get_periodic_partners()
                best_valid_min_edge = temp_lattice.calculate_edge_lengths().min()
                best_shift_vector = np.zeros(3)
                max_iter = ITER_IMPROVE
            except AssertionError:
                pass
        else:
            best_valid_min_edge = 0.0
            best_shift_vector = np.full(3,np.nan)
            max_iter = ITER_FAIL
        at_least_one_window = window_satisfied
        niter = 0
        while ((
                not window_satisfied 
                or best_valid_min_edge<IMPROVE_THRESHOLD
                )
                and niter<max_iter):
            temp_lattice = Lattice(**self.to_dict())
            dr = 0.5*np.random.randn(3)
            temp_lattice.reduced_node_coordinates = (
                temp_lattice.reduced_node_coordinates + dr
            )
            temp_lattice.node_adjacency_to_edge_coordinates()
            temp_lattice.crop_lattice()
            temp_lattice.merge_colinear_edges()
            if temp_lattice.closest_node_distance()[0]<1e-2:
                temp_lattice.merge_node_clusters(tol=5e-3, types='all')
                temp_lattice.collapse_nodes_onto_boundaries(1e-3)
            nodes_on_edges = temp_lattice.find_nodes_on_edges()
            if nodes_on_edges:
                temp_lattice.split_edges_by_nodes(nodes_on_edges)
            window_satisfied = temp_lattice.check_window_conditions()
            if window_satisfied:
                try:
                    _ = temp_lattice.get_periodic_partners()
                except AssertionError:
                    window_satisfied = False
            at_least_one_window = at_least_one_window or window_satisfied
            e_min = temp_lattice.calculate_edge_lengths().min()
            if (e_min>best_valid_min_edge
                    and window_satisfied):
                best_valid_min_edge = e_min
                best_shift_vector = dr
                max_iter = ITER_IMPROVE
                niter = 0

            niter += 1

        if not at_least_one_window:
            raise WindowingException('Failed to obtain a valid window')
        if best_valid_min_edge < MIN_EDGE_LENGTH:
            raise WindowingException('Failed to obtain sufficiently long edge')

        return best_shift_vector

    def create_windowed(self):
        """
        Create a windowed representation of a lattice

        Operate on reduced node coordinates. 
        Return a new lattice instance.
        """
        best_shift_vector = self._obtain_best_shift_vector()
        newlat = Lattice(**self.to_dict())
        newlat.reduced_node_coordinates += best_shift_vector
        newlat.node_adjacency_to_edge_coordinates()
        newlat.crop_lattice()
        newlat.merge_colinear_edges()
        if newlat.closest_node_distance()[0]<1e-2:
            newlat.merge_node_clusters(tol=5e-3, types='all')
            newlat.collapse_nodes_onto_boundaries(1e-3)
        nodes_on_edges = newlat.find_nodes_on_edges()
        if nodes_on_edges:
            newlat.split_edges_by_nodes(nodes_on_edges)
        assert newlat.check_window_conditions()
        return newlat

    def window_lattice(self) -> None:
        """
        Create a windowed representation of a lattice

        Operate on reduced node coordinates
        """
        best_shift_vector = self._obtain_best_shift_vector()
        self.reduced_node_coordinates = (
            self.reduced_node_coordinates + best_shift_vector
            )
        self.node_adjacency_to_edge_coordinates()
        self.crop_lattice()
        self.merge_colinear_edges()
        if self.closest_node_distance()[0]<1e-2:
            self.merge_node_clusters(tol=5e-3, types='all')
            self.collapse_nodes_onto_boundaries(1e-3)
        nodes_on_edges = self.find_nodes_on_edges()
        if nodes_on_edges:
            self.split_edges_by_nodes(nodes_on_edges)
        self.split_edges_by_nodes(self.find_nodes_on_edges())
        assert self.check_window_conditions()

    def collapse_nodes_onto_boundaries(self, tol=1e-4):
        """
        Collapse nodes which are close to boundaries onto the boundaries.

        Operates on reduced nodal coordinates.
        All nodal coordinates which are very close to 0 or 1
        (within tolerance) will be replaced by 0 or 1, respectively.
        """
        nodes = self.reduced_node_coordinates
        nodes[np.abs(nodes)<tol] = 0
        nodes[np.abs(nodes-1)<tol] = 1
        self.reduced_node_coordinates = nodes
        self.update_representations()

    def remove_short_edges(self, tol=1e-4):
        """Remove edges which are below tolerance"""
        edge_lengths = self.calculate_edge_lengths()
        mask_delete = edge_lengths<tol
        self.reduced_edge_coordinates = (
            self.reduced_edge_coordinates[~mask_delete,:]
        )
        self.update_representations('reduced_edge_coords')

    def check_node_on_edge(self):
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        node_nums = set(np.arange(self.num_nodes))
        for e in edges:
            nodes_not_endpoints = node_nums - set(e)
            p0 = nodes[e[0]]
            p1 = nodes[e[1]]
            v = p1 - p0
            v2 = np.dot(v,v)
            assert v2>0
            v_norm = np.sqrt(v2)
            v_unit = v / v_norm
            for n in list(nodes_not_endpoints):
                p_n = nodes[n]
                u = p_n - p0
                u_tangent = np.dot(u,v_unit)
                u_radial = u - v_unit*u_tangent
                if (np.linalg.norm(u_radial)<self.TOL_DIST
                    and u_tangent>0
                    and u_tangent<v_norm):
                    return True
        return False

    def find_nodes_on_edges(self) -> list[tuple[set,list]]:
        """
        Find nodes which lie on edges and are not endpoints.

        Operates on reduced nodal coordinates - adjacency representation
        Returns a list of tuples (edge: set, nodes: list)
        """
        TOL_RADIAL = 5e-3
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        node_nums = set(np.arange(self.num_nodes))
        nodes_on_edges = []
        for e in edges:
            nodes_not_endpoints = node_nums - set(e)
            p0 = nodes[e[0]]
            p1 = nodes[e[1]]
            v = p1 - p0
            v_norm = np.linalg.norm(v)
            v_unit = v / v_norm
            nodes_splitting_edge = []
            for n in list(nodes_not_endpoints):
                p_n = nodes[n]
                u = p_n - p0
                u_tangent = np.dot(u,v_unit)
                u_radial = u - v_unit*u_tangent
                if (np.linalg.norm(u_radial)<TOL_RADIAL
                    and u_tangent>self.TOL_DIST
                    and u_tangent<v_norm-self.TOL_DIST):
                    nodes_splitting_edge.append(n)
            if nodes_splitting_edge:
                nodes_on_edges.append((set(e), nodes_splitting_edge))
        return nodes_on_edges

    def split_edges_by_nodes(self, nodes_on_edges: list[tuple]) -> None:
        """
        Split edges which contain internal nodes.

        Operates on reduced nodal coordinates - adjacency representation.
        May lead to the creation of overlapping edges.
        """
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        new_edges = []
        list_of_edges_to_split = [tup[0] for tup in nodes_on_edges]
        for e in edges:
            if set(e) not in list_of_edges_to_split:
                new_edges.append(e)
            else:
                i_edge = list_of_edges_to_split.index(set(e))
                internal_nodes = nodes_on_edges[i_edge][1]
                if len(internal_nodes)==1:
                    n = internal_nodes[0]
                    new_edges.append([e[0],n])
                    new_edges.append([e[1],n])
                else:
                    internal_nodes = np.array(internal_nodes)
                    # Need to choose the correct order
                    p0 = nodes[e[0]]
                    p1 = nodes[e[1]]
                    v = p1 - p0
                    v2 = np.dot(v,v)
                    assert v2>0
                    u_dots = []
                    for n in internal_nodes:
                        p_n = nodes[n]
                        u = p_n - p0
                        u2 = np.dot(u,u)
                        u_dots.append(u2)
                    sorting_indices = np.argsort(u_dots)
                    internal_nodes = internal_nodes[sorting_indices]
                    assert max(u_dots) < v2
                    p_n1 = internal_nodes[0]
                    new_edges.append([e[0],p_n1])
                    for i_internal in range(len(internal_nodes)-1):
                        p_n0 = internal_nodes[i_internal]
                        p_n1 = internal_nodes[i_internal+1]
                        new_edges.append([p_n0,p_n1])
                    p_n0 = internal_nodes[-1]
                    new_edges.append([p_n0,e[1]])
        
        edges = np.row_stack(new_edges)
        # remove self-connecting edges
        mask_self_incident = edges[:,0]==edges[:,1]
        edges = edges[~mask_self_incident]
        self.edge_adjacency = edges
        self.update_representations()     
        self.remove_duplicate_edges_adjacency()


    def find_edge_intersections(self) -> dict[int,list]:
        """
        Find intersections between edge pairs.

        Operates on reduced adjacency representation.

        Returns:
            edge_intersection_points: dictionary with edge indices
                as keys and coordinates of intersection points as values
        """
        TOL = 1e-4
        TOL_ANGLE = 2e-3
        edges = self.edge_adjacency
        nodes = self.reduced_node_coordinates
        assert edges.shape[0]==self.num_edges
    
        edge_intersection_points = {}
        for i_ei in range(self.num_edges):
            ei = edges[i_ei]
            pi = nodes[ei[0]]
            vi = nodes[ei[1]] - nodes[ei[0]]
            vi_norm = np.linalg.norm(vi)
            for i_ej in range(i_ei+1, self.num_edges):
                ej = edges[i_ej]
                pj = nodes[ej[0]]
                vj = nodes[ej[1]] - nodes[ej[0]]
                vj_norm = np.linalg.norm(vj)
                vi_vj_cross = np.cross(vi, vj)
                unit_plane_normal = vi_vj_cross/(vi_norm*vj_norm)
                sin_alpha = np.linalg.norm(unit_plane_normal)
                sin_alpha = round(sin_alpha, 5)
                alpha = np.arcsin(sin_alpha)
                if alpha < TOL_ANGLE:
                    # edges parallel
                    pass
                else:
                    dist = abs(np.dot(unit_plane_normal, pi-pj))
                    if dist < self.TOL_DIST:
                        # edges lie in plane and are not parallel
                        ni = np.cross(unit_plane_normal, vi)
                        nj = np.cross(unit_plane_normal, vj)
                        eta_i = - np.dot(pi-pj, nj)/np.dot(vi,nj)
                        eta_j = np.dot(pi-pj, ni)/np.dot(vj,ni)
                        if (eta_i>1e-2
                            and eta_i<1-1e-2
                            and eta_j>1e-2
                            and eta_j<1-1e-2):
                            # intersect if eta_i and eta_j 
                            # are between 0 and 1
                            p_int_i = pi + eta_i*vi
                            p_int_j = pj + eta_j*vj
                            assert np.sum(p_int_i-p_int_j)<TOL
                            
                            for ind in [i_ei, i_ej]:
                                if ind not in edge_intersection_points:
                                    edge_intersection_points[ind] = []
                                found = False
                                for p in edge_intersection_points[ind]:
                                    if np.allclose(p, p_int_i):
                                        found = True
                                        break
                                if found:
                                    pass
                                else:
                                    edge_intersection_points[ind].append(p_int_i)
                        else:
                            pass
                    else:
                        pass
        return edge_intersection_points
                
    def split_edges_at_intersections(
        self, edge_intersection_points: dict[int,list]
    ) -> None:
        """
        Split edges at precomputed intersection points.

        Parameters:
            edge_intersection_points: as returned by function
                'find_edge_intersections'
        """
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        new_edge_coordinates = []

        for i_edge in edge_intersection_points:
            internal_points = edge_intersection_points[i_edge]
            e = edges[i_edge]
            p0 = nodes[e[0]]
            p1 = nodes[e[1]]
            if len(internal_points)==1:
                p_n = internal_points[0]

                new_edge_coordinates.append(np.concatenate([p0,p_n]))
                new_edge_coordinates.append(np.concatenate([p_n,p1]))
            else:
                internal_points = np.row_stack(internal_points)
                # Need to choose the correct order
                v = p1 - p0
                v2 = np.dot(v,v)
                assert v2>0
                u_dots = []
                for p_n in internal_points:
                    u = p_n - p0
                    u2 = np.dot(u,u)
                    u_dots.append(u2)
                sorting_indices = np.argsort(u_dots)
                internal_points = internal_points[sorting_indices]
                assert max(u_dots) < v2

                p_n1 = internal_points[0]
                new_edge_coordinates.append(np.concatenate([p0,p_n1]))
                for i_internal in range(len(internal_points)-1):
                    p_n0 = internal_points[i_internal]
                    p_n1 = internal_points[i_internal+1]
                    new_edge_coordinates.append(np.concatenate([p_n0,p_n1]))


                p_n0 = internal_points[-1]
                new_edge_coordinates.append(np.concatenate([p_n0,p1]))
        
        i_edges_to_copy = list(
            set(range(edges.shape[0]))-set(edge_intersection_points.keys())
            )
        copied_edge_coords = np.zeros((len(i_edges_to_copy),6))
        copied_edge_coords[:,:3] = nodes[edges[i_edges_to_copy,0]]
        copied_edge_coords[:,3:] = nodes[edges[i_edges_to_copy,1]]
        
        new_edge_coordinates.extend(copied_edge_coords)
        new_edge_coordinates = np.row_stack(new_edge_coordinates)
        self.reduced_edge_coordinates = new_edge_coordinates
        self.remove_duplicate_edges_nodes()
        

    def closest_node_distance(self, repr='reduced'):
        """
        Find the minimum distance between all pairs of nodes.

        Parameters:
            repr: 'reduced' or 'transformed'
        """
        if repr=='reduced':
            nodes = self.reduced_node_coordinates
        elif repr=='transformed':
            self.calculate_transformed_coordinates()
            nodes = self.transformed_node_coordinates
        else:
            raise ValueError

        assert nodes.shape[0]==self.num_nodes

        min_dist = np.inf
        min_dist_node_pair = None
        for i in range(0, self.num_nodes):
            n_0 = nodes[i]
            for j in range(i+1, self.num_nodes):
                n_1 = nodes[j]
                v = n_1 - n_0
                dist = np.linalg.norm(v)
                if dist<min_dist:
                    min_dist = dist
                    min_dist_node_pair = {i,j}
        
        return min_dist, min_dist_node_pair

    def find_node_clusters(self, repr='reduced', tol=1e-4):
        if repr=='reduced':
            nodes = self.reduced_node_coordinates
        elif repr=='transformed':
            raise NotImplementedError
        else:
            raise ValueError

        assert nodes.shape[0]==self.num_nodes

        clusters = []
        in_cluster = {}
        for i in range(0, self.num_nodes):
            n_0 = nodes[i]
            if i in in_cluster:
                i_cluster = in_cluster[i]
            else:
                i_cluster = len(clusters)
                in_cluster[i] = i_cluster
                clusters.append([i])
            
            for j in range(i+1, self.num_nodes):
                n_1 = nodes[j]
                v = n_1 - n_0
                dist = np.linalg.norm(v)
                if dist<tol:
                    if j in in_cluster:
                        if in_cluster[j]==i_cluster:
                            pass
                        else:
                            # need to merge the two clusters
                            # into cluster where node i is
                            for n in clusters[in_cluster[j]]:
                                in_cluster[n] = i_cluster
                                clusters[i_cluster].append(n)
                    else:
                        clusters[i_cluster].append(j)
                        in_cluster[j] = i_cluster
        uq_clusters = np.unique(list(in_cluster.values()))
        clusters = [clusters[i] for i in uq_clusters]
        return clusters

    def merge_node_clusters(
            self, repr='reduced', tol=1e-4, types='all'
            ):
        """
        Merge clusters of nodes.
        """
        if repr=='reduced':
            nodes = self.reduced_node_coordinates
        elif repr=='transformed':
            raise NotImplementedError
        else:
            raise ValueError
        assert types in ['all', 'inner']
        if types=='inner':
            node_types = self.calculate_node_types()
            free_nodes = node_types['inner_nodes']
        elif types=='all':
            free_nodes = set(range(self.num_nodes))

        assert nodes.shape[0]==self.num_nodes

        clusters = []
        in_cluster = {}
        for i in range(0, self.num_nodes):
            n_0 = nodes[i]
            if i not in free_nodes:
                # node must not yet be in in_cluster
                assert i not in in_cluster
            if i in in_cluster:
                i_cluster = in_cluster[i]
            else:
                i_cluster = len(clusters)
                in_cluster[i] = i_cluster
                clusters.append([i])
            if i not in free_nodes:
                continue
            
            for j in range(i+1, self.num_nodes):
                if j not in free_nodes:
                    continue
                n_1 = nodes[j]
                v = n_1 - n_0
                dist = np.linalg.norm(v)
                if dist<tol:
                    if j in in_cluster:
                        if in_cluster[j]==i_cluster:
                            pass
                        else:
                            # need to merge the two clusters
                            # into cluster where node i is
                            for n in clusters[in_cluster[j]]:
                                in_cluster[n] = i_cluster
                                clusters[i_cluster].append(n)
                    else:
                        clusters[i_cluster].append(j)
                        in_cluster[j] = i_cluster
        
        clustered_nodes = np.full((len(clusters),3), np.nan)
        for i_cluster in np.unique(list(in_cluster.values())):
            node_nums = clusters[i_cluster]
            coords = nodes[node_nums, :]
            cg = np.mean(coords, axis=0)
            clustered_nodes[i_cluster] = cg
        
        node_map = []
        for n in sorted(in_cluster.keys()):
            node_map.append([n, in_cluster[n]])
        node_map = np.array(node_map)
        assert np.allclose(np.arange(self.num_nodes), node_map[:,0])
        edges = node_map[:,1][self.edge_adjacency]
        # remove self-connecting edges
        mask_self_incident = edges[:,0]==edges[:,1]
        edges = edges[~mask_self_incident]
        self.edge_adjacency = edges
        self.reduced_node_coordinates = clustered_nodes
        self.node_adjacency_to_edge_coordinates()
        self.edge_coordinates_to_node_adjacency()
        self.remove_duplicate_edges_adjacency()

    def calculate_fundamental_representation(self) -> None:
        """
        Calculate the fundamental representation of lattice 
        which is based on inner nodes and tesselation vectors.

        Operates on reduced node - edge adjacency representation
        """
        pp_list = self.get_periodic_partners()
        pp_dict = self._pp_list_to_dict(pp_list)
        # node types were calculated in get_periodic_partners call
        node_types = self.node_types
        inner_nodes = node_types['inner_nodes']
        assert len(inner_nodes)>0
        edges = self.edge_adjacency
        nodes = self.reduced_node_coordinates

        used_edge_indices = []
        new_edges = []
        t_vecs = []
        for i_edge, e in enumerate(edges):
            if i_edge in used_edge_indices: 
                # every edge is transcribed just once
                continue 
            used_edge_indices.append(i_edge) 
            loc_edge = []
            loc_t_vecs = [np.zeros(3), np.zeros(3)]
            for i_point, point_num in enumerate(e):
                p_loc = point_num
                ntrials = 0
                # backtrack the point all the way until we hit an inner node
                while not p_loc in inner_nodes:    
                    # partner is either connected to p_loc 
                    # by unused edge (priority) or it is periodic partner
                    conn_edge_ind = np.flatnonzero(
                                        np.any(edges==p_loc, axis=1)
                                        )
                    assert len(conn_edge_ind)==1
                    conn_edge_ind = conn_edge_ind[0]
                    if not conn_edge_ind in used_edge_indices:
                        partner_edge = edges[conn_edge_ind]
                        partner = (set(partner_edge) - {p_loc}).pop()
                        used_edge_indices.append(conn_edge_ind) 
                    else: 
                        # pick periodic partner and 
                        # need to add to translation vector
                        partner = pp_dict[p_loc]
                        t = nodes[p_loc, :] - nodes[partner, :] 
                        loc_t_vecs[i_point] += t
                    
                    ntrials += 1
                    p_loc = partner
                    assert ntrials<=10 # avoid hanging loop

                # Use p_loc in the edge adjacency. 
                # Together with loc_t_vec they set edge coordinates
                loc_edge.append(p_loc)

            new_edges.append(loc_edge)
            t_vecs.append(np.concatenate(loc_t_vecs))
        
        new_edges = np.row_stack(new_edges)
        t_vecs = np.row_stack(t_vecs)
        # reduce t_vecs to zero first 3 columns
        t_0 = np.copy(t_vecs[:,:3])
        t_vecs[:,:3] -= t_0
        t_vecs[:,3:] -= t_0
        assert set(new_edges.flatten())==inner_nodes
        self.fundamental_edge_adjacency = new_edges
        self.fundamental_tesselation_vecs = t_vecs
        self.num_fundamental_edges = self.fundamental_edge_adjacency.shape[0]

    def perturb_inner_nodes(self, dr_mag: float, kind: str) -> None:
        """
        Displace inner nodes using the fundamental representation.

        Parameters:
            - dr_mag: magnitude of perturbation
            - kind: 'sphere_surf' or 'sphere_solid' or 'gaussian'
        """
        self.calculate_fundamental_representation()
        nodes_to_perturb = np.unique(self.fundamental_edge_adjacency.flatten())
        num_nodes_to_perturb = len(nodes_to_perturb)
            
        if kind=='sphere_surf':
            dr = np.random.randn(num_nodes_to_perturb, 3)
            dr = dr / np.linalg.norm(dr, axis=1, keepdims=True)
            dr = dr * dr_mag
        elif kind=='sphere_solid':
            phi = 2*np.pi*np.random.rand(num_nodes_to_perturb)
            costheta = 2*(np.random.rand(num_nodes_to_perturb) - 0.5)
            u = np.random.rand(num_nodes_to_perturb)
            theta = np.arccos( costheta )
            r = dr_mag * u**(1/3)
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            dr = np.column_stack((x,y,z))
        elif kind=='gaussian':
            dr = np.random.randn(num_nodes_to_perturb, 3)
            dr = dr * dr_mag
        else: raise ValueError('Unknown type of argument "kind"')

        self.reduced_node_coordinates[nodes_to_perturb] += dr
        nodes = self.reduced_node_coordinates
        edge_coords = self._node_adj_to_ec(
                        nodes, self.fundamental_edge_adjacency
                        )
        edge_coords += self.fundamental_tesselation_vecs
        self.reduced_edge_coordinates = edge_coords
        self.edge_coordinates_to_node_adjacency()
        try:
            self.window_lattice()
        except Exception:
            self.window_lattice()
        # self.crop_lattice()


    def to_dict(self) -> dict:
        """Obtain a dictionary with the reduced representation."""
        d = dict()
        d['name'] = self.name
        if hasattr(self, 'lattice_constants'):
            d['lattice_constants'] = self.lattice_constants
        d['reduced_node_coordinates'] = self.reduced_node_coordinates
        d['edge_adjacency'] = self.edge_adjacency
        return d

    def print_lattice_lines(self) -> list[str]:
        """
        Obtain a text representation of lattice.
        
        Suitable for exporting to file using writelines.
        """
        lines = []

        assert hasattr(self, 'name')
        lines.append(f'Name: {self.name}')
        lines.append('')
        if hasattr(self, 'lattice_constants'):
            assert len(self.lattice_constants)==6
            lines.append(
                'Normalized unit cell parameters (a,b,c,alpha,beta,gamma):'
            )
            line = ''
            for i, x in enumerate(self.lattice_constants):
                line = line + f'{x:.5f}'
                if i!=5:
                    line = line + ', '
            lines.append(line)

        if hasattr(self, 'compliance_tensors'):
            lines.append('')
            lines.append('Compliance tensors start (flattened upper triangular)')
            for rel_dens in self.compliance_tensors:
                lines.append(f'-> at relative density {rel_dens}:')
                S = self.compliance_tensors[rel_dens]
                assert S.shape==(6,6)
                nums = S[np.triu_indices(6)].tolist()
                line = ''
                for s in nums:
                    line = line + f'{s:.5g},'
                lines.append(line)
            lines.append('Compliance tensors end')

        if hasattr(self, 'Youngs_moduli'):
            lines.append('')
            lines.append(
                f"Young's moduli at relative density "\
                f"{self.Youngs_moduli['rel_dens']}"
            )
            lines.append(
                f'  max: {self.Youngs_moduli["E_max"]:.5g} '\
                f'at x,y,z=('
                f'{self.Youngs_moduli["max_dir"][0]:.5g},'
                f'{self.Youngs_moduli["max_dir"][1]:.5g},'
                f'{self.Youngs_moduli["max_dir"][2]:.5g}'
                ')'
            )
            lines.append(
                f'  min: {self.Youngs_moduli["E_min"]:.5g} '\
                f'at x,y,z=('
                f'{self.Youngs_moduli["min_dir"][0]:.5g},'
                f'{self.Youngs_moduli["min_dir"][1]:.5g},'
                f'{self.Youngs_moduli["min_dir"][2]:.5g}'
                ')'
            )

        if hasattr(self, 'scaling_exponents'):
            lines.append('')
            lines.append(f"Scaling exponents")
            lines.append(
                f'  max: {self.scaling_exponents["n_max"]:.5f} '\
                f'at x,y,z=('
                f'{self.scaling_exponents["max_dir"][0]:.5g},'
                f'{self.scaling_exponents["max_dir"][1]:.5g},'
                f'{self.scaling_exponents["max_dir"][2]:.5g}'
                ')'
            )
            lines.append(
                f'  min: {self.scaling_exponents["n_min"]:.5f} '\
                f'at x,y,z=('
                f'{self.scaling_exponents["min_dir"][0]:.5g},'
                f'{self.scaling_exponents["min_dir"][1]:.5g},'
                f'{self.scaling_exponents["min_dir"][2]:.5g}'
                ')'
            )
        
        assert hasattr(self,'reduced_node_coordinates')
        assert hasattr(self,'edge_adjacency')
        
        lines.append('')
        lines.append('Nodal positions:')
        for x in self.reduced_node_coordinates:
            lines.append(f'{x[0]:.5f} {x[1]:.5f} {x[2]:.5f}')

        lines.append('')
        lines.append('Bar connectivities:')
        for e in self.edge_adjacency:
            lines.append(f'{int(e[0])} {int(e[1])}')
        lines.append('')
        return lines

class WindowingException(Exception):
    pass