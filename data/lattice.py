import copy
from collections.abc import Sequence, Iterable
from typing import Optional, Tuple, List, Set, Dict
import numpy as np
import numpy.typing as npt
from math import ceil
from scipy.spatial import transform

class Lattice:
    TOL_DIST: float = 1e-5
    TOL_ANGLE: float = 1e-5
    UC_L = 1.0 # length of unit cell
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
    node_types: dict
    periodic_partners: list
    reduced_edge_lengths: npt.NDArray[np.float_]
    transformed_edge_lengths: npt.NDArray[np.float_]
    fundamental_edge_lengths: npt.NDArray[np.float_]
    # elasticity properties
    S_tens: npt.NDArray[np.float_]
    compliance_tensors: dict
    Youngs_moduli: dict
    scaling_exponents: dict
    # other properties
    lattice_constants: npt.NDArray[np.float_]
    rel_dens: float
    edge_radii: npt.NDArray[np.float_]    

    def __init__(
            self,*, name=None, code=None, lattice_constants=None,
            nodal_positions=None, edge_adjacency=None, 
            edge_coordinates=None, **kwargs
            ) -> None:
        """Construct lattice unit cell.

        Takes in keyword-only arguments. 

        Examples:
        
            Three ways of initialisation.
                - by unpacking the catalogue dictionary
                
                >>> from data import Lattice, Catalogue
                >>> cat = Catalogue.from_file('Unit_Cell_Catalog.txt', 1)
                >>> lat = Lattice(**cat.get_unit_cell(cat.names[0]))
                >>> lat
                {'name': 'cub_Z06.0_E1', 'num_nodes': 8, 'num_edges': 12}

                - by specifying node coordinates and edge adjacency

                >>> import numpy as np
                >>> nodes = np.array([[0,0,0],[1,0,0],[0.5,1,0],[0.5,0.5,1]])
                >>> edges = np.array([[0,1],[1,2],[0,2],[0,3],[1,3],[2,3]])
                >>> lat = Lattice(nodal_positions=nodes, edge_adjacency=edges, 
                >>>               name='pyramid')
                >>> lat
                {'name': 'pyramid', 'num_nodes': 4, 'num_edges': 6}

                - by specifying edge coordinates

                >>> edge_coords = np.array([
                >>>    [0,0,0,1,0,0],
                >>>    [1,0,0,0.5,1,0],
                >>>    [0.5,1,0,0,0,0],
                >>>    [0,0,0,0.5,0.5,1],
                >>>    [1,0,0,0.5,0.5,1],
                >>>    [0.5,1,0,0.5,0.5,1]
                >>> ])
                >>> lat = Lattice(edge_coordinates=edge_coords, name='pyramid')
                >>> lat
                {'name': 'pyramid', 'num_edges': 6}

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
        
        if isinstance(lattice_constants, Iterable):
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


    def calculate_UC_volume(self) -> float:
        """Calculate unit cell volume from internally-stored crystal data

        Given internally-stored crystal data 
        :math:`a,b,c,{\\alpha},{\\beta},{\\gamma}`, 
        calculate unit cell volume as

        .. math::

            abc \sqrt{1 - \cos({\\alpha})^2 - \cos(\\beta)^2 - \cos(\\gamma)^2 + 2\cos(\\alpha)\cos(\\beta)\cos(\\gamma)}

        For more details see the supporting information of PNAS paper at
        https://www.pnas.org/doi/10.1073/pnas.2003504118

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
        uc_volume = a*b*c* np.sqrt( term )  
        return uc_volume

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
        
        omega = self.calculate_UC_volume()
        
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
        
        self.transformed_node_coordinates = self._transform_coordinates(
            nodes_in, transform_mat
            )

    @staticmethod
    def _transform_coordinates(
        coordinates: npt.NDArray[np.float_],
        transform_matrix: npt.NDArray[np.float_]
        ):
        """
        Transform nodal coordinates or edge vectors.

        Parameters:
            - coordinates: (N,3) array of coordinates
            - transform_matrix (3,3)
        """
        nodes_out = np.matmul( transform_matrix, np.transpose(coordinates) )
        return np.transpose( nodes_out )

    @staticmethod
    def _rotate_coordinates(
        coordinates: npt.NDArray[np.float_], 
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
        elif repr=='fundamental':
            edge_lengths = self.calculate_edge_lengths(repr='fundamental')
        else:
            raise NotImplementedError
        sum_edge_lengths = edge_lengths.sum()

        uc_vol = self.calculate_UC_volume()

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
        

    def collapse_nodes_onto_boundaries(self, tol=1e-4):
        """
        Collapse nodes which are close to boundaries onto the boundaries.

        Operates on reduced nodal coordinates.
        All nodal coordinates which are very close to 0 or UC_L
        (within tolerance) will be replaced by 0 or UC_L, respectively.
        """
        nodes = self.reduced_node_coordinates
        nodes[np.abs(nodes)<tol] = 0
        nodes[np.abs(nodes-self.UC_L)<tol] = self.UC_L
        self.reduced_node_coordinates = nodes
        self.update_representations()


    def merge_nonunique_nodes(self, decimals: Optional[int] = None) -> None:
        """
        Merge nodes with identical coordinates
        and collapse self-incident edges
        """
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        if decimals:
            nodes = np.round(nodes, decimals=decimals)
        uq_nodes, uq_inv = np.unique(nodes, axis=0, return_inverse=True)
        edges = uq_inv[edges]

        mask_self_edge = edges[:,0]==edges[:,1]
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

    def remove_self_incident_edges(self) -> None:
        """
        Remove edges which have identical endpoints

        Operates on the adjacency representation.
        """
        edges = self.edge_adjacency
        mask_self_connecting = edges[:,0]==edges[:,1]
        edges = edges[~mask_self_connecting]

        self.edge_adjacency = edges
        self.update_representations()

    def remove_duplicate_edges_nodes(self) -> None:
        """
        Remove edges which are on top of each other.

        Operates on edge coordinate representation.
        To deal with machine precision, round all nodal coordinates
        to a specific number of decimal places.
        """
        NDECIMALS = 5
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        edge_coords = self._node_adj_to_ec(nodes, edges)
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
        # TODO: update all relevant attributes
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

    @staticmethod
    def _node_adj_to_ec(nodes, edges) -> npt.NDArray[np.float_]:
        ec = np.zeros((edges.shape[0], 6))
        ec[:,:3] = nodes[edges[:,0]]
        ec[:,3:] = nodes[edges[:,1]]
        return ec
        
    @staticmethod
    def _ec_to_node_adj(edge_coords) -> tuple:
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

    @staticmethod
    def _node_conn_edge_colin(
            nodes: npt.ArrayLike, edges: npt.ArrayLike
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
            p0 = n
            for line in lines:
                p1 = (set(line)-{p0}).pop()
                v = nodes[p1, :] - nodes[p0, :]
                unit_vecs.append(v/np.linalg.norm(v, keepdims=True))
            dev = np.abs(np.dot(unit_vecs[0], unit_vecs[1]) + 1)
            dev_colin[n] = dev
        
        return node_connectivity, dev_colin

    def find_nodes_on_edges(self) -> List[Tuple[Set,npt.NDArray]]:
        """
        Find nodes which lie on edges and are not endpoints.

        Returns a list of tuples (edge: set, nodes: coordinates of points that need to split edges)

        Example:

            >>> nodes = [[0,0,0],[1,0,0],[0.5,0,0]]
            >>> edges = [[0,1]]
            >>> lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
            >>> lat
            {'num_nodes': 3, 'num_edges': 1}
            >>> nodes_on_edges = lat.find_nodes_on_edges()
            >>> print(nodes_on_edges)
            [({0, 1}, array([[0.5, 0. , 0. ]]))]

            .. plot::

                from utils import plotting
                from data import Lattice
                import matplotlib.pyplot as plt
                nodes = [[0,0,0],[1,0,0],[0.5,0,0]]
                edges = [[0,1]]
                lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
                fig, ax = plt.subplots(figsize=(4,2))
                ax = plotting.plot_unit_cell_2d(lat, ax=ax)
                ax.set_yticks([]); ax.set_ylabel('')
                plt.tight_layout()
                ax


        """
        TOL_RADIAL = 5e-3
        nodes = self.reduced_node_coordinates
        edges = self.edge_adjacency
        node_nums = set(np.arange(self.num_nodes))
        nodes_on_edges = []
        for e in edges:
            nodes_not_endpoints = list(node_nums - set(e))
            p0 = nodes[e[0]]
            p1 = nodes[e[1]]
            v = p1 - p0
            v_norm = np.linalg.norm(v)
            v_unit = v / v_norm
            u_vecs = nodes[nodes_not_endpoints, :] - p0
            u_tangent = np.einsum('ij,j->i', u_vecs, v_unit)
            u_radial = u_vecs - v_unit.reshape(1,3)*u_tangent.reshape(-1,1)
            inds_nodes_on_edge = np.flatnonzero(
                (np.linalg.norm(u_radial, axis=1)<TOL_RADIAL) 
                & (u_tangent>self.TOL_DIST) 
                & (u_tangent<v_norm-self.TOL_DIST)
            )
            nodes_splitting_edge = [nodes_not_endpoints[i] for i in inds_nodes_on_edge]
            if nodes_splitting_edge:
                internal_pts = nodes[nodes_splitting_edge,:]
                nodes_on_edges.append((set(e), internal_pts))
        return nodes_on_edges


    
    def find_edge_intersections(self) -> List[Tuple[Set, npt.NDArray]]:
        """
        Find intersections between edge pairs.

        Operates on reduced adjacency representation.

        Returns:
            edge_intersection_points: dictionary with edge indices
                as keys and coordinates of intersection points as values
            
        Example:

            >>> nodes = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
            >>> edges = [[0,2],[1,3]]
            >>> lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
            >>> lat
            {'num_nodes': 4, 'num_edges': 2}
            >>> edge_intersections = lat.find_edge_intersections()
            >>> print(edge_intersections)
            [({0, 2}, array([[0.5, 0.5, 0. ]])), ({1, 3}, array([[0.5, 0.5, 0. ]]))]

            .. plot::

                from utils import plotting
                from data import Lattice
                import matplotlib.pyplot as plt
                nodes = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
                edges = [[0,2],[1,3]]
                lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
                fig, ax = plt.subplots(figsize=(4,2))
                ax = plotting.plot_unit_cell_2d(lat, ax=ax)
                plt.tight_layout()
                ax


        """
        TOL = 1e-4
        TOL_ANGLE = 2e-3
        edges = self.edge_adjacency
        nodes = self.reduced_node_coordinates
        assert edges.shape[0]==self.num_edges
    
        edge_intersection_points = {}
        for i_ei in range(self.num_edges-1):
            ei = edges[i_ei]
            pi = nodes[ei[0]]
            vi = nodes[ei[1]] - nodes[ei[0]]
            vi_norm = np.linalg.norm(vi)
            indices_pair_edge = np.arange(i_ei+1, self.num_edges)
            ej = edges[indices_pair_edge,:]
            pj = nodes[ej[:,0]]
            vj = nodes[ej[:,1]] - nodes[ej[:,0]]
            vj_norm = np.linalg.norm(vj, axis=-1)
            vi_vj_cross = np.cross(vi, vj)
            unit_plane_normal = vi_vj_cross/(vi_norm*vj_norm).reshape(-1,1)
            sin_alpha = np.linalg.norm(unit_plane_normal, axis=-1)
            sin_alpha = np.round(sin_alpha, 5)
            alpha = np.arcsin(sin_alpha)
            dist = np.abs(np.einsum('ij,ij->i', unit_plane_normal, pi-pj))
            inds_planar_edges = np.flatnonzero(
                (dist<self.TOL_DIST) & (alpha>TOL_ANGLE)
            )
            if len(inds_planar_edges)>0:
                ni = np.cross(unit_plane_normal[inds_planar_edges,:], vi)
                nj = np.cross(unit_plane_normal[inds_planar_edges,:], vj[inds_planar_edges])
                eta_i = - np.einsum('ij,ij->i', pi-pj[inds_planar_edges], nj) / np.einsum('i,ji->j',vi,nj)
                eta_j = np.einsum('ij,ij->i', pi-pj[inds_planar_edges], ni) / np.einsum('ij,ij->i',vj[inds_planar_edges],ni)
                inds_inside = np.flatnonzero(
                    (eta_i>1e-2) & (eta_i<1-1e-2) & (eta_j>1e-2) & (eta_j<1-1e-2)
                )
                if len(inds_inside)>0:
                    p_int_i = pi + eta_i[inds_inside].reshape(-1,1)*vi.reshape(1,3)
                    p_int_j = pj[inds_planar_edges][inds_inside] + eta_j[inds_inside].reshape(-1,1)*vj[inds_planar_edges][inds_inside]
                    assert np.sum(p_int_i-p_int_j)<TOL
                    i_ejs = indices_pair_edge[inds_planar_edges][inds_inside]
                    if i_ei not in edge_intersection_points:
                        edge_intersection_points[i_ei] = []
                    edge_intersection_points[i_ei].extend(p_int_i)

                    for ind, p in zip(i_ejs, p_int_i):
                        if ind not in edge_intersection_points:
                            edge_intersection_points[ind] = []
                        edge_intersection_points[ind].append(p)

        output_list = []
        for ind in edge_intersection_points.keys():
            tup = (
                {edges[ind,0], edges[ind,1]}, 
                np.unique(np.row_stack(edge_intersection_points[ind]), axis=0)
            )
            output_list.append(tup)
        return output_list


    def split_edges_by_points(
        self, edge_split_coords: List[Tuple[Set, npt.NDArray]]
    ) -> None:
        """
        Split edges at specific coordinate points. 
        
        Args:
            edge_split_coords (List[Tuple[Set, npt.NDArray]]): list of tuples 
                (edge, point_array) in format as returned by functions
                :func:`find_nodes_on_edges` and :func:`find_edge_intersections`.
                Each edge will be split into multiple edges by the points 
                in point_array.

        Examples:

            >>> nodes = [[0,0,0],[1,0,0],[0.5,0,0]]
            >>> edges = [[0,1]]
            >>> lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
            >>> lat
            {'num_nodes': 3, 'num_edges': 1}
            >>> nodes_on_edges = lat.find_nodes_on_edges()
            >>> lat.split_edges_by_points(nodes_on_edges)
            >>> lat
            {'num_nodes': 3, 'num_edges': 2}

            .. plot::

                from utils import plotting
                from data import Lattice
                import matplotlib.pyplot as plt
                nodes = [[0,0,0],[1,0,0],[0.5,0,0]]
                edges = [[0,1]]
                lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
                fig, axes = plt.subplots(ncols=2, figsize=(6,2))
                plotting.plot_unit_cell_2d(lat, ax=axes[0])
                nodes_on_edges = lat.find_nodes_on_edges()
                lat.split_edges_by_points(nodes_on_edges)
                plotting.plot_unit_cell_2d(lat, ax=axes[1])
                for ax in axes: ax.set_yticks([]); ax.set_ylabel('')
                axes[0].set_title('Before splitting')
                axes[1].set_title('After splitting')
                fig.tight_layout()
                axes

            >>> nodes = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
            >>> edges = [[0,2],[1,3]]
            >>> lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
            >>> lat
            {'num_nodes': 4, 'num_edges': 2}
            >>> edge_intersections = lat.find_edge_intersections()
            >>> lat.split_edges_by_points(edge_intersections)
            >>> lat
            {'num_nodes': 4, 'num_edges': 4}

            .. plot::

                from utils import plotting
                from data import Lattice
                import matplotlib.pyplot as plt
                nodes = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
                edges = [[0,2],[1,3]]
                lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
                fig, axes = plt.subplots(ncols=2, figsize=(6,3))
                plotting.plot_unit_cell_2d(lat, ax=axes[0])
                edge_intersections = lat.find_edge_intersections()
                lat.split_edges_by_points(edge_intersections)
                plotting.plot_unit_cell_2d(lat, ax=axes[1])
                axes[0].set_title('Before splitting')
                axes[1].set_title('After splitting')
                fig.tight_layout()
                axes

        See Also:
            :func:`find_nodes_on_edges`
            :func:`find_edge_intersections`

        """
        nodes = self.reduced_node_coordinates
        edges = np.sort(self.edge_adjacency, axis=1)
        remaining_edge_indices = set(range(self.num_edges))
        new_edge_coords = []
        for e_set, mid_points in edge_split_coords:
            endpoint_inds = sorted(list(e_set))
            p0,p1 = nodes[endpoint_inds, :]
            i_edge = np.flatnonzero(np.all(edges==endpoint_inds, axis=1))
            assert len(i_edge)==1
            i_edge = i_edge[0]
            remaining_edge_indices.remove(i_edge)
            assert mid_points.ndim==2
            if mid_points.shape[0]==1:
                p_mid = mid_points[0,:]
                new_edge_coords.append(np.concatenate([p0, p_mid]))
                new_edge_coords.append(np.concatenate([p_mid, p1]))
            else:
                pts = np.row_stack((mid_points, p0, p1))
                v = p1 - p0
                dots = np.einsum('ij,j->i', pts-p0.reshape(1,3), v)
                sorting_inds = np.argsort(dots)
                pts = pts[sorting_inds, :]
                new_edge_coords.extend(
                    np.column_stack((pts[:-1,:], pts[1:,:]))
                )
        if len(remaining_edge_indices)>0:
            edge_inds_to_copy = list(remaining_edge_indices)
            end_points = edges[edge_inds_to_copy, :]
            new_edge_coords.extend(
                np.column_stack((nodes[end_points[:,0]], nodes[end_points[:,1]]))
            )
        self.reduced_edge_coordinates = np.around(np.row_stack(new_edge_coords), decimals=4)
        self.update_representations(basis='reduced_edge_coords')
        self.remove_self_incident_edges()
        self.remove_duplicate_edges_adjacency() 
        

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

        TODO: is fundamental transformed or not?
        """
        assert self.edge_adjacency.shape==(self.num_edges, 2)
        if repr=='reduced':
            edge_coords = self._node_adj_to_ec(
                self.reduced_node_coordinates, self.edge_adjacency
            )
        elif repr=='transformed':
            self.calculate_transformed_coordinates()
            edge_coords = self._node_adj_to_ec(
                self.transformed_node_coordinates, self.edge_adjacency
            )
        elif repr=='fundamental':
            if not hasattr(self, 'fundamental_edge_adjacency'):
                self.calculate_fundamental_representation()
            edge_coords = self._node_adj_to_ec(
                        self.reduced_node_coordinates, 
                        self.fundamental_edge_adjacency
                        )
            edge_coords += self.fundamental_tesselation_vecs
            assert self.num_fundamental_edges==edge_coords.shape[0]
        else:
            raise ValueError('Unsupported representation')

        edge_lengths = self._edge_lengths_from_coords(edge_coords)

        if repr=='reduced':
            self.reduced_edge_lengths = edge_lengths
        elif repr=='transformed':
            self.transformed_edge_lengths = edge_lengths
        elif repr=='fundamental':
            self.fundamental_edge_lengths = edge_lengths

        return edge_lengths
    
    @staticmethod
    def _edge_lengths_from_coords(edge_coords: npt.NDArray[np.float_]):
        assert edge_coords.shape[1]==6
        p0 = edge_coords[:,:3]
        p1 = edge_coords[:,3:]
        v = p1 - p0
        edge_lengths = np.linalg.norm(v, axis=1)
        return edge_lengths

    def calculate_node_distances(self, repr='reduced'):
        """
        Find the minimum distance between all pairs of nodes.

        Parameters:
            repr: 'reduced' or 'transformed'
        Returns:
            distances [num_nodes]: array of distances
            indices [num_nodes, 2]: indices of nodes corresponding to distances
        """
        if repr=='reduced':
            positions = self.reduced_node_coordinates
        elif repr=='transformed':
            self.calculate_transformed_coordinates()
            positions = self.transformed_node_coordinates
        else:
            raise ValueError('`repr` has to be `reduced` or `transformed`')

        assert positions.shape[0]==self.num_nodes

        matrix = (
            positions.reshape((self.num_nodes, 1, 3)) 
            - positions.reshape((1, self.num_nodes, 3))
        )
        distances = np.sqrt(np.sum(matrix**2, axis=2))
        indices = np.triu_indices_from(distances, k=1)
        distances = distances[indices]

        return distances, np.column_stack(indices)

    def calculate_node_types(self) -> Dict[str, Set]:
        """
        Calculate types of nodes from reduced representation

        Returns a dictionary with sets of 
            - corner nodes (3 d.o.f. lie on UC boundary)
            - edge nodes (2 d.o.f. lie on UC boundary)
            - face nodes (1 d.o.f. lies on UC boundary)
            - inner nodes (no d.o.f lie on UC boundary)
        """
        self.verify_num_nodes_edges()
        nodes = self.reduced_node_coordinates

        coords_on_bnds = (np.sum(np.abs(nodes)<self.TOL_DIST, axis=1)
                        + np.sum(np.abs(nodes-self.UC_L)<self.TOL_DIST, axis=1))

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
        """Calculate how many times nodes appear in the edge adjacency map.

        Returns:
            array[int] (num_nodes,): connectivity of each node
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
            - minimum reduced node coordinate is 0 and maximum is UC_L (1)
            - the only boundary nodes are face nodes
            - the connectivity of these nodes is 1

        See Also:
            :func:`calculate_node_types`,
            :func:`calculate_nodal_connectivity`
        """
        nodes = self.reduced_node_coordinates
        if (np.abs(nodes.min())>self.TOL_DIST
            or np.abs(nodes.max()-self.UC_L)>self.TOL_DIST):
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
        return True

    def get_periodic_partners(self) -> list:
        """Calculate periodic partners.

        Check is done whether lattice is in a valid window condition.

        Returns:
            periodic_partners: list of 2-element sets of node numbers
                of periodic partners
        """
        assert self.check_window_conditions()
        # node types and nodal connectivity 
        # have been calculated in the window conditions call
        TOL_PARTNER = 5e-4
        UC_L = self.UC_L
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
            dim = np.flatnonzero((np.abs(x)<self.TOL_DIST) | (np.abs(x-UC_L)<self.TOL_DIST))
            assert len(dim)==1
            dim = dim[0]
            if abs(x[dim])<self.TOL_DIST: 
                dim_partner = UC_L
            elif abs(x[dim]-UC_L)<self.TOL_DIST:
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

    def _pp_list_to_dict(self, pp_list : list) -> dict:
        """Create a dictionary map for periodic partners"""
        pp_dict = dict()
        for pp in pp_list:
            pp_l = list(pp)
            pp_dict[pp_l[0]] = pp_l[1]
            pp_dict[pp_l[1]] = pp_l[0]
        return pp_dict


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

    def crop_unit_cell(self) -> None:
        """Crop lattice to fit within unit cell window.

        Examples:

            >>> nodes = [[0,0,0],[1.5,0.5,0]]
            >>> edges = [[0,1]]
            >>> lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
            >>> lat.crop_unit_cell()

            .. plot::

                from utils import plotting
                from data import Lattice
                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle
                nodes = [[0,0,0],[1.5,0.5,0]]
                edges = [[0,1]]
                lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
                fig, axes = plt.subplots(ncols=2, figsize=(6,2), sharey=True)
                plotting.plot_unit_cell_2d(lat, ax=axes[0])
                lat.crop_unit_cell()
                plotting.plot_unit_cell_2d(lat, ax=axes[1])
                axes[0].set_title('Before cropping')
                axes[1].set_title('After cropping')
                for ax in axes: 
                    ax.axis('equal'); ax.set_xlim(-0.2,2); ax.set_ylim(-0.2,1.2)
                for ax in axes: 
                    rect = Rectangle((0,0), width=1, height=1, ec='k', fc='none')
                    ax.add_patch(rect)
                fig.tight_layout()
                axes

        See Also:
            :func:`create_windowed`
        """
        UC_L = self.UC_L

        if not hasattr(self, 'reduced_edge_coordinates'):
            self.update_representations('reduced_adjacency')
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

    def merge_colinear_edges(self) -> None:
        """Merge colinear edges based on nodal connectivity 2 
        and similar angle between two unit vectors along the edges.

        Example:
                
            >>> nodes = [[0,0,0],[1,0,0],[0.5,0,0]]
            >>> edges = [[0,2],[1,2]]
            >>> lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
            >>> lat
            {'num_nodes': 3, 'num_edges': 2}
            >>> lat.merge_colinear_edges()
            >>> lat
            {'num_nodes': 2, 'num_edges': 1}
                

            .. plot::
                
                from utils import plotting
                from data import Lattice
                import matplotlib.pyplot as plt
                nodes = [[0,0,0],[1,0,0],[0.5,0,0]]
                edges = [[0,2],[1,2]]
                lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
                fig, axes = plt.subplots(ncols=2, figsize=(6,2))
                plotting.plot_unit_cell_2d(lat, ax=axes[0])
                lat.merge_colinear_edges()
                plotting.plot_unit_cell_2d(lat, ax=axes[1])
                for ax in axes: ax.set_yticks([]); ax.set_ylabel('')
                axes[0].set_title('Before merging')
                axes[1].set_title('After merging')
                fig.tight_layout()
                axes

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
            merged_edge = []
            for line in lines:
                p1 = (set(line)-{min_dev_num}).pop()
                merged_edge.append(p1)
            new_edges = []
            new_edges.append(merged_edge)
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

            assert nodes_deleted<1000  # Avoid hanging while loop
        
        # Delete disconnected nodes
        self.reduced_edge_coordinates = self._node_adj_to_ec(
            self.reduced_node_coordinates, edges
        )
        self.update_representations(basis='reduced_edge_coords')
        
    def check_periodic_partners(self) -> bool:
        node_types = self.calculate_node_types()
        face_nodes_nums = list(node_types['face_nodes'])
        face_pts = self.reduced_node_coordinates[face_nodes_nums, :]
        for view_dim in [0,1,2]:
            dims = list({0,1,2} - {view_dim})
            planar_pts = face_pts[:, dims]
            mask_sides = np.any(
                (planar_pts<self.TOL_DIST) | (planar_pts>self.UC_L-self.TOL_DIST),
                axis=1
            )
            selected_pts = planar_pts[~mask_sides, :]
            _, cnts = np.unique(selected_pts, axis=0, return_counts=True)
            if not np.all(cnts==2):
                return False
        return True

    def obtain_shift_vector(
        self, max_num_attempts: int = 10, min_edge_length: float = 5e-3, 
        return_attempts: bool = False
    ) -> npt.NDArray:
        """Get a shift vector for windowed representation.

        Raises:
            WindowingException: _description_
            WindowingException: _description_

        Returns:
            npt.NDArray (3,): _description_
        """
        window_satisfied = self.check_window_conditions()
        if window_satisfied:
            shortest_edge = self.calculate_edge_lengths().min()
            shift_vector = np.zeros(3)
        niter = 0
        while ((not window_satisfied or shortest_edge<min_edge_length)
                and niter<max_num_attempts):
            temp_lattice = Lattice(**self.to_dict())
            shift_vector = np.random.rand(3)
            temp_lattice.reduced_node_coordinates = (
                temp_lattice.reduced_node_coordinates + shift_vector
            )
            temp_lattice.update_representations(basis='reduced_adjacency')
            temp_lattice.crop_unit_cell()
            temp_lattice.merge_colinear_edges()
            nodes_on_edges = temp_lattice.find_nodes_on_edges()
            if len(nodes_on_edges)>0:
                temp_lattice.split_edges_by_points(nodes_on_edges)
            edge_intersections = temp_lattice.find_nodes_on_edges()
            if len(edge_intersections)>0:
                temp_lattice.split_edges_by_points(edge_intersections)
            window_satisfied = temp_lattice.check_window_conditions()
            window_satisfied = window_satisfied and temp_lattice.check_periodic_partners()
            # if window_satisfied:
            #     _ = temp_lattice.get_periodic_partners()
            shortest_edge = temp_lattice.calculate_edge_lengths().min()
            
            niter += 1

        if (window_satisfied) and (shortest_edge>min_edge_length):
            if return_attempts:
                return shift_vector, niter
            else:
                return shift_vector
        else:
            raise WindowingException(f'Failed to obtain a window for lattice {self.name}')


    def create_windowed(self) -> "Lattice":
        """Create a windowed representation of a lattice

        Returns:
            Lattice: a new lattice instance

        Examples:
            >>> nodes = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
            >>> edges = [[0,1],[1,2],[2,3],[3,0]]
            >>> lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
            >>> lat
            {'num_nodes': 4, 'num_edges': 4}
            >>> lat_w = lat.create_windowed()
            >>> lat_w
            {'num_nodes': 5, 'num_edges': 4}

            .. plot::

                import matplotlib.pyplot as plt
                from data import Lattice
                from utils import plotting
                nodes = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
                edges = [[0,1],[1,2],[2,3],[3,0]]
                lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
                fig, axes = plt.subplots(ncols=2, figsize=(6,3))
                plotting.plot_unit_cell_2d(lat, ax=axes[0])
                lat_w = lat.create_windowed()
                plotting.plot_unit_cell_2d(lat_w, ax=axes[1])
                for ax in axes: ax.set_aspect('equal')
                axes[0].set_title('Original')
                axes[1].set_title('Windowed')
                fig.tight_layout()
                axes
        """
        shift_vector = self.obtain_shift_vector()
        newlat = Lattice(**self.to_dict())
        newlat.reduced_node_coordinates += shift_vector
        newlat.crop_unit_cell()
        newlat.merge_colinear_edges()
        nodes_on_edges = newlat.find_nodes_on_edges()
        if len(nodes_on_edges)>0:
            newlat.split_edges_by_points(nodes_on_edges)
        edge_intersections = newlat.find_nodes_on_edges()
        if len(edge_intersections)>0:
            newlat.split_edges_by_points(edge_intersections)
        assert newlat.check_window_conditions()
        return newlat


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


    def to_dict(self) -> dict:
        """Obtain a dictionary with the reduced representation."""
        d = dict()
        if hasattr(self, 'name'):
            d['name'] = self.name
        if hasattr(self, 'lattice_constants'):
            d['lattice_constants'] = self.lattice_constants
        d['reduced_node_coordinates'] = self.reduced_node_coordinates
        d['edge_adjacency'] = self.edge_adjacency
        if hasattr(self, 'compliance_tensors'):
            d['compliance_tensors'] = self.compliance_tensors
        return d


class WindowingException(Exception):
    pass