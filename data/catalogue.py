from typing import Dict, Iterable, List, Union
import numpy as np

class Catalogue:
    """Unit cell catalogue object.

    Note:
        Two class methods are available to initialise the object:
            - :func:`from_file`
                Used to read the catalogue from a file

                >>> cat = Catalogue.from_file('Unit_Cell_Catalog.txt', 1)

            - :func:`from_dict`
                Used to create the catalogue from unit cells either from scratch
                or when unit cells from a file are modified

                >>> nodes = [[0,0,0],[1,0,0],[0.5,1,0],[0.5,1,1]]
                >>> edges = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3]]
                >>> lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
                >>> lat
                {'num_nodes': 4, 'num_edges': 6}
                >>> cat_dict = {'pyramid':lat.to_dict()}
                >>> cat = Catalogue.from_dict(cat_dict)


                See Also:
                    :func:`Lattice.to_dict`
    """
    names: List
    lines: dict
    INDEXING: int
    iter: int

    def __init__(self, data: dict, indexing: int) -> None:
        self.lines = data
        self.names = list(data.keys())
        self.INDEXING = indexing

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        desc = "Unit cell catalogue "\
            f"with {len(self.names)} entries"
        return desc

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter<len(self):
            data = self.get_unit_cell(self.names[self.iter])
            self.iter += 1
            return data
        else:
            raise StopIteration

    def __getitem__(self, ind: Union[int, slice]):
        if isinstance(ind, int):
            return self.get_unit_cell(self.names[ind])
        elif isinstance(ind, slice):
            selected_names = self.names[ind]
            selected_data = {name: self.lines[name] for name in selected_names}
            return Catalogue(data=selected_data, indexing=self.INDEXING)
        else:
            raise NotImplementedError


    @classmethod
    def from_file(cls, fn: str, indexing: int) -> "Catalogue":
        """Read catalogue from a file.

        Args:
            fn (str): path to input file
            indexing (int): 0 or 1 as the basis of edge indexing

        Returns:
            Catalogue
        """
        with open(fn, 'r') as fin:
            lines = fin.readlines()
        lines = [line.rstrip() for line in lines]
        line_ranges = dict()
        data = dict()

        name = None
        start_line = None
        end_line = None
    
        for i_line in range(len(lines)):
            line = lines[i_line]

            if line.startswith('Name:'):
                phrases = line.split()
                name = phrases[1]
                start_line = i_line
                end_line = None
            elif 'lattice_transition' in line:
                if not isinstance(name, str): continue
                end_line = i_line
                line_ranges[name] = slice(start_line, end_line)
                name = None
                start_line = None
                end_line = None
            else:
                pass
        if isinstance(name, str) and (not isinstance(end_line, int)):
            assert isinstance(start_line, int)
            end_line = i_line+1
            line_ranges[name] = slice(start_line, end_line)

        for name in line_ranges.keys():
            text = lines[line_ranges[name]]
            data[name] = text
        
        return cls(data=data, indexing=indexing)

    @classmethod
    def from_dict(cls, lattice_dicts: Dict[str, Dict]) -> "Catalogue":
        """Generate unit cell catalogue from dictionary representation

        Args:
            lattice_dicts (Dict[str, Dict]): dictionary of lattice dictionaries

        Returns:
            Catalogue

        Note:
            Lattice dictionaries must contain the following keys:
                - `edge_adjacency`
                - `nodal_positions` or `reduced_node_coordinates`
            They can contain also:
                - `lattice_constants`
                - `compliance_tensors`
        """
        data = dict()

        for name in lattice_dicts:
            lat_dict = lattice_dicts[name]
            lines = []

            if 'name' in lat_dict:
                assert lat_dict['name']==name

            lines.append(f'Name: {name}')
            lines.append('')

            if 'lattice_constants' in lat_dict:
                lattice_constants = lat_dict['lattice_constants']
                assert len(lattice_constants)==6
                lines.append(
                    'Normalized unit cell parameters (a,b,c,alpha,beta,gamma):'
                )
                line = ''
                for i, x in enumerate(lattice_constants):
                    line = line + f'{x:.5f}'
                    if i!=5:
                        line = line + ', '
                lines.append(line)

            if 'compliance_tensors' in lat_dict:
                compliance_tensors = lat_dict['compliance_tensors']
                lines.append('')
                lines.append('Compliance tensors start (flattened upper triangular)')
                for rel_dens in compliance_tensors:
                    lines.append(f'-> at relative density {rel_dens}:')
                    S = compliance_tensors[rel_dens]
                    assert S.shape==(6,6)
                    nums = S[np.triu_indices(6)].tolist()
                    line = ''
                    for s in nums:
                        line = line + f'{s:.5g},'
                    lines.append(line)
                lines.append('Compliance tensors end')
                lines.append('')
            
            assert ('reduced_node_coordinates' in lat_dict) or ('nodal_positions' in lat_dict)
            assert 'edge_adjacency' in lat_dict
            
            if 'reduced_node_coordinates' in lat_dict:
                nodal_positions = lat_dict['reduced_node_coordinates']
            else:
                nodal_positions = lat_dict['nodal_positions']

            lines.append('Nodal positions:')
            for x,y,z in nodal_positions:
                lines.append(f'{x:.5f} {y:.5f} {z:.5f}')

            lines.append('')
            lines.append('Bar connectivities:')
            for e in lat_dict['edge_adjacency']:
                lines.append(f'{int(e[0])} {int(e[1])}')
            lines.append('')
        
            data.update({name:lines})

        return cls(data=data, indexing=0)

    def get_unit_cell(self, name: str) -> dict:
        """Return a dictionary which represents unit cell.

        Args:
            name (str): Name of the unit cell from the catalogue that 
                will be returned
            
        Returns:
            dict: Dictionary describing the unit cell

        Note:
            Returned dictionary contains all available keys from the following:
                - `name`
                - `lattice constants`: [a,b,c,alpha,beta,gamma]
                - `average connectivity`
                - `compliance_tensors`
                - `nodal_positions`: nested list of shape (num_nodes, 3)
                - `edge_adjacency`: nested list of shape (num_edges, 2) (0-indexed)

            The dictionary can be unpacked in the creation of a `Lattice` object

                >>> from data import Lattice, Catalogue
                >>> cat = Catalogue.from_file('Unit_Cell_Catalog.txt', 1)
                >>> lat = Lattice(**cat.get_unit_cell(cat.names[0]))
                >>> lat
                {'name': 'cub_Z06.0_E1', 'num_nodes': 8, 'num_edges': 12}

        See Also:
            :func:`data.Lattice.__init__`
        """
        lines = self.lines[name]

        uc_dict = {}
        assert name in lines[0]
        uc_dict['name'] = name
        
        compl_start = compl_end = None

        for i_line, line in enumerate(lines):
            if 'unit cell parameters' in line:
                l_1 = lines[i_line+1]
                lat_params = [float(w) for w in l_1.split(',')]
                uc_dict['lattice_constants'] = lat_params
            elif 'connectivity' in line:
                z = float(line.split('Z_avg = ')[1])
                uc_dict['average_connectivity'] = z
            elif 'Nodal positions' in line:
                nod_pos_start = i_line
            elif 'Bar connectivities' in line:
                edge_adj_start = i_line
            elif 'Compliance tensors start' in line:
                compl_start = i_line
            elif 'Compliance tensors end' in line:
                compl_end = i_line

        nodal_coords = []
        for i_line in range(nod_pos_start+1, edge_adj_start):
            line = lines[i_line]
            assert (i_line==edge_adj_start-1) or (len(line)>1)
            if len(line)>1:
                nc = [float(w) for w in line.split()]
                nodal_coords.append(nc)
        uc_dict['nodal_positions'] = nodal_coords

        edge_adjacency = []
        for i_line in range(edge_adj_start+1, len(lines)):
            line = lines[i_line]
            assert (i_line==len(lines)-1) or (len(line)>1)
            if len(line)>1:
                ea = [int(w)-self.INDEXING for w in line.split()]
                edge_adjacency.append(ea)
        uc_dict['edge_adjacency'] = edge_adjacency

        if isinstance(compl_start, int):
            assert isinstance(compl_end, int)
            compliance_tensors = dict()
            for i_line in range(compl_start+1, compl_end):
                line = lines[i_line]
                if 'at relative density' in line:
                    rel_dens = float(line.rstrip(':').split('density')[1])
                    line = lines[i_line+1]
                    nums = []
                    for num in line.split(','):
                        try:
                            s = float(num)
                        except ValueError:
                            continue
                        nums.append(s)
                    assert len(nums)==21
                    S = np.zeros((6,6))
                    S[np.triu_indices(6)] = nums
                    S[np.triu_indices(6)[::-1]] = nums
                    compliance_tensors[rel_dens] = S
            uc_dict['compliance_tensors'] = compliance_tensors

        return uc_dict

    def to_file(self, fn: str) -> None:
        """Export unit cell catalogue to file.

        Args:
            fn (str): output file path
        """
        outlines = []

        for name in self.lines.keys():
            outlines.append('----- lattice_transition -----')
            outlines.extend(self.lines[name])
        
        outlines = [line + '\n' for line in outlines]
        with open(fn, 'w') as fout:
            fout.writelines(outlines)