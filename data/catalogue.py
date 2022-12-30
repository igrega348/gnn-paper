from collections.abc import Sequence

class Catalogue:
    names: Sequence
    lines: dict
    INDEXING: int

    def __init__(self, data: dict, indexing: int) -> None:
        self.lines = data
        self.names = list(data.keys())
        self.INDEXING = indexing
        

    @classmethod
    def from_file(cls, fn: str, indexing: int):
        """
        Read the combined catalog with specific formatting

        See PNAS paper by Lumpe, T. S. and Stankovic, T. (2020) at
        https://www.pnas.org/doi/10.1073/pnas.2003504118.
        Catalog can be downloaded from 
        https://doi.org/10.3929/ethz-b-000457598.
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
    def from_dict(cls, data: dict) -> None:
        return cls(data=data, indexing=0)

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        desc = "Unit cell catalogue "\
            f"with {len(self.names)} entries"
        return desc

    def get_unit_cell(self, name: str) -> dict:
        """
        Return a dictionary which represents unit cell.

        Returned dictionary contains keys:
            name
            symmetry
            code: E- or R- based code according to databases 
                (see the catalogue website for more info)
            lattice_constants: [a,b,c,alpha,beta,gamma]
            average_connectivity: average connectivity
            nodal_positions: nested list of shape (num_nodes, 3)
            edge_adjacency: nested list of shape (num_edges, 2)
                            0-indexed 
        """
        lines = self.lines[name]

        uc_dict = {}
        assert name in lines[0]
        uc_dict['name'] = name
        fields = name.split('_')
        sym = fields[0]
        uc_dict['symmetry'] = sym
        code = fields[2]
        uc_dict['code'] = code
        
        compl_start = compl_end = None
        scaling_exp_line = youngs_line = None

        for i_line, line in enumerate(lines):
            if 'unit cell parameters' in line:
                l_1 = lines[i_line+1]
                lat_params = [float(w) for w in l_1.split(',')]
                uc_dict['lattice_constants'] = lat_params
            elif 'connectivity' in line:
                z = float(line.split('Z_avg = ')[1])
                uc_dict['average_connectivity'] = z
            elif 'Nodal positions' in line:
                np_start = i_line
            elif 'Bar connectivities' in line:
                adj_start = i_line
            elif 'Compliance tensors start' in line:
                compl_start = i_line
            elif 'Compliance tensors end' in line:
                compl_end = i_line

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
                    compliance_tensors[rel_dens] = nums
            uc_dict['compliance_tensors'] = compliance_tensors

        nodal_coords = []
        for i_line in range(np_start+1, adj_start):
            line = lines[i_line]
            assert (i_line==adj_start-1) or (len(line)>1)
            if len(line)>1:
                nc = [float(w) for w in line.split()]
                nodal_coords.append(nc)
        uc_dict['nodal_positions'] = nodal_coords

        edge_adjacency = []
        for i_line in range(adj_start+1, len(lines)):
            line = lines[i_line]
            assert (i_line==len(lines)-1) or (len(line)>1)
            if len(line)>1:
                ea = [int(w)-self.INDEXING for w in line.split()]
                edge_adjacency.append(ea)
        uc_dict['edge_adjacency'] = edge_adjacency

        return uc_dict

    def to_file(self, fn: str) -> None:
        outlines = []

        for name in self.lines.keys():
            outlines.append('----- lattice_transition -----')
            outlines.extend(self.lines[name])
        
        outlines = [line + '\n' for line in outlines]
        with open(fn, 'w') as fout:
            fout.writelines(outlines)