# %%
from math import ceil
import numpy as np
from typing import Optional, List, Tuple, Dict, Union, Iterable
# %%
def get_common_normal_guess(nodes : np.ndarray, edges : np.ndarray):
    assert nodes.shape[1]==3
    assert nodes.shape[0]>1
    assert edges.shape[1]==2
    n0 = edges[:,0]
    n1 = edges[:,1]
    edge_vecs = nodes[n1] - nodes[n0]
    edge_vecs_unit = edge_vecs / np.linalg.norm(edge_vecs, axis=1, keepdims=True)
    itry = 0
    accept = False
    while itry<20 and  (not accept):
        nvec = np.random.rand(3)
        nvec = nvec / np.linalg.norm(nvec)
        dotprod = edge_vecs_unit @ nvec.reshape((3,1))
        dotprod = np.abs(dotprod)
        if np.all(dotprod - 1)>1e-3: accept = True
    assert accept, "Could not find a common normal guess"
    return nvec
# %%
def guess_normals(t_vec: np.ndarray):
    t = t_vec / np.linalg.norm(t_vec)
    for i_try in range(2000):    
        guess = np.random.rand(3)
        guess = guess / np.linalg.norm(guess)
        dot = np.dot(guess, t)
        if np.abs(dot)<0.5:
            n1 = guess - t*dot
            n1 = n1 / np.linalg.norm(n1)
            n2 = np.cross(t, n1)
            return n1, n2
    raise RuntimeError(f'Normal not found for vector {t_vec}')
# %%
def mesh_lattice(
    nodes: np.ndarray, edges: np.ndarray, max_length: float, min_div: int
):
    meshed_nodes = []
    meshed_elements = []
    meshed_nodes.extend(nodes)

    for i_edge, e in enumerate(edges):
        node_0, node_1 = sorted(e)
        x0 = nodes[node_0]
        x1 = nodes[node_1]

        node_list = [node_0]

        vec = x1 - x0
        length = np.linalg.norm(vec)
        t_unit = vec/length
        num_div = max(min_div, ceil(length/max_length))
        L_step = length/num_div
        _x0 = x0
        for i in range(num_div-1):
            _x1 = _x0 + t_unit*L_step
            _n1 = len(meshed_nodes) # calculate len before node gets appended
            meshed_nodes.append(_x1)
            node_list.append(_n1)
            _x0 = _x1
        node_list.append(node_1)
        
        n1, n2 = guess_normals(t_vec=t_unit)

        meshed_elements.append({'node_sequence':tuple(node_list), 'n1':n1, 'n2':n2, 'physical_strut':(node_0, node_1)})
    
    meshed_nodes = np.row_stack(meshed_nodes)
    return meshed_nodes, meshed_elements
# %%
def write_abaqus_inp(
    lat: "Lattice", loading : List[Tuple], 
    strut_radii: Iterable,
    metadata : Dict[str, str], 
    fname : Optional[str] = None,
    element_type: Optional[str] = 'B33',
    ):
    """Write abaqus input script for a specific lattice and loading 

    Parameters:
            lat: Lattice
            loading: list of 3-element tuples where first element is reference point number 
                    second is degree of freedom to which displacement is applied
                    and third is the magnitude of displacement
            fname : name of input script file or stream to write to
            metadata: dictionary that will be written in header

    Args:
        lat (Lattice): Lattice object
        loading (List[Tuple]): list of 3-element tuples where first element \
            is reference point number, second is degree of freedom \
            to which displacement is applied \
            and third is the magnitude of displacement
        metadata (Dict[str, str]): Extra information to put in the *Header section
        fname (Optional[str], optional): if `fname` is provided, output \
            is written to file `fname`. Otherwise, return lines of text. \
            Defaults to None.

    """
    lines = []
    lines.append('*Heading')
    lines.append('** Start header')
    for key in metadata:
        lines.append(f'**{key}: {metadata[key]}')
    lines.append('** End header')
    lines.append('**')
    lines.append('**')
    lines.append('*Material, name=Material-1')
    lines.append('*Elastic')
    lines.append('1., 0.3')
    lines.append('**')
    lines.append('** PARTS')
    lines.append('**')
    
    lines.append(f'*Part, name=LATTICE')

    # no more calls to `lat` after this
    nodes = lat.transformed_node_coordinates
    edges = lat.edge_adjacency
    periodic_pairs = lat.get_periodic_partners()
    #

    lines.append('**')
    lines.append('*Node')
    for k,node in enumerate(nodes):
        # Abaqus uses 1-indexing
        lines.append(f'{k+1}, {node[0]:.8g}, {node[1]:.8g}, {node[2]:.8g}')

    lines.append('**')
    lines.append(f'*Element, type={element_type}, elset=FULL_LATTICE')
    for k,edge in enumerate(edges):
        lines.append(f'{k+1}, {edge[0]+1}, {edge[1]+1}')

    lines.append('**')
    lines.append('* End Part')

    lines.append('**')
    # REFERENCE POINTS
    for i_instance, _ in enumerate(strut_radii):
        lines.append(f'*Part, name=INST{i_instance}-REF1')
        lines.append('*Node')
        lines.append('  1, -0.1, -0.1, -0.1')
        lines.append('*End Part')
        lines.append(f'*Part, name=INST{i_instance}-REF2')
        lines.append('*Node')
        lines.append('  1, -0.2, -0.2, -0.2')
        lines.append('*End Part')
        lines.append('**')

    # ASSEMBLY
    lines.append('**')
    lines.append('** ASSEMBLY')
    lines.append('**')
    lines.append('*Assembly, name=Assembly')
    # for each instance:
    for i_instance, strut_radius in enumerate(strut_radii):
        lines.append(f'*Instance, name=INST{i_instance}-LAT, part=LATTICE')
        lines.append(f'{i_instance*1.5:.4g} 0 0')
        lines.append(f'*Beam Section, elset=FULL_LATTICE, material=Material-1, section=CIRC')
        # strut_radius = '_STRUT_RADIUS_PLACEHOLDER_'
        lines.append(f'{strut_radius}') # radius of circular section
        # orientation of section
        normal_vec = get_common_normal_guess(nodes, edges)
        lines.append(f'{normal_vec[0]:.5g}, ' +
                        f'{normal_vec[1]:.5g}, ' + 
                        f'{normal_vec[2]:.5g}'
                    )
        lines.append('**')
        lines.append('*End Instance')
        lines.append(f'*Instance, name=INST{i_instance}-REF1-A, part=INST{i_instance}-REF1')
        lines.append(f'{i_instance*1.5:.4g} 0 0')
        lines.append('*End Instance')
        lines.append(f'*Instance, name=INST{i_instance}-REF2-A, part=INST{i_instance}-REF2')
        lines.append(f'{i_instance*1.5:.4g} 0 0')
        lines.append('*End Instance')
        lines.append(f'*Nset, nset=INST{i_instance}-REF1, instance=INST{i_instance}-REF1-A')
        lines.append('  1,')
        lines.append(f'*Nset, nset=INST{i_instance}-REF2, instance=INST{i_instance}-REF2-A')
        lines.append('  1,')
        lines.append(f'*Nset, nset=INST{i_instance}-REF-PTS')
        lines.append(f'  INST{i_instance}-REF1, INST{i_instance}-REF2')
        lines.append('**')
        lines.append('** EQUATIONS')
        lines.append('**')
        # periodic node sets and equations
        ###################################
        # REFERENCE POINT ASSIGNMENT
        # strain epsilon_ij is mapped to (refpointnum, refpointdeg) as:
        # eps_11, eps_22, eps_33, eps_12, eps_13, eps_23
        # (1,1) , (1,2) , (1,3) , (2,1) , (2,2) , (2,3)
        for ipair, pair_set in enumerate(periodic_pairs):
            pair_tup = tuple(pair_set)
            n1 = pair_tup[0]
            n2 = pair_tup[1]
            lines.append(f'** INST{i_instance}: Constraint {ipair}, nodes {{{n1+1},{n2+1}}}')
            lines.append(f'*Nset, nset=LAT{i_instance}-PBC_NODE_{ipair}, instance=INST{i_instance}-LAT')
            lines.append(f'  {n1+1},')
            lines.append(f'*Nset, nset=LAT{i_instance}-MIRROR_NODE_{ipair}, instance=INST{i_instance}-LAT')
            lines.append(f'  {n2+1},')
            # 3 rotations
            for ideg in range(4,7):
                lines.append(f'*Equation')
                lines.append('2')
                lines.append(f'LAT{i_instance}-MIRROR_NODE_{ipair}, {ideg}, 1.')
                lines.append(f'LAT{i_instance}-PBC_NODE_{ipair}, {ideg}, -1.')
            # 
            dr = nodes[n2, :] - nodes[n1, :]
            # active degrees of freedom in constraint
            i_dr_nonzero = np.flatnonzero(np.abs(dr)>1e-3) 
            nactive = i_dr_nonzero.shape[0]
            assert nactive>0
            # 3 displacements
            for ideg in range(1,4):
                lines.append(f'*Equation')
                lines.append(f'{2+nactive}')
                lines.append(f'LAT{i_instance}-MIRROR_NODE_{ipair}, {ideg}, 1.')
                lines.append(f'LAT{i_instance}-PBC_NODE_{ipair}, {ideg}, -1.')
                for jdeg in i_dr_nonzero+1:
                    ddr = dr[jdeg-1]
                    refptnum=1 if jdeg==ideg else 2
                    refptdeg=jdeg if jdeg==ideg else ideg + jdeg - 2
                    lines.append(f'INST{i_instance}-REF{refptnum}, {refptdeg}, {-ddr}')
        lines.append('**')

    lines.append('*End Assembly')
    lines.append('**')
    lines.append('** STEPS')
    for i_load, load in enumerate(loading):
        lines.append(f'*Step, name=Load-REF{load[0]}-dof{load[1]}, nlgeom=NO')
        lines.append('*Static')
        lines.append('1., 1., 1e-5, 1')
        for i_instance, _ in enumerate(strut_radii):
            lines.append('*Boundary, OP=NEW')
            lines.append(f'INST{i_instance}-REF{load[0]}, {load[1]}, {load[1]}, {load[2]}')
        lines.append('*Restart, write, frequency=0')
        # lines.append('*Output, field')
        # lines.append('*Node Output, nset=REF-PTS')
        # lines.append('U, RF')
        # lines.append('*Output, history')
        lines.append('*Output, field, variable=PRESELECT')
        lines.append('*Output, history, variable=PRESELECT')
        lines.append('*End Step')
        lines.append('**')

    lines = [line + '\n' for line in lines]
    
    if isinstance(fname, str):
        with open(fname, 'w') as fout:
            fout.writelines(lines)
    else: 
        return lines
# %%
def write_abaqus_inp_normals(
    lat: "Lattice", loading : List[Tuple], 
    strut_radii: Iterable,
    metadata : Dict[str, str], 
    fname : Optional[str] = None,
    element_type: Optional[str] = 'B33',
    ):
    """Write abaqus input script for a specific lattice and loading 

    Parameters:
            lat: Lattice
            loading: list of 3-element tuples where first element is reference point number 
                    second is degree of freedom to which displacement is applied
                    and third is the magnitude of displacement
            fname : name of input script file or stream to write to
            metadata: dictionary that will be written in header

    Args:
        lat (Lattice): Lattice object
        loading (List[Tuple]): list of 3-element tuples where first element \
            is reference point number, second is degree of freedom \
            to which displacement is applied \
            and third is the magnitude of displacement
        metadata (Dict[str, str]): Extra information to put in the *Header section
        fname (Optional[str], optional): if `fname` is provided, output \
            is written to file `fname`. Otherwise, return lines of text. \
            Defaults to None.

    """
    lines = []
    lines.append('*Heading')
    lines.append('** Start header')
    for key in metadata:
        lines.append(f'**{key}: {metadata[key]}')
    lines.append('** End header')
    lines.append('**')
    lines.append('**')
    lines.append('*Material, name=Material-1')
    lines.append('*Elastic')
    lines.append('1., 0.3')
    lines.append('**')
    lines.append('** PARTS')
    lines.append('**')
    
    lines.append(f'*Part, name=LATTICE')

    # no more calls to `lat` after this
    nodes = lat.transformed_node_coordinates
    edges = lat.edge_adjacency
    periodic_pairs = lat.get_periodic_partners()
    meshed_nodes, meshed_elements = mesh_lattice(nodes, edges, max_length=1, min_div=3)
    assert np.allclose(nodes, meshed_nodes[:len(nodes)]) # make sure physical nodes are copied without changing order
    #

    lines.append('**')
    lines.append('*Node')
    for k,node in enumerate(meshed_nodes):
        # Abaqus uses 1-indexing
        lines.append(f'{k+1}, {node[0]:.8g}, {node[1]:.8g}, {node[2]:.8g}')

    lines.append('**')
    lines.append('** Elements with element sets')
    elem_num = 1
    for edge_dict in meshed_elements:
        node_0, node_1 = edge_dict['physical_strut']
        lines.append(f'*Element, type={element_type}, elset=STRUT_{node_0}-{node_1}')
        node_sequence = edge_dict['node_sequence']
        for _n0, _n1 in zip(node_sequence[:-1], node_sequence[1:]):
            lines.append(f'{elem_num}, {_n0+1}, {_n1+1}')
            elem_num += 1

    lines.append('**')
    lines.append('* End Part')

    lines.append('**')
    # REFERENCE POINTS
    for i_instance, _ in enumerate(strut_radii):
        lines.append(f'*Part, name=INST{i_instance}-REF1')
        lines.append('*Node')
        lines.append('  1, -0.1, -0.1, -0.1')
        lines.append('*End Part')
        lines.append(f'*Part, name=INST{i_instance}-REF2')
        lines.append('*Node')
        lines.append('  1, -0.2, -0.2, -0.2')
        lines.append('*End Part')
        lines.append('**')

    # ASSEMBLY
    lines.append('**')
    lines.append('** ASSEMBLY')
    lines.append('**')
    lines.append('*Assembly, name=Assembly')
    # for each instance:
    for i_instance, strut_radius in enumerate(strut_radii):
        lines.append(f'*Instance, name=INST{i_instance}-LAT, part=LATTICE')
        lines.append(f'{i_instance*1.5:.4g} 0 0')

        for edge_dict in meshed_elements:
            node_0, node_1 = edge_dict['physical_strut']
            lines.append(f'*Beam Section, elset=STRUT_{node_0}-{node_1}, material=Material-1, section=CIRC')
            lines.append(f'{strut_radius}') # radius of circular section
            n1 = edge_dict['n1']
            n2 = edge_dict['n2']
            # orientation of section
            lines.append(', '.join([f'{x:.5g}' for x in n1] + [f'{x:.5g}' for x in n2]))

        lines.append('**')
        lines.append('*End Instance')
        lines.append(f'*Instance, name=INST{i_instance}-REF1-A, part=INST{i_instance}-REF1')
        lines.append(f'{i_instance*1.5:.4g} 0 0')
        lines.append('*End Instance')
        lines.append(f'*Instance, name=INST{i_instance}-REF2-A, part=INST{i_instance}-REF2')
        lines.append(f'{i_instance*1.5:.4g} 0 0')
        lines.append('*End Instance')
        lines.append(f'*Nset, nset=INST{i_instance}-REF1, instance=INST{i_instance}-REF1-A')
        lines.append('  1,')
        lines.append(f'*Nset, nset=INST{i_instance}-REF2, instance=INST{i_instance}-REF2-A')
        lines.append('  1,')
        lines.append(f'*Nset, nset=INST{i_instance}-REF-PTS')
        lines.append(f'  INST{i_instance}-REF1, INST{i_instance}-REF2')
        lines.append('**')
        lines.append('** EQUATIONS')
        lines.append('**')
        # periodic node sets and equations
        ###################################
        # REFERENCE POINT ASSIGNMENT
        # strain epsilon_ij is mapped to (refpointnum, refpointdeg) as:
        # eps_11, eps_22, eps_33, eps_12, eps_13, eps_23
        # (1,1) , (1,2) , (1,3) , (2,1) , (2,2) , (2,3)
        for ipair, pair_set in enumerate(periodic_pairs):
            pair_tup = tuple(pair_set)
            n1 = pair_tup[0]
            n2 = pair_tup[1]
            lines.append(f'** INST{i_instance}: Constraint {ipair}, nodes {{{n1+1},{n2+1}}}')
            lines.append(f'*Nset, nset=LAT{i_instance}-PBC_NODE_{ipair}, instance=INST{i_instance}-LAT')
            lines.append(f'  {n1+1},')
            lines.append(f'*Nset, nset=LAT{i_instance}-MIRROR_NODE_{ipair}, instance=INST{i_instance}-LAT')
            lines.append(f'  {n2+1},')
            # 3 rotations
            for ideg in range(4,7):
                lines.append(f'*Equation')
                lines.append('2')
                lines.append(f'LAT{i_instance}-MIRROR_NODE_{ipair}, {ideg}, 1.')
                lines.append(f'LAT{i_instance}-PBC_NODE_{ipair}, {ideg}, -1.')
            # 
            dr = meshed_nodes[n2, :] - meshed_nodes[n1, :]
            # active degrees of freedom in constraint
            i_dr_nonzero = np.flatnonzero(np.abs(dr)>1e-3) 
            nactive = i_dr_nonzero.shape[0]
            assert nactive>0
            # 3 displacements
            for ideg in range(1,4):
                lines.append(f'*Equation')
                lines.append(f'{2+nactive}')
                lines.append(f'LAT{i_instance}-MIRROR_NODE_{ipair}, {ideg}, 1.')
                lines.append(f'LAT{i_instance}-PBC_NODE_{ipair}, {ideg}, -1.')
                for jdeg in i_dr_nonzero+1:
                    ddr = dr[jdeg-1]
                    refptnum=1 if jdeg==ideg else 2
                    refptdeg=jdeg if jdeg==ideg else ideg + jdeg - 2
                    lines.append(f'INST{i_instance}-REF{refptnum}, {refptdeg}, {-ddr}')
        lines.append('**')

    lines.append('*End Assembly')
    lines.append('**')
    lines.append('** STEPS')
    for i_load, load in enumerate(loading):
        lines.append(f'*Step, name=Load-REF{load[0]}-dof{load[1]}, nlgeom=NO')
        lines.append('*Static')
        lines.append('1., 1., 1e-5, 1')
        for i_instance, _ in enumerate(strut_radii):
            lines.append('*Boundary, OP=NEW')
            lines.append(f'INST{i_instance}-REF{load[0]}, {load[1]}, {load[1]}, {load[2]}')
        lines.append('*Restart, write, frequency=0')
        # lines.append('*Output, field')
        # lines.append('*Node Output, nset=REF-PTS')
        # lines.append('U, RF')
        # lines.append('*Output, history')
        lines.append('*Output, field, variable=PRESELECT')
        lines.append('*Output, history, variable=PRESELECT')
        lines.append('*End Step')
        lines.append('**')

    lines = [line + '\n' for line in lines]
    
    if isinstance(fname, str):
        with open(fname, 'w') as fout:
            fout.writelines(lines)
    else: 
        return lines
# %%
def calculate_compliance_tensor(
    odict : Dict[str, Dict[str, float]], uc_vol : float
) -> np.ndarray:
    # ordered loading cases = list[ tuple( reference point number, degree of freedom) ]
    # simulations use convention: 
    # eps_11,   eps_22,     eps_33,     eps_12,     eps_13,     eps_23
    # (1,1),    (1,2),      (1,3),      (2,1),      (2,2),      (2,3)
    # need to reorder to standard Voigt notation
    # eps_11, eps_22, eps_33, 2*eps_23, 2*eps_13, 2*eps_12
    ordered_loading_cases = [(1,1),(1,2),(1,3),(2,3),(2,2),(2,1)]
    # calculation in 4th order representation 
    S = np.zeros((3,3,3,3))
    failed = False
    for refpt, dof in ordered_loading_cases:
        load_dict = odict[f'Load-REF{refpt}-dof{dof}']
        if len(load_dict.keys())<1: 
            raise ValueError
        force = load_dict[f'REF{refpt}, RF{dof}']
        stress = force/uc_vol if refpt==1 else 0.5*force/uc_vol
        if stress==0: 
            raise ValueError
        # k,l are indices of stress: sig_kl
        if refpt==1:
            k=l=dof
        else:
            k=1 if dof<3 else 2
            l=2 if dof<2 else 3
        # i,j are indices of strain: eps_ij
        for i in range(1,4):
            for j in range(i,4):
                refptnum=1 if i==j else 2
                refptdeg=j if refptnum==1 else i + j - 2
                displacement = load_dict[f'REF{refptnum}, U{refptdeg}']
                strain = displacement
                # strain = 2*displacement if i!=j else displacement THIS WAS FALSE
                fct = strain / stress if k==l else 0.5 * strain / stress
                S[i-1,j-1,k-1,l-1] = fct
                S[j-1,i-1,k-1,l-1] = fct
                S[i-1,j-1,l-1,k-1] = fct
                S[j-1,i-1,l-1,k-1] = fct

    return S

def calculate_compliance_Voigt(
    odict : Dict[str, Dict[str, float]], uc_vol : float
) -> np.ndarray:
        # ordered loading cases = list[ tuple( reference point number, degree of freedom) ]
    # simulations use convention: 
    # eps_11,   eps_22,     eps_33,     eps_12,     eps_13,     eps_23
    # (1,1),    (1,2),      (1,3),      (2,1),      (2,2),      (2,3)
    # need to reorder to standard Voigt notation
    # eps_11, eps_22, eps_33, 2*eps_23, 2*eps_13, 2*eps_12
    ordered_loading_cases = [(1,1),(1,2),(1,3),(2,3),(2,2),(2,1)]
    # calculation in Voigt representation 
    S = np.zeros((6,6))
    for i_column in range(6):
        refpt, dof = ordered_loading_cases[i_column]

        load_dict = odict[f'Load-REF{refpt}-dof{dof}']
        if len(load_dict.keys())<1: 
            raise ValueError
        force = load_dict[f'REF{refpt}, RF{dof}']
        stress = force/uc_vol if refpt==1 else 0.5*force/uc_vol
        if stress==0: 
            raise ValueError

        
        for i_row in range(6):
            refptnum, refptdeg = ordered_loading_cases[i_row]
    
            displacement = load_dict[f'REF{refptnum}, U{refptdeg}']
            strain = displacement

            fct = strain / stress if i_row<3 else 2 * strain / stress
            
            S[i_row, i_column] = fct

    return S