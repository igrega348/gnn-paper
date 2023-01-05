# %%
import numpy as np
import io
from typing import Union, Optional
from lattice import Lattice
# %%
def get_common_normal_guess(nodes : np.ndarray, edges : np.ndarray):
    assert nodes.shape[1]==3
    assert nodes.shape[0]>1
    assert edges.shape[1]==2
    n1 = edges[:,0]
    n2 = edges[:,1]
    edge_coords = np.column_stack((nodes[n1,:], nodes[n2,:]))
    edge_vecs = edge_coords[:,3:] - edge_coords[:,:3]
    edge_vecs_unit = edge_vecs / np.linalg.norm(edge_vecs, axis=1, keepdims=True)
    itry = 0
    accept = False
    while itry<20 and  (not accept):
        nvec = np.random.rand(3)
        nvec = nvec / np.linalg.norm(nvec)
        dprod = edge_vecs_unit @ nvec.reshape((3,1))
        dprod = np.abs(dprod)
        if np.all(dprod - 1)>1e-3: accept = True
    if not accept: return -1
    else:
        return nvec
# %%
def write_abaqus_inp(
    lat: Lattice, loading : list[tuple], 
    metadata : dict, fname : Optional[str] = None
    ):
    '''
    Write abaqus input script for a specific lattice and loading 

    Parameters:
            lat: Lattice
            loading: list of 3-element tuples where first element is reference point number 
                    second is degree of freedom to which displacement is applied
                    and third is the magnitude of displacement
            fname : name of input script file or stream to write to
            metadata: dictionary that will be written in header
    '''                
    lines = []
    lines.append('*Heading')
    for key in metadata:
        lines.append(f'**{key}: {metadata[key]}')
    lines.append('*Material, name=Material-1')
    lines.append('*Elastic')
    lines.append('1., 0.3')
    lines.append('** PARTS')
    lines.append('*Part, name=LATTICE')
    lines.append('*Node')
    nodes = lat.transformed_node_coordinates
    for k,node in enumerate(nodes):
        # Abaqus uses 1-indexing
        lines.append(f'{k+1}, {node[0]:.8f}, {node[1]:.8f}, {node[2]:.8f}')
    lines.append(f'*Element, type=B33, elset=FULL_LATTICE')
    edges = lat.edge_adjacency
    for k,edge in enumerate(edges):
        lines.append(f'{k+1}, {edge[0]+1}, {edge[1]+1}')
    lines.append(f'*Beam Section, elset=FULL_LATTICE, material=Material-1, section=CIRC')
    strut_radius = lat.edge_radii
    assert len(np.unique(strut_radius))==1
    strut_radius = strut_radius[0]
    lines.append(f'{strut_radius}') # radius of circular section
    # orientation of section
    normal_vec = get_common_normal_guess(nodes, edges)
    lines.append(f'{normal_vec[0]:.8f}, ' +
                    f'{normal_vec[1]:.8f}, ' + 
                    f'{normal_vec[2]:.8f}'
                )
    lines.append('* End Part')
    # REFERENCE POINTS
    lines.append('*Part, name=REF1')
    lines.append('*Node')
    lines.append('  1, -0.1, -0.1, -0.1')
    lines.append('*End Part')
    lines.append('*Part, name=REF2')
    lines.append('*Node')
    lines.append('  1, -0.2, -0.2, -0.2')
    lines.append('*End Part')
    # ASSEMBLY
    lines.append('** ASSEMBLY')
    lines.append('*Assembly, name=Assembly')
    lines.append('*Instance, name=LATTICE, part=LATTICE')
    lines.append('*End Instance')
    lines.append('*Instance, name=REF1-1, part=REF1')
    lines.append('*End Instance')
    lines.append('*Instance, name=REF2-1, part=REF2')
    lines.append('*End Instance')
    lines.append('*Nset, nset=REF1, instance=REF1-1')
    lines.append('  1,')
    lines.append('*Nset, nset=REF2, instance=REF2-1')
    lines.append('  1,')
    lines.append('*Nset, nset=REF-PTS')
    lines.append('  REF1, REF2')
    # periodic node sets and equations
    ###################################
    # REFERENCE POINT ASSIGNMENT
    # strain epsilon_ij is mapped to (refpointnum, refpointdeg) as:
    # eps_11, eps_22, eps_33, eps_12, eps_13, eps_23
    # (1,1) , (1,2) , (1,3) , (2,1) , (2,2) , (2,3)
    periodic_pairs = lat.get_periodic_partners()
    for ipair, pair_set in enumerate(periodic_pairs):
        lines.append(f'** Constraint {ipair}')
        pair_tup = tuple(pair_set)
        n1 = pair_tup[0]
        n2 = pair_tup[1]
        lines.append(f'*Nset, nset=PBC_NODE_{ipair}, instance=LATTICE')
        lines.append(f'  {n1+1},')
        lines.append(f'*Nset, nset=MIRROR_NODE_{ipair}, instance=LATTICE')
        lines.append(f'  {n2+1},')
        # 3 rotations
        for ideg in range(4,7):
            lines.append(f'*Equation')
            lines.append('2')
            lines.append(f'MIRROR_NODE_{ipair}, {ideg}, 1.')
            lines.append(f'PBC_NODE_{ipair}, {ideg}, -1.')
        # 
        dr = nodes[n2, :] - nodes[n1, :]
        dr = dr.reshape(3)
        # active degrees of freedom in constraint
        i_dr_nonzero = np.flatnonzero(np.abs(dr)>1e-3) 
        nactive = i_dr_nonzero.shape[0]
        assert nactive>0
        # 3 displacements
        for ideg in range(1,4):
            lines.append(f'*Equation')
            lines.append(f'{2+nactive}')
            lines.append(f'MIRROR_NODE_{ipair}, {ideg}, 1.')
            lines.append(f'PBC_NODE_{ipair}, {ideg}, -1.')
            for jdeg in i_dr_nonzero+1:
                ddr = dr[jdeg-1]
                refptnum=1 if jdeg==ideg else 2
                refptdeg=jdeg if jdeg==ideg else ideg + jdeg - 2
                lines.append(f'REF{refptnum}, {refptdeg}, {-ddr}')
    lines.append('*End Assembly')
    lines.append('** STEPS')
    for i_load, load in enumerate(loading):
        lines.append(f'*Step, name=Load-REF{load[0]}-dof{load[1]}, nlgeom=NO')
        lines.append('*Static')
        lines.append('1., 1., 1e-5, 1')
        lines.append('*Boundary, OP=NEW')
        lines.append(f'REF{load[0]}, {load[1]}, {load[1]}, {load[2]}')
        lines.append('*Restart, write, frequency=0')
        # lines.append('*Output, field')
        # lines.append('*Node Output, nset=REF-PTS')
        # lines.append('U, RF')
        # lines.append('*Output, history')

        lines.append('*Output, field, variable=PRESELECT')
        lines.append('*Output, history, variable=PRESELECT')
        lines.append('*END Step')

    lines = [line + '\n' for line in lines]
    
    if isinstance(fname, str):
        with open(fname, 'w') as fout:
            fout.writelines(lines)
    else: # return lines
        return lines
# %%
def calculate_compliance_tensor(odict : dict, uc_vol : float):
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
        if len(load_dict.keys())<1: failed==True; return []
        force = load_dict[f'REF{refpt}, RF{dof}']
        stress = force/uc_vol if refpt==1 else 0.5*force/uc_vol
        if stress==0: failed==True; return []
        if refpt==1:
            k=l=dof
        else:
            k=1 if dof<3 else 2
            l=2 if dof<2 else 3
        for i in range(1,4):
            for j in range(i,4):
                refptnum=1 if i==j else 2
                refptdeg=j if refptnum==1 else i + j - 2
                displacement = load_dict[f'REF{refptnum}, U{refptdeg}']
                strain = 2*displacement if i!=j else displacement
                fct = strain / stress if k==l else 0.5 * strain / stress
                S[i-1,j-1,k-1,l-1] = fct
                S[j-1,i-1,k-1,l-1] = fct
                S[i-1,j-1,l-1,k-1] = fct
                S[j-1,i-1,l-1,k-1] = fct

    return S