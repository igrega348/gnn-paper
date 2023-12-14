# %%
import os
import sys
import json
from tqdm import tqdm
import numpy as np
from data import Catalogue, elasticity_func
from utils import abaqus
import tarfile
# %%
def check_data(data: dict) -> bool:
    for refpt in [1,2]:
        for dof in [1,2,3]:
            if not f'Load-REF{refpt}-dof{dof}' in data:
                return False
            if len(data[f'Load-REF{refpt}-dof{dof}'])<12:
                return False
    return True
# %%
# write a function that takes a list of lattice names and outputs 
# a nested dictionary. Example:
# from list ['tet_Z05.6_E11_p_0.01_4978306909024421972','tet_Z05.6_E11_p_0.01_-5940306230906848563']
# create dictionary:
# {'tet_Z05.6_E11': {'p_0.01': {'4978306909024421972': 0, '-5940306230906848563': 0}
def create_dict(names: list) -> dict:
    d = dict()
    for name in names:
        lat_name, params = name.split('_p_')
        params = params.split('_')
        d.setdefault(lat_name, dict())
        d[lat_name].setdefault(params[0], dict())
        d[lat_name][params[0]][params[1]] = 0
    return d
# function that checks if the bottom level of the dictionary is all required_val
# return the list of top level keys that are not all required_val
def check_dict(d: dict, required_val: int = 10) -> list:
    failed = []
    for lat_name, lat_dict in d.items():
        for imp_level, imp_dict in lat_dict.items():
            if not all([v==required_val for v in imp_dict.values()]):
                failed.append(lat_name)
    return failed
# %%
def main(cat_num: int, post_script: str = ''):
    dname = 'E:/dset_4_B31'
    input_cat_fn = os.path.join(dname,f'm_cat_{cat_num:02d}{post_script}.lat')
    print('Loading catalogue from')
    print('\t', input_cat_fn)
    cat = Catalogue.from_file(input_cat_fn, 0)
    print(cat)
    input_dict = create_dict(cat.names)


    abq_archive_fn = os.path.join(dname, f'output_{cat_num:02d}.tar.gz')

    updated_cat_dict = dict()
    failed = []

    with tarfile.open(abq_archive_fn, 'r:gz') as archive:
        members = archive.getmembers()
        for member in tqdm(members):
            if not member.name.endswith('.json'):
                continue
            fin = archive.extractfile(member)
            
            try:
                sim_dict = json.load(fin)
            except json.decoder.JSONDecodeError:
                print(f'Failed to load {member.name}')
                continue
            
            name = sim_dict['Lattice name']
            if name not in cat.names:
                continue
            # break up name into base name and parameters
            lat_name, params = name.split('_p_')
            imp_level, nodal_hash = params.split('_')

            compliance_tensors = {}

            for instance_name, data in sim_dict['Instances'].items():

                uc_volume = float(sim_dict['Unit cell volume'])
                rel_dens = float(data['Relative density'])
                
                if not check_data(data):
                    failed.append(name)
                    continue
                

                S_m = abaqus.calculate_compliance_Mandel(data, uc_volume)
                if not np.all((S_m - S_m.T)/S_m.max() < 1e-3):
                    print(f'Compliance tensor is not symmetric for {name}' \
                    f' \n {np.around(S_m,0)}')
                    failed.append(name)
                    continue
                # symmetrise to machine precision
                S = 0.5*(S_m+S_m.T)

                compliance_tensors[rel_dens] = S
                
                # update input_dict if did not fail
                input_dict[lat_name][imp_level][nodal_hash] += 1

            lat_dict = cat[sim_dict['Lattice name']]
            # only save the last 3 columns
            assert len(lat_dict['fundamental_tesselation_vecs'][0]) == 6
            lat_dict['fundamental_tesselation_vecs'] = [x[3:] for x in lat_dict['fundamental_tesselation_vecs']]
            #
            lat_dict['compliance_tensors_M'] = compliance_tensors
            updated_cat_dict[name] = lat_dict
        
    failed_base = check_dict(input_dict, required_val=3)
    print(f'Base lattices that are kept: {len(input_dict)-len(failed_base)}')
    for name in cat.names:
        if name.split('_p_')[0] in failed_base:
            updated_cat_dict.pop(name, None)

    cat = Catalogue.from_dict(updated_cat_dict)
    print(cat)
    cat.to_file(os.path.join(dname, f'cat_{cat_num:02d}{post_script}.lat'))
# %%
if __name__=="__main__":
    cat_num = int(sys.argv[1])
    main(cat_num, f'_4_B31')