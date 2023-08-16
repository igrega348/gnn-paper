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
# function that checks if the bottom level of the dictionary is all 10
# return the list of top level keys that are not all 10
def check_dict(d: dict, required_val: int = 10) -> list:
    failed = []
    for lat_name, lat_dict in d.items():
        for imp_level, imp_dict in lat_dict.items():
            if not all([v==required_val for v in imp_dict.values()]):
                failed.append(lat_name)
    return failed
# %%
def main(cat_num: int):
    dname = 'E:/new_dset_1/'
    input_cat_fn = os.path.join(dname, f'm_cat_{cat_num:02d}.lat')
    print('Loading catalogue from')
    print('\t', input_cat_fn)
    cat = Catalogue.from_file(input_cat_fn, 0)
    print(cat)
    input_dict = create_dict(cat.names)


    abq_archive_fn = os.path.join(dname, f'processed_data_{cat_num}.tar.gz')

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
                # update input_dict
                input_dict[lat_name][imp_level][nodal_hash] += 1

                S = abaqus.calculate_compliance_Voigt(data, uc_volume)
                # symmetrise
                S = 0.5*(S+S.T)

                compliance_tensors[rel_dens] = S

            lat_dict = cat[sim_dict['Lattice name']]
            lat_dict['compliance_tensors'] = compliance_tensors
            updated_cat_dict[name] = lat_dict
        
    failed_base = check_dict(input_dict)
    print(f'Base lattices that are kept: {len(input_dict)-len(failed_base)}')
    for name in cat.names:
        if name.split('_p_')[0] in failed_base:
            updated_cat_dict.pop(name, None)
    # print(set(failed))
    # for name in failed:
    #     updated_cat_dict.pop(name, None)

    cat = Catalogue.from_dict(updated_cat_dict)
    print(cat)
    cat.to_file(os.path.join(dname, f'cat_{cat_num:02d}.lat'))
# %%
if __name__=="__main__":
    cat_num = int(sys.argv[1])
    main(cat_num)
# %%
