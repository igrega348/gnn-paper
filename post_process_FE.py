# %%
import os
import json
from tqdm import tqdm
import numpy as np
from data import Catalogue, Lattice, elasticity_func
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
cat_num = 3
cat = Catalogue.from_file(f'./imperf_cat_{cat_num}.lat', 0)
print(cat)
abq_archive_fn = f'C:/temp/gnn-paper/processed_data_{cat_num}.tar.gz'

updated_cat_dict = dict()
failed = []

with tarfile.open(abq_archive_fn, 'r:gz') as archive:
    members = archive.getmembers()
    for member in tqdm(members):
        if not member.name.endswith('.json'):
            continue
        fin = archive.extractfile(member)
        
        sim_dict = json.load(fin)
        name = sim_dict['Lattice name']

        if not check_data(sim_dict):
            failed.append(name)
            continue

            
        lat_dict = cat[sim_dict['Lattice name']]
        lat_dict.pop('edge_adjacency')
        lat = Lattice(**lat_dict)
        uc_volume = lat.calculate_UC_volume()
        rel_dens = float(sim_dict['Relative density'])

        S = abaqus.calculate_compliance_tensor(sim_dict, uc_volume)
        # symmetrise
        S = 0.5*(S + np.transpose(S, (2,3,0,1)))
        S = elasticity_func.compliance_4th_order_to_Voigt(S)


        if name not in updated_cat_dict:
            lat_dict = cat[sim_dict['Lattice name']]
            lat_dict['compliance_tensors'] = {rel_dens:S}
            updated_cat_dict[name] = lat_dict
        else:
            updated_cat_dict[name]['compliance_tensors'].update({rel_dens:S})
    
print(failed)
for name in failed:
    updated_cat_dict.pop(name, None)

cat = Catalogue.from_dict(updated_cat_dict)
print(cat)
cat.to_file(f'updated_cat_{cat_num}.lat')