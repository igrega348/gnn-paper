# %%
import os
import json
from tqdm import tqdm
import numpy as np
from data import Catalogue, Lattice, elasticity_func
from utils import abaqus
# %%
cat = Catalogue.from_file('./imperf_cat_100.lat', 0)
print(cat)
abq_data_dir = 'C:/temp/gnn-paper/processed_data'

updated_cat_dict = dict()

for f in tqdm(os.listdir(abq_data_dir)):
    if not f.endswith('.json'):
        continue

    with open(os.path.join(abq_data_dir, f), 'r') as fin:
        sim_dict = json.load(fin)

    name = sim_dict['Lattice name']
    if len(sim_dict['Load-REF1-dof1'])<1: continue
        
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
    
cat = Catalogue.from_dict(updated_cat_dict)
print(cat)
cat.to_file('updated_cat.lat')