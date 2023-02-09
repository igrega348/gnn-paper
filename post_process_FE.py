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
def main(cat_num: int):
    # cat_num = 6
    input_cat_fn = f'C:/temp/gnn-paper/imperf_cat_{cat_num}.lat'
    print('Loading catalogue from')
    print('\t', input_cat_fn)
    cat = Catalogue.from_file(input_cat_fn, 0)
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

            compliance_tensors = {}

            for instance_name, data in sim_dict['Instances'].items():

                uc_volume = float(sim_dict['Unit cell volume'])
                rel_dens = float(data['Relative density'])
                
                if not check_data(data):
                    failed.append(name)
                    continue

                S = abaqus.calculate_compliance_tensor(data, uc_volume)
                # symmetrise
                S = 0.5*(S + np.transpose(S, (2,3,0,1)))
                S = elasticity_func.compliance_4th_order_to_Voigt(S)

                compliance_tensors[rel_dens] = S

            lat_dict = cat[sim_dict['Lattice name']]
            lat_dict['compliance_tensors'] = compliance_tensors
            updated_cat_dict[name] = lat_dict
        
    print(set(failed))
    for name in failed:
        updated_cat_dict.pop(name, None)

    cat = Catalogue.from_dict(updated_cat_dict)
    print(cat)
    cat.to_file(f'C:/temp/gnn-paper/cat_{cat_num}.lat')

if __name__=="__main__":
    cat_num = int(sys.argv[1])
    main(cat_num)