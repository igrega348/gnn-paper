# %%
import os
import numpy as np
from tqdm import tqdm, trange
from catalogue import Catalogue, write_lattice
from lattice import Lattice, WindowingException
import plotting
import abq_custom_func
import copy
import matplotlib.pyplot as plt
from io import BytesIO
import tarfile
# %% Load cleaned catalogue
fn='./catalogue_sparse_windowed.lat'
cat = Catalogue.from_file(fn, indexing=0)
print(cat)
names = cat.names
# %%
with open('../../mace-lattices/20_names.txt','r') as fin:
    lines = fin.readlines()

names = ['_'.join(i.rstrip().split('_')[:3]) for i in lines]
print(f"In total {len(names)} topologies")
# # %%
# flagged = []
# pbar = trange(0,len(names))
# for j in pbar:
#     lattice = names[j]
#     lat = Lattice(**cat.get_unit_cell(lattice))
#     lat.window_lattice()
#     el = lat.calculate_edge_lengths(repr='fundamental')
#     e_min = el.min()
#     e_max = el.max()
#     ratio = e_min/e_max
#     if ratio<0.1:
#         flagged.append(lattice)
#         break
#     pbar.set_postfix(
#         flagged=len(flagged),
#         percentage=len(flagged)/(j+1)
#     )
# %%
newdata = dict()
job_num = 1544000
pbar = trange(0,len(names))
delta_bar = 0.0
new_cat_name = f'cat_20_reldens_{delta_bar}.lat'
num_imp_real = 1
num_reldens = 10
expected = len(names)*num_imp_real*num_reldens
print(f'Imp level {delta_bar}. Expected {expected} simulation files. Starting num {job_num}. Final num {job_num+expected}')
rel_densities = []
edge_radii = []
with tarfile.open('trytarfile.tar.gz', 'w:gz') as archive:
    for j in pbar:
        lattice = names[j]
        # if lattice not in ['tet_Z04.3_E11525']: continue

        for realisation in range(num_imp_real):

            lat = Lattice(**cat.get_unit_cell(lattice))
        
            # n_try = 0
            # keep_trying = True
            # while (n_try<10) and keep_trying:
            #     try:
            #         lat.perturb_inner_nodes(delta_bar, 'sphere_surf')
            #         keep_trying = False
            #     except Exception:
            #         print(f'Lattice {lattice} try {n_try} failed')
            #         lat = Lattice(**cat.get_unit_cell(lattice))
            #         pass
            #     finally:
            #         n_try += 1
            # if keep_trying:
            #     print(f'Lattice {lattice} failed')
            #     continue

            hsh = hash(lat.reduced_node_coordinates.tobytes())
            lat.name = f'{lattice}_p_{delta_bar}_{hsh}'
            newdata[lat.name] = lat.print_lattice_lines()
            lat.refine_mesh(0.2, 4)
            for rel_dens in 0.001+0.05*np.random.rand(num_reldens):
                rel_densities.append(rel_dens)
                lat.set_edge_radii(rel_dens)
                edge_radii.append(lat.edge_radii.mean())
                lines = abq_custom_func.write_abaqus_inp(
                    lat, 
                    [(1,1,1.0),(1,2,1.0),(1,3,1.0),(2,1,0.5),(2,2,0.5),(2,3,0.5)],
                    {'Job name':f'{job_num:06d}','Base lattice':lattice,
                    'Date':'2022-12-05', 'Relative density':rel_dens,
                    'Description':f'Imperfection level {delta_bar} with various relative densities',
                    'Imperfection level':f'{delta_bar}',
                    'Catalogue':new_cat_name,
                    'Hash':hsh}, 
                    # fname=os.path.join('C:/temp/input_files', str(job_num)+'.inp')
                )
                data = ''.join(lines).encode('utf8')
                out_stream = BytesIO(data)
                tarinfo = tarfile.TarInfo(name=os.path.join('./input_files', str(job_num)+'.inp'))
                tarinfo.size = out_stream.seek(0, os.SEEK_END)
                out_stream.seek(0)
                archive.addfile(tarinfo=tarinfo, fileobj=out_stream)
                job_num += 1
        # pbar.set_postfix(
        #     refresh=False
        #     )
# %%
cat_p = Catalogue.from_dict(newdata)
cat_p.to_file(new_cat_name)
# %%
