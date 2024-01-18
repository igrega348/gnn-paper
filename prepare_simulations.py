# %%
import sys
import os
from io import BytesIO
import tarfile
from tqdm import tqdm

import numpy as np
from data import Lattice, Catalogue
from data import WindowingError, PeriodicPartnersError
from utils import plotting, abaqus
# %%
cat = Catalogue.from_file('./filt_wind.lat', 0)
names = cat.names
cat = Catalogue.from_file('./Unit_Cell_Catalog.txt', 1)
cat.names = names
print(cat)
# %%
# process catalogue in chunks of 500
# 0 0:1000
# 1 1000:2000
# 2 2000:3000
# 3 3000:4000
# 4 4000:5000
# 5 5000:6000
# 6 6000:7000
# 7 7000:8000
# 8 8000:
num_cat = 8
cat = cat[1000*num_cat:1000*(1+num_cat)]

MAX_TRY = 10
IMP_KIND = 'sphere_surf'
NUM_RELDENS = 1

new_cat_name = f'C:/temp/gnn-paper-onerd/imperf_cat_{num_cat}_0.01.lat'
new_cat_dict = dict()

job_num = 0

with tarfile.open(f'C:/temp/gnn-paper-onerd/input_files_cat_{num_cat}_0.01.tar.gz', 'w:gz') as archive:

    for lat_data in tqdm(cat):
        
        lat = Lattice(**lat_data)
        try:
            lat = lat.create_windowed()
        except WindowingError:
            continue
        lat.calculate_fundamental_representation()
        num_fundamental_nodes = len(np.unique(lat.fundamental_edge_adjacency))
        # lat.crop_unit_cell()


        if num_fundamental_nodes==1:
            imp_levels = [0.0]
        else:
            imp_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

        for imperfection_level in imp_levels:

            if imperfection_level==0.0:
                num_imperf = 1
            else:
                num_imperf = 10

            for i_imperf in range(num_imperf):

                found = False
                ntry = 0
                while (ntry<MAX_TRY) and (not found):
                    ntry += 1
                    try:
                        lat_imp = lat.apply_nodal_imperfections(
                            dr_mag=imperfection_level, kind=IMP_KIND
                        )
                        lat_imp = lat_imp.create_windowed()
                        lat_imp.calculate_fundamental_representation()
                        found = True
                    except (PeriodicPartnersError, WindowingError):
                        pass
                if not found:
                    print(f'Lattice {lat.name} failed')
                    break

                r0 = np.sqrt(0.001)
                r1 = np.sqrt(0.05)
                r = np.linspace(r0,r1,NUM_RELDENS)
                relative_densities = r**2
                strut_radii = [lat_imp.calculate_edge_radius(rel_dens) for rel_dens in relative_densities]

                hsh = hash(lat_imp.reduced_node_coordinates.tobytes())
                base_name = lat_imp.name
                lat_imp.name = lat_imp.name + f'_p_{imperfection_level}_{hsh}'

                lat_dict = lat_imp.to_dict()
                lat_dict['base_name'] = base_name
                lat_dict['imperfection_level'] = imperfection_level
                lat_dict['imperfection_kind'] = IMP_KIND
                lat_dict['nodal_hash'] = hsh

                new_cat_dict[lat_imp.name] = lat_dict

                lat_mesh = lat_imp.refine_mesh(0.2, 1)
                
                abq_input_lines = abaqus.write_abaqus_inp(
                    lat_mesh, 
                    loading=[(1,1,1.0),(1,2,1.0),(1,3,1.0),(2,1,0.5),(2,2,0.5),(2,3,0.5)],
                    strut_radii=strut_radii,
                    metadata={
                        'Job name':f'{job_num:06d}',
                        'Lattice name':lat_imp.name,
                        'Base lattice':base_name,
                        'Date':'2023-06-14', 
                        'Relative densities': ', '.join([f'{rd:.4g}' for rd in relative_densities]),
                        'Strut radii': ', '.join([f'{sr:.4g}' for sr in strut_radii]),
                        'Unit cell volume':f'{lat_imp.UC_volume:.5g}',
                        'Description':f'All lattices at 1% relative density',
                        'Imperfection level':f'{imperfection_level}',
                        'Catalogue':new_cat_name,
                        'Hash':hsh
                    },
                )

                data = ''.join(abq_input_lines).encode('utf8')
                out_stream = BytesIO(data)
                tarinfo = tarfile.TarInfo(name=f'./input_files/{job_num:06d}.inp')
                tarinfo.size = out_stream.seek(0, os.SEEK_END)
                out_stream.seek(0)
                archive.addfile(tarinfo=tarinfo, fileobj=out_stream)
                
                job_num += 1

            if not found:
                break

new_cat = Catalogue.from_dict(new_cat_dict)
new_cat.to_file(new_cat_name)
# %%
