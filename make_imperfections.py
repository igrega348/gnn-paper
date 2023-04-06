import os
from io import BytesIO
import tarfile
from tqdm import tqdm

import numpy as np
from data import Lattice, Catalogue
from data import WindowingError, PeriodicPartnersError
from utils import plotting, abaqus


cat = Catalogue.from_file('./filt_wind.lat', 0)
# process catalogue in chunks of 500
# 0 0:500
# 1 500:1000
# 2 1000:1500
# 3 1500:2000
# 4 2000:2500
# 5 2500:3000
# 6 3000:3500
# 7 3500:4000
# 8 4000:4500
# 9 4500:5000
#10 5000:5500
#11 5500:6000
#12 6000:6500
#13 6500:7000
#14 7000:7500
#15 7500:8000
#16 8000:
num_cat = 0
cat = cat[500*num_cat:500*(num_cat+1)]

MAX_TRY = 10
IMP_KIND = 'sphere_surf'
NUM_RELDENS = 10

new_cat_name = f'./imperf_cat_{num_cat}.lat'
new_cat_dict = dict()

job_num = 1

with tarfile.open(f'./input_files1_cat_{num_cat}.tar.gz', 'w:gz') as archive:

    for lat_data in tqdm(cat):
        lat_data.pop('edge_adjacency')
        num_fundamental_nodes = len(np.unique(lat_data['fundamental_edge_adjacency']))
        lat = Lattice(**lat_data)
        lat.crop_unit_cell()
        lat = lat.create_windowed()

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

                relative_densities = 0.001 + 0.05*np.random.rand(NUM_RELDENS)
                relative_densities.sort()
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
                
                abq_input_lines = abaqus.write_abaqus_inp_normals(
                    lat_imp, 
                    loading=[(1,1,1.0),(1,2,1.0),(1,3,1.0),(2,1,0.5),(2,2,0.5),(2,3,0.5)],
                    strut_radii=strut_radii,
                    metadata={
                        'Job name':f'{job_num:06d}',
                        'Lattice name':lat_imp.name,
                        'Base lattice':base_name,
                        'Date':'2023-03-09', 
                        'Relative densities': ', '.join([f'{rd:.4g}' for rd in relative_densities]),
                        'Strut radii': ', '.join([f'{sr:.4g}' for sr in strut_radii]),
                        'Unit cell volume':f'{lat_imp.calculate_UC_volume():.5g}',
                        'Description':f'Simulations with imperfection level {imperfection_level}',
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