from tqdm import tqdm
import numpy as np
from data import Lattice, Catalogue
from data import WindowingError, PeriodicPartnersError
from utils import plotting, abaqus

cat = Catalogue.from_file('./filt_wind.lat', 0)
cat = cat[:1]

MAX_TRY = 10
IMP_KIND = 'sphere_surf'
NUM_RELDENS = 10

new_cat_name = 'lat_cat.lat'


job_num = 0

for lat_data in tqdm(cat):
    lat_data.pop('edge_adjacency')
    lat = Lattice(**lat_data)
    lat.crop_unit_cell()
    lat = lat.create_windowed()

    for imperfection_level in [0.01]:

        found = False
        ntry = 0
        while (ntry<MAX_TRY) and (not found):
            ntry += 1
            try:
                lat_imp = lat.apply_nodal_imperfections(
                    dr_mag=imperfection_level, kind=IMP_KIND
                )
                lat_imp.calculate_fundamental_representation()
                found = True
            except (PeriodicPartnersError, WindowingError):
                pass
        assert found, f'Lattice {lat.name} failed'

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

        lat_mesh = lat_imp.refine_mesh(0.2, 4)
        
        abaqus.write_abaqus_inp(
            lat_mesh, 
            loading=[(1,1,1.0),(1,2,1.0),(1,3,1.0),(2,1,0.5),(2,2,0.5),(2,3,0.5)],
            metadata={
                'Job name':f'{job_num:06d}',
                'Lattice name':lat_imp.name,
                'Base lattice':base_name,
                'Date':'2023-01-05', 
                'Relative densities': ', '.join([f'{rd:.4g}' for rd in relative_densities]),
                'Strut radii': ', '.join([f'{r:.4g}' for r in strut_radii]),
                'Description':f'Imperfection level {imperfection_level} with various relative densities',
                'Imperfection level':f'{imperfection_level}',
                'Catalogue':new_cat_name,
                'Hash':hsh
            }, 
            fname='./try_abq.inp'
        )
        
        job_num += 1


