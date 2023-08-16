# %%
import sys
import os
from io import BytesIO
import tarfile
import datetime
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import shutil

import numpy as np
from data import Lattice, Catalogue
from data import WindowingError, PeriodicPartnersError
from utils import abaqus
# %%
def process_one(lat_data: dict, dname: str) -> list:
    if not os.path.isdir(dname):
        os.makedirs(dname)

    MAX_TRY = 10
    IMP_KIND = 'sphere_surf'
    NUM_RELDENS = 10

    outputs = []

    base_job_num = lat_data.pop('base_job_num')
    loc_job_num = 0

    lat = Lattice(**lat_data)

    distances, dist_indices = lat.calculate_node_distances(repr='reduced')
    if distances.min()<0.05:
        return []

    modified = False
    nodes_on_edges = lat.find_nodes_on_edges()
    if nodes_on_edges:
        lat.split_edges_by_points(nodes_on_edges)
        modified = True
    edge_intersections = lat.find_edge_intersections()
    if edge_intersections:
        lat.split_edges_by_points(edge_intersections)
        modified = True
    
    # check nodal distances again
    distances, dist_indices = lat.calculate_node_distances(repr='reduced')
    if distances.min()<0.05:
        discarded += 1
        return []

    if modified: # safety check again
        if lat.find_nodes_on_edges() or lat.find_edge_intersections():
            discarded += 1
            return []

    try:
        _ = lat.obtain_shift_vector(max_num_attempts=3)
    except WindowingError:
        return []

    try:
        lat = lat.create_windowed()
        lat.calculate_fundamental_representation()
    except WindowingError:
        return []

    num_fundamental_nodes = lat.num_fundamental_nodes

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

            # cat_dict[lat_imp.name] = lat_dict

            lat_mesh = lat_imp.refine_mesh(0.2, 1)

            job_name = f'{base_job_num:06d}_{loc_job_num:02d}'

            abaqus.write_abaqus_inp(
                lat_mesh, 
                loading=[(1,1,1.0),(1,2,1.0),(1,3,1.0),(2,1,0.5),(2,2,0.5),(2,3,0.5)],
                strut_radii=strut_radii,
                metadata={
                    'Job name':job_name,
                    'Lattice name':lat_imp.name,
                    'Base lattice':base_name,
                    'Date':datetime.datetime.now().strftime("%Y-%m-%d"), 
                    'Relative densities': ', '.join([f'{rd:.5g}' for rd in relative_densities]),
                    'Strut radii': ', '.join([f'{sr:.5g}' for sr in strut_radii]),
                    'Unit cell volume':f'{lat_imp.calculate_UC_volume():.5g}',
                    'Description':f'Fresh dataset, all the same relative densities',
                    'Imperfection level':f'{imperfection_level}',
                    'Hash':hsh,
                },
                fname=os.path.join(dname, job_name+'.inp')
            )

            outputs.append(lat_dict)
            loc_job_num += 1

        if not found:
            break

    if not found:
        return []
    else:
        return outputs

def main():
    cat = Catalogue.from_file('./Unit_Cell_Catalog.txt', 1)
    print('Full catalogue: ', cat)
    # %%
    # process catalogue in 10 chunks
    num_cat = int(sys.argv[1])
    cat = cat[slice(num_cat, len(cat), 10)]
    print('Selected: ', cat)
    
    dname = 'C:/temp/new_dset_0'
    new_cat_name = f'{dname}/m_cat_{num_cat:02d}.lat'
    new_cat_dict = dict()

    base_job_num = 0

    input_dicts = [data for data in cat[:20]]
    # add job number
    for data in input_dicts:
        data['base_job_num'] = base_job_num
        base_job_num += 1

    process_partial = partial(process_one, dname=os.path.join(dname, f'input_files_{num_cat:02d}'))

    with Pool(processes=2) as p:
        results = list(tqdm(p.imap(process_partial, input_dicts), total=len(input_dicts), smoothing=0.1))

    for result in results:
        if not result:
            continue
        for cat_dict in result:
            name = cat_dict['name']
            new_cat_dict.update({name:cat_dict})

    new_cat = Catalogue.from_dict(new_cat_dict)
    print(new_cat)
    new_cat.to_file(new_cat_name)

    # use tarfile to compress the contents of the input_files folder
    with tarfile.open(os.path.join(dname, f'input_files_{num_cat:02d}.tar.gz'), 'w:gz') as archive:
        archive.add(os.path.join(dname, f'input_files_{num_cat:02d}'), arcname='input_files')

    # delete the input_files folder with all the input files
    shutil.rmtree(os.path.join(dname, f'input_files_{num_cat:02d}'))

# %%
if __name__=='__main__':
    main()