import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import Lattice, Catalogue
from utils import plotting, abaqus



# lat = Lattice.make_simple_cubic()

# # plotting.plot_unit_cell_3d(lat)
# # plt.show()

# lat.calculate_fundamental_representation()
# edge_radius = lat.calculate_edge_radius(0.01)
# # create a normally distributed random vector around edge_radius
# edge_radii = np.random.normal(edge_radius, 0.2*edge_radius, lat.num_fundamental_edges)
# # df = pd.DataFrame({'x':[], 'c':[]})
# # for c in [0.1, 0.5, 1.0]:
# #     edge_radii = np.random.normal(c, 0.2*c, 1000)
# #     edge_radii[edge_radii < 0.2*c] = 0.2*c
# #     df = pd.concat([df, pd.DataFrame({'x':edge_radii, 'c':c})], axis=0, ignore_index=True, )

# # sns.histplot(data=df, x='x', hue='c', stat='density', common_norm=False, kde=True, kde_kws={'cut':0}, bins=50)
# # plt.show()

# print(edge_radii)
# lat.set_fundamental_edge_radii(edge_radii)
# # print(lat.fundamental_edge_radii)
# # print(lat.windowed_edge_radii)

# abaqus.write_abaqus_inp_normals(
#     lat, strut_radii=lat.windowed_edge_radii[:,None], metadata={'name':'simple_cubic_test'},
#     fname='E:/abq_trials/test.inp'
# )

cat = Catalogue.from_file('./Unit_Cell_Catalog.txt', 1, regex=r'\bcub_.*') # only cubic
lats = {}

lat = Lattice(**cat['cub_Z12.0_E19']) # octet

lat = lat.create_windowed()
lat.calculate_fundamental_representation()
lat_dict = lat.to_dict()
lat_dict['fundamental_edge_radii'] = {}

r0 = np.sqrt(0.001)
r1 = np.sqrt(0.05)
r = np.linspace(r0,r1,50)
relative_densities = r**2
windowed_edge_radii = []
real_relative_densities = []

for base_rel_dens in relative_densities:
    edge_radius = lat.calculate_edge_radius(base_rel_dens)
    # # create a normally distributed random vector around edge_radius
    print(f'base_rel_dens: {base_rel_dens}, edge_radius: {edge_radius}')

    edge_radii = np.random.normal(edge_radius, 0.2*edge_radius, lat.num_fundamental_edges)
    edge_radii[edge_radii < 0.2*edge_radius] = 0.2*edge_radius

    # print(edge_radii)
    lat.set_fundamental_edge_radii(edge_radii)
    windowed_edge_radii.append(lat.windowed_edge_radii)
    
    rel_dens = lat.calculate_relative_density()
    real_relative_densities.append(rel_dens)
    lat_dict['fundamental_edge_radii'].update({rel_dens:lat.fundamental_edge_radii})

abaqus.write_abaqus_inp_normals(
    lat, 
    strut_radii=np.column_stack(windowed_edge_radii),
    metadata={
        'Job name':'octet_50',
        'Lattice name':lat.name,
        'Base lattice':lat.name,
        'Date':datetime.datetime.now().strftime("%Y-%m-%d"), 
        'Relative densities': ', '.join([f'{rd:.5g}' for rd in real_relative_densities]),
        'Unit cell volume':f'{lat.calculate_UC_volume():.5g}',
        'Description':f'Non-uniform strut radii',
    },
    fname='E:/abq_trials/octet_nonuniform.inp'
)


lats = {lat_dict['name']:lat_dict}
cat = Catalogue.from_dict(lats)
cat.to_file('E:/abq_trials/octet_nonuniform.lat')

outputs = abaqus.run_abq_sim(['octet_nonuniform'], 'E:/abq_trials')
print(outputs)