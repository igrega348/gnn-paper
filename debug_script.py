from utils import plotting
from data import Lattice, Catalogue
from data.lattice import WindowingException
import numpy as np
from tqdm import tqdm
from plotly.subplots import make_subplots


nrows=1; ncols=1
# fig = make_subplots(
#     rows=nrows, cols=ncols,
#     subplot_titles=['t' for _ in range(nrows*ncols)],
#     specs=[[{"type": "scatter3d"} for _ in range(ncols)] for _ in range(nrows)]
# )
cat = Catalogue.from_file('./filtered_cat.lat', 0)
# tet_Z05.7_E2424
# names = ['cub_Z06.9_E14510']
# names = cat.names[7800:]
#### failed
# names = ['tet_Z06.5_E7537']
# names = ['cub_Z05.7_E12503']
# names = ['cub_Z05.1_E14181']
# names = ['cub_Z06.9_E11597']
failed = []
####
names = cat.names
pbar = tqdm(names)
for name in pbar:
    lat = Lattice(**cat.get_unit_cell(name))
    # fig = plotting.plotly_unit_cell_3d(lat, fig=fig, subplot=dict(nrows=nrows, ncols=ncols, index=0))
    try:
        dr = lat.obtain_shift_vector(max_num_attempts=5)
    except WindowingException:
        # print(f'Lattice {name} failed')
        failed.append(name)

    pbar.set_postfix({'Failed':len(failed)})
    # fig = plotting.plotly_unit_cell_3d(lat, fig=fig, subplot=dict(nrows=nrows, ncols=ncols, index=0))
    # lat.reduced_node_coordinates += dr
    # lat.crop_unit_cell()
    # fig = plotting.plotly_unit_cell_3d(lat, fig=fig, subplot=dict(nrows=nrows, ncols=ncols, index=1))

print(f'Failed lattices:')
print(failed)
# fig.show()