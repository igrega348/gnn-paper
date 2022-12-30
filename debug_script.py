from data import Catalogue
from data import Lattice
# from lattice import Lattice
from utils import plotting
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
np.seterr('raise')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

cat = Catalogue.from_file('./Unit_Cell_Catalog.txt', indexing=1)
print(cat)

selected = []
print(f'Catalogue size: {len(cat)}')
k = 0
df_data = {}
for name in tqdm(cat.names):
    lat = Lattice(**cat.get_unit_cell(name))
    # min_dist,_ = lat.closest_node_distance('transformed')
    distances, _ = lat.calculate_node_distances('transformed')
    min_dist = distances.min()
    if min_dist<0.05:
        if k<16:
            k += 1
    else:
        selected.append(name)
        df_data.update({name:{
            'min_dist':min_dist, 
            'num_nodes':lat.num_nodes, 
            'num_edges':lat.num_edges
            }
        })

df = pd.DataFrame(df_data).T
print(f'Catalogue size: {len(selected)}')
df = df.sort_values(by='min_dist')
df.head()

fig = make_subplots(
    rows=1, cols=4, 
    subplot_titles=['t' for _ in range(4)],
    specs=[[{"type": "scatter3d"} for _ in range(4)]]
)
k = 0
skipped = 0
pbar = tqdm(selected)
selected = []
for name in pbar:
    lat = Lattice(**cat.get_unit_cell(name))
    nodes_on_edges = lat.find_nodes_on_edges()
    modifed = False
    if nodes_on_edges:
        lat.split_edges_by_points(nodes_on_edges)
        modified = True
    edge_intersections = lat.find_edge_intersections()
    if edge_intersections:
        lat.split_edges_by_points(edge_intersections)
        modifed = True
    # check nodal distances
    distances, _ = lat.calculate_node_distances('reduced')
    if distances.min()<0.05:
        if skipped<4:
            fig = plotting.plotly_unit_cell_3d(lat, 'transformed', node_numbers=True, fig=fig, subplot=dict(nrows=1, ncols=4, index=skipped), show_uc_box=False)
        else:
            pass
        skipped += 1
        continue
    if modifed:
        k += 1
        # check that succeeded
        if lat.find_nodes_on_edges() or lat.find_edge_intersections():
            print(f'Lattice {name} failed')
            break
    selected.append(name)
    pbar.set_postfix({'Modified lattices':k, 'Skipped':skipped})

print(f'Catalogue size: {len(selected)}')
fig.show()

    # if len(edge_int)>0:
    #     print(edge_int)
    #     fig = plotting.plotly_unit_cell_3d(lat)
    #     fig.show()
    #     lat.split_edges_by_points(edge_int)
    #     fig = plotting.plotly_unit_cell_3d(lat)
    #     fig.show()
    #     if lat.find_edge_intersections():
    #         print(f'Lattice {name} failed')
    #         break
    #     # highlight_nodes = np.concatenate([tup[1] for tup in nodes_on_edges]) 
    #     k += 1
        
