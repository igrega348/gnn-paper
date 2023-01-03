import matplotlib.pyplot as plt
from data import Lattice
from utils import plotting
nodes = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
edges = [[0,1],[1,2],[2,3],[3,0]]
lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
fig, axes = plt.subplots(ncols=2, figsize=(6,3))
plotting.plot_unit_cell_2d(lat, ax=axes[0])
lat_w = lat.create_windowed()
plotting.plot_unit_cell_2d(lat_w, ax=axes[1])
for ax in axes: ax.set_aspect('equal')
axes[0].set_title('Original')
axes[1].set_title('Windowed')
fig.tight_layout()
axes