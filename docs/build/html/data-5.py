from utils import plotting
from data import Lattice
import matplotlib.pyplot as plt
nodes = [[0,0,0],[1,0,0],[0.5,0,0]]
edges = [[0,1]]
lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
fig, axes = plt.subplots(ncols=2, figsize=(6,2))
plotting.plot_unit_cell_2d(lat, ax=axes[0])
nodes_on_edges = lat.find_nodes_on_edges()
lat.split_edges_by_points(nodes_on_edges)
plotting.plot_unit_cell_2d(lat, ax=axes[1])
for ax in axes: ax.set_yticks([]); ax.set_ylabel('')
axes[0].set_title('Before splitting')
axes[1].set_title('After splitting')
fig.tight_layout()
axes