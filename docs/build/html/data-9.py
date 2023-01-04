from utils import plotting
from data import Lattice
import matplotlib.pyplot as plt
nodes = [[0,0,0],[1,0,0],[0.5,0,0]]
edges = [[0,2],[1,2]]
lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
fig, axes = plt.subplots(ncols=2, figsize=(6,2))
plotting.plot_unit_cell_2d(lat, ax=axes[0])
lat.merge_colinear_edges()
plotting.plot_unit_cell_2d(lat, ax=axes[1])
for ax in axes: ax.set_yticks([]); ax.set_ylabel('')
axes[0].set_title('Before merging')
axes[1].set_title('After merging')
fig.tight_layout()
axes