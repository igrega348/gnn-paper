from utils import plotting
from data import Lattice
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
nodes = [[0,0,0],[1.5,0.5,0]]
edges = [[0,1]]
lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
fig, axes = plt.subplots(ncols=2, figsize=(6,2), sharey=True)
plotting.plot_unit_cell_2d(lat, ax=axes[0])
lat.crop_unit_cell()
plotting.plot_unit_cell_2d(lat, ax=axes[1])
axes[0].set_title('Before cropping')
axes[1].set_title('After cropping')
for ax in axes:
    ax.axis('equal'); ax.set_xlim(-0.2,2); ax.set_ylim(-0.2,1.2)
for ax in axes:
    rect = Rectangle((0,0), width=1, height=1, ec='k', fc='none')
    ax.add_patch(rect)
fig.tight_layout()
axes