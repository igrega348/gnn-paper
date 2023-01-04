import matplotlib.pyplot as plt
from data import Lattice
from utils import plotting
nodes = [[0.2,0.3,0],[0.2,0,0],[1,0.3,0],[0.2,1,0],[0,0.3,0]]
edges = [[0,1],[0,2],[0,3],[0,4]]
lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
fig, axes = plt.subplots(ncols=2, figsize=(6,3))
plotting.plot_unit_cell_2d(lat, ax=axes[0])
lt = lat.create_tesselated(2,2,1)
plotting.plot_unit_cell_2d(lt, ax=axes[1])
for ax in axes: ax.set_aspect('equal');
fig.tight_layout()
axes