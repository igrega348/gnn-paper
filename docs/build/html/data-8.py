import matplotlib.pyplot as plt
from utils import plotting
from data import Lattice
nodes = [[0,0.5,0.5],[1,0.5,0.5],[0.5,0,0.5],[0.5,1,0.5],[0.5,0.5,0.5]]
edges = [[0,4],[1,4],[2,4],[3,4]]
lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
fig, ax = plt.subplots(figsize=(3,3))
ax = plotting.plot_unit_cell_2d(lat, ax=ax)
ax.set_aspect('equal')
plt.tight_layout()
ax