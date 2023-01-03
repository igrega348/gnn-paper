from utils import plotting
from data import Lattice
import matplotlib.pyplot as plt
nodes = [[0,0,0],[1,0,0],[0.5,0,0]]
edges = [[0,1]]
lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
fig, ax = plt.subplots(figsize=(4,2))
ax = plotting.plot_unit_cell_2d(lat, ax=ax)
ax.set_yticks([]); ax.set_ylabel('')
plt.tight_layout()
ax