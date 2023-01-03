from utils import plotting
from data import Lattice
import matplotlib.pyplot as plt
nodes = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
edges = [[0,2],[1,3]]
lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
fig, ax = plt.subplots(figsize=(4,2))
ax = plotting.plot_unit_cell_2d(lat, ax=ax)
plt.tight_layout()
ax