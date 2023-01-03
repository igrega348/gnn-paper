from utils import plotting
from data import Lattice
import matplotlib.pyplot as plt
nodes = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
edges = [[0,2],[1,3]]
lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
fig, axes = plt.subplots(ncols=2, figsize=(6,3))
plotting.plot_unit_cell_2d(lat, ax=axes[0])
edge_intersections = lat.find_edge_intersections()
lat.split_edges_by_points(edge_intersections)
plotting.plot_unit_cell_2d(lat, ax=axes[1])
axes[0].set_title('Before splitting')
axes[1].set_title('After splitting')
fig.tight_layout()
axes