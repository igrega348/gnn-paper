import matplotlib.pyplot as plt
from utils import plotting
from data import Lattice
nodes = [[0.3,0.3,0.5],[0.7,0.3,0.5],[0.3,0.7,0.5],[0.7,0.7,0.5]]
fundamental_edge_adjacency=[[0,1],[1,3],[2,3],[0,2],[2,0],[3,1],[1,0],[3,2]]
tess_vecs = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
             [0,0,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,0,0],[0,0,0,1,0,0]]
lat = Lattice(
    nodal_positions=nodes,
    fundamental_edge_adjacency=fundamental_edge_adjacency,
    fundamental_tesselation_vecs=tess_vecs
)
lat.crop_unit_cell()
fig, axes = plt.subplots(ncols=2, figsize=(6,3), sharey=True)
plotting.plot_unit_cell_2d(lat, ax=axes[0])
lat_imp = lat.apply_nodal_imperfections(0.1, 'sphere_surf')
plotting.plot_unit_cell_2d(lat_imp, ax=axes[1])
for ax in axes: ax.set_aspect('equal')
axes[1].set_ylabel('')
fig.tight_layout()
axes