# %%
import os
import numpy as np
from tqdm import tqdm, trange
from catalogue import Catalogue, write_lattice
from lattice import Lattice, WindowingException
import plotting
import copy
import matplotlib.pyplot as plt
# %% Load new baseline catalogue
fn='./catalogue_v0.lat'
cat = Catalogue.from_file(fn, indexing=0)
print(cat)
names = cat.names
# %%
newdata = dict()
clustered = []
nodes_on_edges_lat = []
splitting_edges = []
unmodified = 0
MIN_NODE_DIST = 0.1
pbar = trange(0,len(names))
maxtry = 0
written = 0
maxlat = ''
for j in pbar:
    lattice = names[j]
    lat = Lattice(**cat.get_unit_cell(lattice))
    min_dist = lat.closest_node_distance()[0]
    if min_dist<MIN_NODE_DIST:
        continue
    modified = False
    nbef = lat.num_nodes
    ebef = lat.num_edges
    lat.collapse_nodes_onto_boundaries()
    #
    nodes_on_edges = lat.find_nodes_on_edges()
    if nodes_on_edges:
        modified = True
        nodes_on_edges_lat.append(lattice)
        lat.split_edges_by_nodes(nodes_on_edges)
        if lat.find_nodes_on_edges():
            print(f'{lattice} nodes on edges not fixed')
    #
    edge_int = lat.find_edge_intersections()
    if len(edge_int)>0:
        modified = True
        splitting_edges.append(lattice)
        lat.split_edges_at_intersections(edge_int)
        min_dist = lat.closest_node_distance()[0]
        if min_dist<MIN_NODE_DIST:
            continue
        edge_int = lat.find_edge_intersections()
        if len(edge_int)>0:
            print(f'{lattice} intersections not fixed')
    eafter = lat.num_edges
    nafter = lat.num_nodes
    if not modified:
        unmodified += 1
    # create window
    try:
        wlat = lat.create_windowed()
    except Exception:
        try:
            wlat = lat.create_windowed()
        except Exception:
            print(f'Lattice {lattice} failed')
            continue
    newdata[lattice] = wlat.print_lattice_lines()
    written += 1
    pbar.set_postfix(
        clustered=len(clustered), 
        nodes_on_edges=len(nodes_on_edges_lat), 
        edge_intersections=len(splitting_edges),
        unmodified=unmodified,
        written=written,
        refresh=False
        )
# %%
print(f'Writing catalogue of {len(newdata)} lattices to file')
newcat = Catalogue.from_dict(newdata)
newcat.to_file('./catalogue_sparse_windowed.lat')