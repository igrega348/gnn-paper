import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
import numpy.typing as npt
from scipy.interpolate import griddata
from matplotlib import cm
import os
from torch.utils.data import ConcatDataset
import plotly.express as px
import plotly.graph_objects as go
from math import floor
from typing import Optional, Iterable


def parity_plot(true, pred, fn: str):
    fig, ax = plt.subplots()
    ax.scatter(true, pred, s=15, alpha=0.5)
    ax.axline((true.min(),true.min()),slope=1,c='k')
    ax.set_xlabel(f'True')
    ax.set_ylabel(f'Predicted')
    validAREL = np.mean(np.abs(pred-true)/true)
    ax.text(0.95,0.05,
        f'MAPE {100*validAREL:.1f}%',
        fontsize=12,
        transform=ax.transAxes,
        horizontalalignment='right') 
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn, dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.1)

def surface_plot(true, pred, dataset, nplot: int, fn: str):
    fig = plt.figure(figsize=(6,3*nplot))
    if isinstance(dataset, ConcatDataset):
        names = np.array([data.name for dset in dataset.datasets for data in dset])
    else:
        names = np.array(dataset.data.name)
    uq_names = np.unique(names)
    nplot = min(nplot, len(uq_names))
    plot_uq_names = np.random.choice(uq_names, nplot, replace=False)
    for i in range(nplot):
        name = plot_uq_names[i]
        idx = np.flatnonzero(names==name)
        phi = [dataset[k].phi.item() for k in idx]
        th = [dataset[k].th.item() for k in idx]
        ax = fig.add_subplot(nplot,2,1+(2*i), projection='3d')
        plot_surface_data(phi, th, true[idx], ax=ax, resolution=400j, clims=(1,2))
        ax = fig.add_subplot(nplot,2,2+(2*i), projection='3d')
        plot_surface_data(phi, th, pred[idx], ax=ax, resolution=400j, clims=(1,2))
    plt.subplot(nplot, 2, 1).set_title('Ground truth')
    plt.subplot(nplot, 2, 2).set_title('Prediction')
    plt.tight_layout()
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn, dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.1)


def plot_unit_cell_2d(
    lat, transformed=False, node_numbers=True,
    ax=None
    ) -> plt.Axes:
    if transformed:
        if not hasattr(lat, 'transformed_node_coordinates'):
            lat.calculate_transformed_coordinates()
            nodes = lat.transformed_node_coordinates
    else:
        nodes = lat.reduced_node_coordinates
    edges = lat.edge_adjacency
    
    if not isinstance(ax, plt.Axes):
        fig = plt.figure(figsize=(5,5),facecolor='w')
        ax = plt.axes()
    ax.scatter(nodes[:,0], nodes[:,1])
    for e in edges:
        line = nodes[e]
        ax.plot(line[:,0], line[:,1])
    if node_numbers:
        for ni, n in enumerate(nodes):
            ax.text(n[0],n[1],f"{ni}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax

def plot_unit_cell_3d(
    lat, repr='reduced', node_numbers=True,
    ax=None
    ) -> plt.Axes:
    if repr=='transformed':
        if not hasattr(lat, 'transformed_node_coordinates'):
            lat.calculate_transformed_coordinates()
        nodes = lat.transformed_node_coordinates
        edges = lat.edge_adjacency
    elif repr=='reduced':
        nodes = lat.reduced_node_coordinates
        edges = lat.edge_adjacency
    elif repr=='fundamental':
        if not hasattr(lat, 'fundamental_edge_adjacency'):
            lat.calculate_fundamental_representation()
        edge_coords = lat._node_adj_to_ec(
            lat.reduced_node_coordinates, lat.fundamental_edge_adjacency
        )
        edge_coords += lat.fundamental_tesselation_vecs
        nodes, edges = lat._ec_to_node_adj(edge_coords)
    else:
        raise NotImplementedError
    
    if not isinstance(ax, plt.Axes):
        fig = plt.figure(figsize=(5,5),facecolor='w')
        ax = plt.axes(projection='3d')
    ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2])
    segments = []
    colors = [] 
    for i_e, e in enumerate(edges):
        line = nodes[e]
        segments.append([(line[0,0], line[0,1], line[0,2]), 
                        (line[1,0], line[1,1], line[1,2])])
        colors.append(f'C{i_e%5}')
    lc = Line3DCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)
    if node_numbers:
        for ni, n in enumerate(nodes):
            ax.text(n[0],n[1],n[2],f"{ni}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax

def plotly_unit_cell_3d(
    lat, repr='reduced', node_numbers=False, 
    fig=None, subplot: Optional[dict] = None,
    highlight_nodes: Optional[Iterable] = None,
    show_uc_box: bool = False
    ):
    if repr=='transformed':
        if not hasattr(lat, 'transformed_node_coordinates'):
            lat.calculate_transformed_coordinates()
        nodes = lat.transformed_node_coordinates
        edges = lat.edge_adjacency
    elif repr=='reduced':
        nodes = lat.reduced_node_coordinates
        edges = lat.edge_adjacency
    elif repr=='fundamental':
        if not hasattr(lat, 'fundamental_edge_adjacency'):
            lat.calculate_fundamental_representation()
        edge_coords = lat._node_adj_to_ec(
            lat.reduced_node_coordinates, lat.fundamental_edge_adjacency
        )
        edge_coords += lat.fundamental_tesselation_vecs
        nodes, edges = lat._ec_to_node_adj(edge_coords)
    else:
        raise NotImplementedError

    colororder = px.colors.qualitative.G10

    x,y,z = nodes.T
    if isinstance(highlight_nodes, Iterable):
        colors = ['rgba(40,40,40,0.3)' for _ in range(len(x))]
        for i_node_highlight in highlight_nodes:
            colors[i_node_highlight] = 'rgb(255,0,0)'
    else:
        colors = [colororder[i%10] for i in range(len(x))]
    if not isinstance(fig, go.Figure):
        fig = go.Figure()
    mode = 'text+markers' if node_numbers else 'markers'
    if isinstance(subplot, dict):
        subplot_args = dict(
                row=floor(subplot['index']/subplot['ncols']) + 1,
                col=subplot['index']%subplot['ncols'] + 1
        )
    else:
        subplot_args = {}

    if show_uc_box:
        pts = np.array(
            [
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [0,1,0],
                [0,0,1],
                [1,0,1],
                [1,1,1],
                [0,1,1]
            ]
        )
        if repr=='transformed':
            pts = lat._transform_coordinates(pts, lat.get_transform_matrix())
        inds = [1,0,3,2,None,0,4,7,3,None,4,5,6,7,None,5,1,2,6]

        fig.add_scatter3d(
            x=[pts[i,0] if isinstance(i,int) else None for i in inds],
            y=[pts[i,1] if isinstance(i,int) else None for i in inds],
            z=[pts[i,2] if isinstance(i,int) else None for i in inds],
            mode='lines',
            line=dict(color='black', width=2),
            name='unit cell',
            showlegend=False,
            **subplot_args
        )                    

    fig.add_scatter3d(
        x=x, y=y, z=z,
        marker={'color':colors},
        mode=mode,
        text=np.arange(len(x)),
        textfont={'size':14},
        showlegend=False,
        name='nodes',
        **subplot_args
    )
    colors = [] 
    x = []
    y = []
    z = []
    for i_e, e in enumerate(edges):
        n0, n1 = nodes[e]
        x_0, y_0, z_0 = n0
        x_1, y_1, z_1 = n1
        x.extend([x_0, x_1, None])
        y.extend([y_0, y_1, None])
        z.extend([z_0, z_1, None])
        col = colororder[i_e%10]
        colors.extend([col, col, col])

    fig.add_scatter3d(
        x=x, y=y, z=z,
        line={'width':7,'color':colors},
        mode='lines',
        hoverinfo='none',
        connectgaps=False,
        showlegend=False,   
        **subplot_args
    )
    if isinstance(subplot, dict):
        fig.layout.annotations[subplot['index']].update(text=lat.name)
    else:
        fig.update_layout(title=lat.name)
    return fig

# %%
def visualize_graph(
    edges : npt.NDArray, nodes= None,  
    node_types=None, ax=None
    ) -> plt.Axes:
    if not isinstance(nodes, np.ndarray):
        nodes = np.unique(edges)
    colors = {'corner':'blue', 'edge':'red', 'face':'green', 'inside':'grey'}
    cmap = ['grey' for i in range(len(nodes))]
    if isinstance(node_types, dict):
        for ntype in node_types.keys():
            for n in node_types[ntype]:
                i = n
                cmap[i] = colors[ntype]
    edges_in = []
    edges_count = []
    for e in edges:
        e_set = set(e)
        if e_set in edges_in:
            edges_count[edges_in.index(e_set)] += 1
        else:
            edges_in.append(e_set)
            edges_count.append(1)
    edges_tuples = []
    for e_set in edges_in:
        for i in range(edges_count[edges_in.index(e_set)]):
            e = list(e_set)
            edges_tuples.append((e[0],e[1],{'r':i}))
    G = nx.Graph()
    G.add_nodes_from(np.arange(nodes.shape[0], dtype=int))
    G.add_edges_from(edges_tuples)
    #
    if isinstance(ax, plt.Axes):
        pass
    else:
        fig = plt.figure(facecolor='w')
        ax = plt.gca()
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos,  
        node_color = cmap, node_size = 200, alpha = 1, ax=ax
    )
    nx.draw_networkx_labels(G, pos, ax=ax)

    edges_in = []
    edges_count = []
    for j,e in enumerate(edges_tuples):
        ec="0"
        if len(e)>2: r=0.3*e[2]['r']
        else: r=0
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color=ec,
                                    shrinkA=10, shrinkB=10,
                                    patchA=None, patchB=None,
                                    connectionstyle=f"arc3,rad=rrr".replace('rrr',str(r)),
                                    ),
                    )
    plt.axis('off')
    return ax

# %%
def plot_surface_data(phi, th, val, ax=None, resolution=200j, clims=None):
    PHI, TH = np.mgrid[0:np.pi:resolution, 0:2*np.pi:resolution]
    R = griddata(np.column_stack((phi,th)), val, (PHI,TH))
    if not isinstance(ax, plt.Axes):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    z = R*np.cos(PHI)
    x = R*np.sin(PHI)*np.cos(TH)
    y = R*np.sin(PHI)*np.sin(TH)
    if isinstance(clims, tuple):
        color_val = (R-clims[0])/(clims[1]-clims[0])
        maxlim = 1.1*clims[1]
        ax.set_xlim(-maxlim, maxlim)
        ax.set_ylim(-maxlim, maxlim)
        ax.set_zlim(-maxlim, maxlim)
    else:
        color_val = (R-np.nanmin(R))/(np.nanmax(R)-np.nanmin(R))
    ax.plot_surface(x,y,z, facecolors=cm.viridis(color_val))
    return ax