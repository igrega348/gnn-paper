import dash_bootstrap_components as dbc
import datetime
import pandas as pd
from dash import Dash, html, dcc, Input, Output, dash_table, ctx
from dash.dash_table.Format import Format, Scheme, Trim
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
from torch_geometric.data import Batch

from data import elasticity_func
from data import Lattice, Catalogue
from exp_180.dash_mace import get_model
from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from utils import plotting

print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} : Loading model...')
class Model:
    def __init__(self):
        pass
    def model(self, batch: Batch):
        return {'stiffness': torch.eye(6).unsqueeze(0)}
# model = Model()
model = get_model()
print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} : Model loaded')
print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} : Loading catalogue...')
cat = Catalogue.from_file('./0imp_cat.lat', 0)
LATTICE_NAMES = {'octet': 'cub_Z12.0_E19', 'simple-cubic': 'cub_Z06.0_E1'}
LATTICE_INDEX = {'octet':274, 'simple-cubic': 281}

def load_lattice(_id: str or int):
    if _id in LATTICE_NAMES:
        _name = LATTICE_NAMES[_id]
    elif isinstance(_id, int):
        _name = _id
    else:
        _name = _id
    data = cat[_name]
    # pop fundamental edge adjacency and tesselation vecs
    data.pop('fundamental_edge_adjacency', None)
    data.pop('fundamental_tesselation_vecs', None)
    lat = Lattice(**data)
    lat = lat.create_windowed()
    lat.calculate_fundamental_representation()
    uq_nodes = np.unique(lat.fundamental_edge_adjacency)
    nodes = lat.reduced_node_coordinates[uq_nodes]
    edges = np.searchsorted(uq_nodes, lat.fundamental_edge_adjacency)
    edges = np.column_stack([edges, lat.fundamental_tesselation_vecs[:, 3:]])
    nodes = pd.DataFrame(nodes, columns=['x','y','z'])
    nodes['#'] = np.arange(len(nodes))
    edges = pd.DataFrame(edges, columns=['n0','n1', 'tx','ty','tz'])
    if 'compliance_tensors' in data:
        S = next(iter(data['compliance_tensors'].values()))
        C = np.linalg.inv(S)
        C_2 = elasticity_func.stiffness_Voigt_to_Mandel(C) * 10000
    else: 
        C_2 = None
    lattice_constants = lat.lattice_constants
    return nodes, edges, lattice_constants, C_2

# TODO: make dataframes for octet and simple cubic

print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} : Catalogue loaded')

DATA_TABLE_COLUMNS = [
    { "id": "#", "name": "#", "type": "numeric", 'format':Format(precision=2, scheme=Scheme.decimal_integer),},
    { "id": "x", "name": "x", "type": "numeric",},
    { "id": "y", "name": "y", "type": "numeric",},
    { "id": "z", "name": "z", "type": "numeric",},
]
EDGE_TABLE_COLUMNS = [
    { "id": "n0", "name": "n0", "type": "numeric", 'format':Format(precision=2, scheme=Scheme.decimal_integer),},
    { "id": "n1", "name": "n1", "type": "numeric", 'format':Format(precision=2, scheme=Scheme.decimal_integer),},
    { "id": "tx", "name": "tx", "type": "numeric",},
    { "id": "ty", "name": "ty", "type": "numeric",},
    { "id": "tz", "name": "tz", "type": "numeric",},
]
LAT_CONST_TABLE_COLUMNS = [
    { "id": "#", "name": "#"},
    { "id": "val", "name": "val", "type": "numeric",}
]

STIFF_PRED_TABLE_COLUMNS = [
    { "id": name, "name": name, "type": "numeric", 'format':Format(precision=2, scheme=Scheme.fixed, trim=Trim.yes),} for name in ['xx','yy','zz','yz','xz','xy']
]


DATA_TABLE_STYLE = {
    "style_header": {
        "color": "white",
        "backgroundColor": "#799DBF",
        "fontWeight": "bold",
    },
    # set font size
    "style_cell": {"font_size": "12px", "font-family": "monospace"},
}

# Default new row for datatable
new_line = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    '#': '0',
}
df_new_task_line = pd.DataFrame(new_line, index=[0])


get_default_table = lambda: load_lattice('simple-cubic')[0]
get_default_edges = lambda: load_lattice('simple-cubic')[1]


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
)

app.layout = dbc.Container(
    [
        html.Div(
            [
                dbc.Stack([
                    dbc.Button(
                        "Simple cubic", id="simple-cubic-btn", size="sm", outline=True, color="primary",
                    ),
                    dbc.Button(
                        "Octet", id="octet-btn", size="sm", outline=True, color="primary",
                    ),
                    dbc.Input(
                        id='cat-lat-index',
                        type='number',
                        placeholder='Enter index',
                        value=0,
                        style={'width': '100px'},
                        min=0,
                        max=len(cat)-1,
                        debounce=True, # TODO: change behaviour: remove load button, and change simple cubic and octet to change value of this box
                    ),
                ], direction='horizontal', gap=3),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H4("Nodes"),
                            dash_table.DataTable(
                                id="nodes-table",
                                sort_action="native",
                                columns=DATA_TABLE_COLUMNS,
                                data=get_default_table().to_dict("records"),
                                editable=True,
                                page_size=8,
                                row_deletable=True,
                                style_data_conditional=DATA_TABLE_STYLE.get("style_data_conditional"),
                                style_header=DATA_TABLE_STYLE.get("style_header"),
                                # set font size
                                style_cell=DATA_TABLE_STYLE.get("style_cell"),
                            ),
                            dbc.Button(
                                "Add node", id="add-row-btn", size="sm", outline=True, color="primary",
                            ),
                        ]
                    )
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H4("Edges"),
                            dash_table.DataTable(
                                id="edges-table",
                                sort_action="native",
                                columns=EDGE_TABLE_COLUMNS,
                                data=get_default_edges().to_dict("records"),
                                editable=True,
                                page_size=8,
                                row_deletable=True,
                                style_data_conditional=DATA_TABLE_STYLE.get("style_data_conditional"),
                                style_header=DATA_TABLE_STYLE.get("style_header"),
                                style_cell=DATA_TABLE_STYLE.get("style_cell"),
                            ),
                            dbc.Button(
                                "Add edge", id="add-edge-btn", size="sm", outline=True, color="primary",
                            ),
                        ]
                    )
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H4("Lattice constants"),
                            dash_table.DataTable(
                                id="lat-const-table",
                                sort_action="native",
                                columns=LAT_CONST_TABLE_COLUMNS,
                                data=[
                                    {"#": "a", "val": 1.0},
                                    {"#": "b", "val": 1.0},
                                    {"#": "c", "val": 1.0},
                                    {"#": "alpha", "val": 90.0},
                                    {"#": "beta", "val": 90.0},
                                    {"#": "gamma", "val": 90.0},
                                ],
                                editable=True,
                                page_size=6,
                                row_deletable=False,
                                style_data_conditional=DATA_TABLE_STYLE.get("style_data_conditional"),
                                style_header=DATA_TABLE_STYLE.get("style_header"),
                                style_cell=DATA_TABLE_STYLE.get("style_cell"),
                            ),
                        ]
                    )
                ),
                dbc.Col(
                html.Div(
                    [
                        html.H4("Unit cell"),
                        dcc.Graph(id="lattice-graph", style={'height': '450px', 'width': '500px'}),
                    ]
                ),
            ),
            ]
        ),
        dbc.Row([
            dbc.Col(
                    html.Div(
                        [
                            html.H4("Ground truth stiffness"),
                            dash_table.DataTable(
                                id="stiff-gt-table",
                                # sort_action="native",
                                data=[{col:None for col in ['xx','yy','zz','yz','xz','xy']} for _ in range(6)],
                                columns=STIFF_PRED_TABLE_COLUMNS,
                                editable=True,
                                page_size=6,
                                row_deletable=False,
                                style_data_conditional=DATA_TABLE_STYLE.get("style_data_conditional"),
                                style_header=DATA_TABLE_STYLE.get("style_header"),
                                style_cell=DATA_TABLE_STYLE.get("style_cell"),
                            ),
                        ]
                    )
                ),
            dbc.Col(
                html.Div(
                    [
                        html.H4("Stiffness"),
                        # set size of graph
                        dcc.Graph(id="stiffness-graph", style={'height': '450px', 'width': '500px'}),
                    ]
                ),
            ),
            dbc.Col(
                    html.Div(
                        [
                            html.H4("Predicted stiffness"),
                            dash_table.DataTable(
                                id="stiff-pred-table",
                                data=pd.DataFrame(
                                    np.eye(6), columns=['xx','yy','zz','yz','xz','xy']
                                ).to_dict("records"),
                                columns=STIFF_PRED_TABLE_COLUMNS,
                                editable=False,
                                page_size=6,
                                row_deletable=False,
                                style_data_conditional=DATA_TABLE_STYLE.get("style_data_conditional"),
                                style_header=DATA_TABLE_STYLE.get("style_header"),
                                style_cell=DATA_TABLE_STYLE.get("style_cell"),
                            ),
                        ]
                    )
                ),
            dbc.Col(
                html.Div(
                    [
                        html.H4("Predicted stiffness"),
                        # set size of graph
                        dcc.Graph(id="pred-stiffness-graph", style={'height': '450px', 'width': '500px'}),
                    ]
                ),
            ),
        ]),
    ],
    fluid=True,
)

@app.callback(
    Output("nodes-table", "data"),
    Output('edges-table', 'data'),
    Output('stiff-gt-table', 'data'),
    Output("stiff-pred-table", "data"),
    Output("lat-const-table", "data"),
    Output("lattice-graph", "figure"),
    Output("pred-stiffness-graph", "figure"),
    Output("stiffness-graph", "figure"),
    Output("cat-lat-index", "value"),
    Input("nodes-table", "derived_virtual_data"),
    Input("edges-table", "derived_virtual_data"),
    Input("lat-const-table", "derived_virtual_data"),
    Input('stiff-gt-table', 'derived_virtual_data'),
    Input('cat-lat-index', 'value'),
    Input("add-row-btn", "n_clicks"),
    Input("add-edge-btn", "n_clicks"),
    Input("simple-cubic-btn", "n_clicks"),
    Input("octet-btn", "n_clicks"),
)
def update_table_and_figure(
    user_datatable: None or list, edges_table: None or list, 
    lat_const_table: None or list, stiff_gt_table: None or list,
    cat_lat_index: int,
    n_clicks: int, n_clicks2: int, n_clicks3: int, n_clicks4: int,
):
    # if user deleted all rows, return the default row:
    if not user_datatable:
        updated_table = df_new_task_line
        updated_table["#"] = 0
    else:
        updated_table = pd.DataFrame(user_datatable)
        # if button clicked, then add a row
        if ctx.triggered_id == "add-row-btn":
            new_row = pd.DataFrame(df_new_task_line)
            new_row["#"] = updated_table["#"].max() + 1
            updated_table = pd.concat([updated_table, new_row])

    if not edges_table:
        updated_edges = get_default_edges()
        edges_table = updated_edges.to_dict("records")
    else:
        # if button clicked, then add an empty row
        if ctx.triggered_id == "add-edge-btn":
            edges_table.append({'n0':None, 'n1':None, 'tx':None, 'ty':None, 'tz':None})

    stiff_gt_table_pd = pd.DataFrame(stiff_gt_table)
    if stiff_gt_table_pd.isnull().values.any():
        C_gt = None
    else:
        C_gt = stiff_gt_table_pd.to_numpy()

    print(ctx.triggered_id)
    if ctx.triggered_id in ["octet-btn", "simple-cubic-btn", "cat-lat-index"]:
        if 'octet' in ctx.triggered_id:
            cat_lat_index = LATTICE_INDEX['octet']
        elif 'simple-cubic' in ctx.triggered_id:
            cat_lat_index = LATTICE_INDEX['simple-cubic']
        else:
            cat_lat_index = cat_lat_index
        updated_table, updated_edges, lattice_constants, C_gt = load_lattice(cat_lat_index)
        edges_table = updated_edges.to_dict("records")
        lat_const_table = [{'#':i, 'val':x} for x,i in zip(lattice_constants, ['a','b','c','alpha','beta','gamma'])]
    

    updated_table["#"] = updated_table["#"].astype(int)
    
    edge_adjacency = [[d['n0'], d['n1']] for d in edges_table if d['n0'] is not None and d['n1'] is not None and d['tx'] is not None and d['ty'] is not None and d['tz'] is not None]
    edge_tess_vecs = [[0,0,0,d['tx'], d['ty'], d['tz']] for d in edges_table if d['n0'] is not None and d['n1'] is not None and d['tx'] is not None and d['ty'] is not None and d['tz'] is not None]
    lat = Lattice(
        name='test', 
        nodal_positions=updated_table[['x','y','z']].values, 
        fundamental_edge_adjacency=edge_adjacency, 
        fundamental_tesselation_vecs=edge_tess_vecs,
        lattice_constants=[d['val'] for d in lat_const_table]
    )
    lat = lat.create_windowed()
    lat.calculate_fundamental_representation()
    lat_data = lat.to_dict()
    lat_data['compliance_tensors'] = {0.01: None, 0.05: None}
    pyg_data = GLAMM_Dataset.process_one(lat_data)
    # print(pyg_data)
    batch = Batch.from_data_list([pyg_data[0]])

    lattice_chart = plotting.plotly_unit_cell_3d(
        lat, repr='fundamental', coords='transformed',
        show_uc_box=True
    )
    # remove title and set figsize to (500, 500)
    lattice_chart.update_layout(
        title=None, #height=500, width=500,
    )
    
    with torch.no_grad():
        C2 = model.model(batch)
        # print(C2)
        C2 = C2['stiffness'][0].cpu()
    stiff_pred_table = pd.DataFrame(C2.numpy(), columns=['xx','yy','zz','yz','xz','xy']).to_dict("records")
    C4 = elasticity_func.stiffness_Mandel_to_cart_4(C2).numpy()
    pred_stiffness_chart = plotting.plotly_stiffness_surf(C4)
    if C_gt is not None:
        stiff_gt_table = pd.DataFrame(C_gt, columns=['xx','yy','zz','yz','xz','xy']).to_dict("records")
        C_gt = elasticity_func.stiffness_Mandel_to_cart_4(torch.from_numpy(C_gt)).numpy()
        stiffness_chart = plotting.plotly_stiffness_surf(C_gt)
    else:
        stiffness_chart = go.Figure()
        stiff_gt_table = [{col:None for col in ['xx','yy','zz','yz','xz','xy']} for _ in range(6)]

    return (
            updated_table.to_dict("records"), 
            edges_table, 
            stiff_gt_table, 
            stiff_pred_table, 
            lat_const_table,
            lattice_chart, 
            pred_stiffness_chart, 
            stiffness_chart,
            cat_lat_index,
        )
# TODO: we get trigger event twice - once for the buttons, once for the table update.
# Only replot if the table is updated

if __name__ == "__main__":
    app.run_server(debug=True)