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
# from exp_180.dash_mace import get_model
from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from utils import plotting

print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} : Loading model...')
class Model:
    def __init__(self):
        pass
    def model(self, batch: Batch):
        return {'stiffness': torch.eye(6).unsqueeze(0)}
model = Model()
# model = get_model()
print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} : Model loaded')
print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} : Loading catalogue...')
cat = Catalogue.from_file('./Unit_Cell_Catalog.txt', 1)
LATTICE_NAMES = {'octet': 'cub_Z12.0_E19', 'simple-cubic': 'cub_Z06.0_E1'}

def load_lattice(name: str):
    if name in LATTICE_NAMES:
        _name = LATTICE_NAMES[name]
    else:
        _name = name
    data = cat[_name]
    nodes = pd.DataFrame(data['nodal_positions'], columns=['x','y','z'])
    nodes['#'] = np.arange(len(nodes))
    edges = pd.DataFrame(data['edge_adjacency'], columns=['n0','n1'])
    return nodes, edges

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


def get_default_table():
    return pd.DataFrame(
        {'x':[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
         'y':[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
         'z':[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
         '#':np.arange(8),},
         index=np.arange(8)
    )

def get_default_edges():
    return pd.DataFrame(
        {'n0':[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
         'n1':[1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7],}
    )


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
)

app.layout = dbc.Container(
    [
        html.H1("Lattices", className="bg-primary text-white p-1 text-center"),
        html.Div(
            [
                dbc.Button(
                    "Simple cubic", n_clicks=0, id="simple-cubic-btn", size="sm"
                ),
                dbc.Button(
                    "Octet", n_clicks=0, id="octet-btn", size="sm"
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H4("Nodes"),
                            dbc.Button(
                                "Add node", n_clicks=0, id="add-row-btn", size="sm"
                            ),
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
                        ]
                    )
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H4("Edges"),
                            dbc.Button(
                                "Add edge", n_clicks=0, id="add-edge-btn", size="sm"
                            ),
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
                        ]
                    )
                ),
            ]
        ),
        dbc.Row([
            dbc.Col(
                html.Div(
                    [
                        html.H4("Unit cell"),
                        dcc.Graph(id="lattice-graph", style={'height': '500px', 'width': '500px'}),
                    ]
                ),
            ),
            dbc.Col(
                html.Div(
                    [
                        html.H4("Stiffness"),
                        # set size of graph
                        dcc.Graph(id="stiffness-graph", style={'height': '500px', 'width': '500px'}),
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
    Output("lattice-graph", "figure"),
    Output("stiffness-graph", "figure"),
    Input("nodes-table", "derived_virtual_data"),
    Input("edges-table", "derived_virtual_data"),
    Input("add-row-btn", "n_clicks"),
    Input("add-edge-btn", "n_clicks"),
    Input("simple-cubic-btn", "n_clicks"),
    Input("octet-btn", "n_clicks"),
)
def update_table_and_figure(user_datatable: None or list, edges_table: None or list, n_clicks: int, n_clicks2: int, n_clicks3: int, n_clicks4: int):
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
            edges_table.append({'n0':None, 'n1':None})

    if ctx.triggered_id == "octet-btn":
        updated_table, updated_edges = load_lattice('octet')
        edges_table = updated_edges.to_dict("records")
        
    if ctx.triggered_id == "simple-cubic-btn":
        updated_table, updated_edges = load_lattice('simple-cubic')        
        edges_table = updated_edges.to_dict("records")

    updated_table["#"] = updated_table["#"].astype(int)
    
    edge_adjacency = [[d['n0'], d['n1']] for d in edges_table if d['n0'] is not None and d['n1'] is not None]
    lat = Lattice(name='test', nodal_positions=updated_table[['x','y','z']].values, edge_adjacency=edge_adjacency)
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
    
    with torch.no_grad():
        C2 = model.model(batch)
        # print(C2)
        C2 = C2['stiffness'][0].cpu()
    C4 = elasticity_func.stiffness_Mandel_to_cart_4(C2).numpy()
    stiffness_chart = plotting.plotly_stiffness_surf(C4)

    return updated_table.to_dict("records"), edges_table, lattice_chart, stiffness_chart


if __name__ == "__main__":
    app.run_server(debug=True)