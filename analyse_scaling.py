# %%
from argparse import Namespace
import logging
import os
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelSummary, 
    ModelCheckpoint, 
    TQDMProgressBar
) 
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from e3nn import o3
from e3nn.io import CartesianTensor
import wandb            
import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from data import elasticity_func, Catalogue
from gnn.models import PositiveGNN
from gnn.callbacks import PrintTableMetrics, upload_evaluations
from utils import plotting
# %%
class RotateLat:
    def __call__(self, lat: Data):
        Q = o3.rand_matrix()
        C_rot = torch.einsum('pijkl,ai,bj,ck,dl->pabcd', lat.stiffness, Q, Q, Q, Q)
        C_mand = elasticity_func.stiffness_cart_4_to_Mandel(C_rot)
        transformed = Data(
            node_attrs=lat.node_attrs,
            edge_attr=lat.edge_attr,
            edge_index=lat.edge_index,
            positions = torch.einsum('ij,pj->pi', Q, lat.positions),
            shifts = torch.einsum('ij,pj->pi', Q, lat.shifts),
            rel_dens=lat.rel_dens,
            stiffness=C_mand,
            name = lat.name
        )
        return transformed
# %%  
# train_dset = GLAMM_Dataset(
#     root='C:/temp/GLAMMDsetF',
#     catalogue_path='C:/temp/GLAMMDsetF/raw/fixed_dset_7000_train.lat',
#     transform=RotateLat(),
#     dset_fname='train.pt',
#     n_reldens=10,
#     choose_reldens='first',
#     graph_ft_format='cartesian_4',
# )
# print(train_dset)
# delattr(train_dset.data, 'compliance')
# train_dset.data.stiffness = train_dset.data.stiffness.float()
cat = Catalogue.from_file('C:/temp/GLAMMDsetF/raw/fixed_dset_1298_val.lat', 0)
# %%
name = None
main = {}
for data in train_dset:
    if name is None:
        name = data.name
    else:
        if name != data.name:
            break
    
    C = data.stiffness.numpy()
    rho = data.rel_dens.item()
    rows, cols = np.triu_indices(6)
    C21 = C[0,rows, cols]
    d = {}
    for i, Ci in enumerate(C21):
        d[f'C{i}'] = Ci
    main[rho] =  d
# %%
df = pd.DataFrame(main).T
df.index.name = 'rho'
df
# %%
rows, cols = np.triu_indices(6)
data = cat[0]['compliance_tensors']
for key in data:
    S = data[key]
    C = np.linalg.inv(S)
    C4 = elasticity_func.stiffness_Voigt_to_4th_order(C)
    C = elasticity_func.stiffness_cart_4_to_Mandel(torch.tensor(C4))
    data[key] = C[rows, cols].numpy()
df = pd.DataFrame(data).T
df.index.name = 'rho'
df
# %%
df_plot = df.reset_index().melt(id_vars='rho', value_name='C', var_name='component')
# %%
sns.scatterplot(data=df_plot, x='rho', y='C', hue='component')
plt.xscale('log')
plt.yscale('log')
# %%
rho = np.array(df.index)
y = np.array(df)
y = y/y[0,:]
x = rho/rho[0]
x = np.log(x)
y = np.log(y)
n, c = np.polyfit(x=x, y=y, deg=1)
sns.histplot(x=n)
# %%
rows, cols = np.triu_indices(6)
el_tens = CartesianTensor('ijkl=jikl=ijlk=klij')
data = cat[1]['compliance_tensors']
for key in data:
    S = data[key]
    C = np.linalg.inv(S)
    C = 1e3*C
    C4 = elasticity_func.stiffness_Voigt_to_4th_order(C)
    C4 = torch.tensor(C4)
    sph_tens = el_tens.from_cartesian(C4)
    data[key] = sph_tens.numpy()
df = pd.DataFrame(data).T
df.index.name = 'rho'
print(df)
df = df / df.iloc[0]
df
# %%
df_plot = df.reset_index().melt(id_vars='rho', value_name='C', var_name='component')
# %%
sns.scatterplot(data=df_plot, x='rho', y='C', hue='component')
plt.xscale('log')
plt.yscale('log')
# %%
rho = np.array(df.index)
y = np.array(df)
y = y/y[0,:]
x = rho/rho[0]
x = np.log(x)
y = np.log(y)
n, c = np.polyfit(x=x, y=y, deg=1)
sns.histplot(x=n)