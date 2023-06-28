# %%
from argparse import Namespace
import logging
import os

from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelSummary, 
    ModelCheckpoint, 
    TQDMProgressBar
) 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from e3nn import o3
from e3nn.io import CartesianTensor
import wandb            
import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
import wandb

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from data import elasticity_func
from gnn.models import PositiveGNN
from gnn.model_torch import PositiveLiteGNN
from gnn.callbacks import PrintTableMetrics, upload_evaluations
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

def get_value(d):
    if isinstance(d, dict):
        if 'value' in d:
            return d['value']
        else:
            return {k: get_value(v) for k, v in d.items()}
    else:
        return d

def get_params(run_name: str) -> None:       
    # run_name = 'zany-forest-89'
    params = Namespace()
    api = wandb.Api()
    runs = api.runs("ivan-grega/glamm-gnn-fresh")
    found = False
    for run in tqdm(runs):
        if run.name==run_name:
            found = True
            break
    assert found
    cfg = run.json_config
    cfg = json.loads(cfg)
    cfg = get_value(cfg)
    params.__dict__ = cfg
    return params

def plot_parity_plots(true, predicted, save_path):
    fig = plt.figure(figsize=(10,10))
    rows, cols = np.triu_indices(6)
    for i in range(21):
        row = rows[i]
        col = cols[i]
        i_subplot = 6*row + col + 1
        ax = fig.add_subplot(6,6,i_subplot)
        x = true[:,i]
        y = predicted[:,i]
        error = np.mean(np.abs(x-y)/np.abs(x).max())
        sns.histplot(x=x, y=y, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(f'{error*100:.2g}%', xy=(0.5,0.9), xycoords='axes fraction', ha='center', va='top')
    # fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    # plt.close(fig)
    plt.show()

class Lite(LightningLite):
    stats = {'step':[], 'loss':[]}

    def run(self, params: Namespace):
        model = PositiveLiteGNN(params)      
        model = self.setup(model)
        model.load_state_dict(self.load('./explore/zany-forest-89/ckpt.pt'))

        train_dset = GLAMM_Dataset(
            root='./GLAMMDsetT',
            catalogue_path='./tiny_dset_7000_train.lat',
            transform=RotateLat(),
            dset_fname='train.pt',
            n_reldens=5,
            choose_reldens='half',
            graph_ft_format='cartesian_4',
        )
        print(train_dset)
        delattr(train_dset.data, 'compliance')

        train_dset.data.stiffness = train_dset.data.stiffness / train_dset.data.rel_dens.view(-1,1,1,1,1)
        normalization_factor = 2/torch.max(torch.abs(train_dset.data.stiffness))

        train_dset.data.stiffness = (train_dset.data.stiffness*normalization_factor).float()

        train_loader = DataLoader(
            dataset=train_dset,
            batch_size=params.batch_size,
            shuffle=False,
            # num_workers=params.num_workers,
        )
        dataloader = self.setup_dataloaders(train_loader)

        model.eval()

        true = []
        pred = []

        step = 0
        stop = False
        with torch.no_grad():
            pbar = tqdm(dataloader)
            for batch in pbar:
                output = model(batch)
                true.append(batch.stiffness)
                pred.append(output['stiffness'])
                
                step += 1

                # stopping conditions
                if step >= params.max_num_steps:
                    stop = True
                    break
        
        true = torch.cat(true, dim=0)
        pred = torch.cat(pred, dim=0)

        return true, pred
# %%
params = get_params('zany-forest-89')
params.max_num_steps = 200
# params.max_num_steps = float('inf')
print(params)
lite = Lite(accelerator='auto', precision=32)
true, pred = lite.run(params)
# %%
rows, cols = np.triu_indices(6)
trues = true[:,rows,cols].cpu().numpy()
preds = pred[:,rows,cols].cpu().numpy()
plot_parity_plots(trues, preds, None)
# %%
