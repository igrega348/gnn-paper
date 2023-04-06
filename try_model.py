from argparse import Namespace
import logging
from tqdm import tqdm

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelSummary, 
    ModelCheckpoint, 
    TQDMProgressBar
) 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from e3nn import o3
from e3nn.io import CartesianTensor
import wandb

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from gnn.models import LatticeGNN
from gnn.callbacks import PrintTableMetrics, upload_evaluations

def rotate_lat(lat: Data):
    Q = o3.rand_matrix()
    irreps = o3.Irreps('2x0e+2x2e+1x4e')
    R = irreps.D_from_matrix(Q)
    transformed = Data(
        node_attrs=lat.node_attrs,
        edge_attr=lat.edge_attr,
        edge_index=lat.edge_index,
        positions = torch.einsum('ij,pj->pi', Q, lat.positions),
        shifts = torch.einsum('ij,pj->pi', Q, lat.shifts),
        # unit_shifts = lat.unit_shifts,
        rel_dens=lat.rel_dens,
        stiffness=torch.einsum('pj,ij->pi', lat.stiffness, R),
        name = lat.name
    )
    return transformed

def compute_avg_num_neighbors(
    data_loader: torch.utils.data.DataLoader,
) -> float:
    num_neighbors = []
    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    num_neighbors = torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    avg_num_neighbors = torch.mean(num_neighbors)
    return avg_num_neighbors


# sweep_config = {
#     'method':'bayes',
#     'name':'hp-optim1',
#     'metric':{'goal':'minimize','name':'val_loss'},
#     'parameters':{
#         'lr': {'min':1e-5, 'max':0.01, 'distribution':'log_uniform_values'},
#         'message_passes': {'min':1, 'max':5, 'distribution':'int_uniform'},
#         'interaction_reduction':{'values':['mean','sum']},
#         'global_reduction':{'values':['mean','sum']},
#     }
# }
# sweep_config = dict(
#     interaction_reduction='mean',
#     global_reduction='sum',
#     message_passes=3,
#     lr=0.001,
# )

def main() -> None:
    params = Namespace(
        lmax=3, # spherical harmonics
        hidden_irreps='32x0e+32x1o+32x2e',
        readout_irreps='32x0e+32x1o+32x2e',
        correlation=3,
        interaction_reduction='pna',
        global_reduction='mean',
        message_passes=3,
        optimizer='nadam',
        weight_decay=0,
        scheduler='linearlr',
        max_num_epochs=5,
        lr=0.001,
        interaction_bias=True,
    )

    # wandb_logger = WandbLogger(project="gnn-paper-v0", entity="ivan-grega", config=params)
    # wandb_logger = WandbLogger()
    # params.lr = wandb_logger.experiment.config.lr
    # params.message_passes = wandb_logger.experiment.config.message_passes
    # params.interaction_reduction = wandb_logger.experiment.config.interaction_reduction
    # params.global_reduction = wandb_logger.experiment.config.global_reduction
    # wandb_logger.experiment.log_code('.')

    train_dset = GLAMM_Dataset(
        root='C:/temp/gnn-paper-data/GLAMMDset',
        catalogue_path='C:/temp/gnn-paper-data/train.lat',
        transform=rotate_lat,
        dset_fname='train.pt',
        n_reldens=4,
        graph_ft_format='cartesian_4',
    )
    el_tens = CartesianTensor('ijkl=jikl=ijlk=klij')
    train_dset.data.stiffness = train_dset.data.stiffness / train_dset.data.rel_dens.view(-1,1,1,1,1)
    normalization_factor = 2/torch.max(torch.abs(train_dset.data.stiffness))
    print(f'Normalization factor {normalization_factor}')
    train_dset.data.stiffness = el_tens.from_cartesian(train_dset.data.stiffness*normalization_factor).float()

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=100,
        shuffle=True,
        num_workers=4
    )

    val_dset = GLAMM_Dataset(
        root='C:/temp/gnn-paper-data/GLAMMDset',
        catalogue_path='C:/temp/gnn-paper-data/train.lat',
        transform=rotate_lat,
        dset_fname='valid.pt',
        n_reldens=4,
        choose_reldens='last',
        graph_ft_format='cartesian_4',
    )
    val_dset.data.stiffness = val_dset.data.stiffness / val_dset.data.rel_dens.view(-1,1,1,1,1)
    val_dset.data.stiffness = el_tens.from_cartesian(val_dset.data.stiffness*normalization_factor).float()

    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=100,
        shuffle=False,
        num_workers=4
    )

    deg = torch.zeros(40, dtype=torch.long)
    for data in tqdm(train_loader):
        d = degree(data.edge_index[1], num_nodes=None, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    deg = deg.to(torch.float)
    total_no_vertices = deg.sum()
    bin_degrees = torch.arange(len(deg))
    avg_deg = {
        'lin': ((bin_degrees * deg).sum() / total_no_vertices).item(),
        'log': (((bin_degrees + 1).log() * deg).sum() / total_no_vertices).item(),
        'exp': ((bin_degrees.exp() * deg).sum() / total_no_vertices).item(),
    }

    params.agg_norm_const = avg_deg

    # avg_num_neighbours = compute_avg_num_neighbors(train_loader)
    # print(f'Average number of neighbors: {avg_num_neighbours}')

    model = LatticeGNN(params)
    # wandb_logger.watch(model, log="all", log_freq=50)


    callbacks = [
        ModelSummary(max_depth=3),
        ModelCheckpoint(save_top_k=1, monitor='val_loss'),
        TQDMProgressBar(refresh_rate=1),
        # PrintTableMetrics(['epoch','loss','val_loss','lr','eta'], every_n_steps=1)
    ]
    trainer = pl.Trainer(
        accelerator='auto',
        # max_epochs=3,
        max_steps=2,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        # enable_progress_bar=False,
        # gradient_clip_val=10,
        # gradient_clip_algorithm='norm',
        # logger=wandb_logger,
        log_every_n_steps=50
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # results = trainer.predict(
    #     model=model, 
    #     dataloaders=DataLoader(
    #         dataset=train_dset,
    #         batch_size=100,
    #         shuffle=False,
    #         drop_last=True,
    #         num_workers=4,
    #         ),
    #     return_predictions=True
    # )
    # upload_evaluations(results, 'train')

    # # Load best validation model
    # logging.info("Predict with best validation model")
    # val_results = trainer.predict(model=model, dataloaders=val_loader, return_predictions=True, ckpt_path='best')
    # upload_evaluations(val_results, 'valid')


if __name__=='__main__':
    main()#
    # sweep_config = {
    #     'method':'bayes',
    #     'name':'hp-optim1',
    #     'metric':{'goal':'minimize','name':'val_loss'},
    #     'parameters':{
    #         'lr': {'min':1e-5, 'max':0.01, 'distribution':'log_uniform_values'},
    #         'message_passes': {'min':1, 'max':5, 'distribution':'int_uniform'},
    #         'interaction_reduction':{'values':['mean','sum']},
    #         'global_reduction':{'values':['mean','sum']},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_config, project='gnn-paper-v0', entity='ivan-grega')
    # wandb.agent(sweep_id=sweep_id, function=main, count=2)