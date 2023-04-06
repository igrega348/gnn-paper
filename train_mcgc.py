from argparse import Namespace
import logging
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset
from torch_geometric.utils import degree
import pytorch_lightning as pl
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

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from gnn.models import MCrystGraphConv as LatticeGNN
from gnn.callbacks import PrintTableMetrics, local_plot_evaluations

# class RotateLat:
#     def __call__(self, lat: Data):
#         Q = o3.rand_matrix()
#         irreps = o3.Irreps('2x0e+2x2e+1x4e')
#         R = irreps.D_from_matrix(Q)
#         transformed = Data(
#             node_attrs=lat.node_attrs,
#             edge_attr=lat.edge_attr,
#             edge_index=lat.edge_index,
#             positions = torch.einsum('ij,pj->pi', Q, lat.positions),
#             shifts = torch.einsum('ij,pj->pi', Q, lat.shifts),
#             rel_dens=lat.rel_dens,
#             stiffness=torch.einsum('pj,ij->pi', lat.stiffness, R),
#             name = lat.name
#         )
#         return transformed

class RotateLat:
    indices = [
        (0,0,0,0),(0,0,1,1),(0,0,2,2),(0,0,0,1),(0,0,0,2),(0,0,1,2),
        (1,1,1,1),(1,1,2,2),(1,1,0,1),(1,1,0,2),(1,1,1,2),
        (2,2,2,2),(2,2,0,1),(2,2,0,2),(2,2,1,2),
        (0,1,0,1),(0,1,0,2),(0,1,1,2),
        (0,2,0,2),(0,2,1,2),
        (1,2,1,2)
    ]

    def __call__(self, lat: Data):
        Q = o3.rand_matrix()
        C_rot = torch.einsum('pijkl,ai,bj,ck,dl->pabcd', lat.stiffness, Q, Q, Q, Q)
        C = torch.cat([C_rot[:,i,j,k,l].view(-1,1) for i,j,k,l in self.indices], dim=1)
        transformed = Data(
            node_attrs=lat.node_attrs,
            edge_attr=lat.edge_attr,
            edge_index=lat.edge_index,
            line_index=lat.line_index,
            positions = torch.einsum('ij,pj->pi', Q, lat.positions),
            shifts = torch.einsum('ij,pj->pi', Q, lat.shifts),
            rel_dens=lat.rel_dens,
            stiffness=C,
            name = lat.name
        )
        return transformed

class LineGraph:
    def __call__(self, lat: Data) -> Data:
        edge_index = lat.edge_index
        index, _ = torch.sort(edge_index, dim=0)
        single_connected = torch.unique(index, dim=1)
        uq_node_nums = torch.unique(single_connected)
        full_adj = []
        for node in uq_node_nums:
            connected_inds = torch.nonzero(torch.any(single_connected==node, dim=0))
            num_connected = connected_inds.shape[0]

            sender = connected_inds.flatten().repeat(num_connected)
            receiver = connected_inds.repeat(1,num_connected).view(-1)
            _loc_index = torch.stack([sender, receiver], dim=0)

            mask_self_loops = _loc_index[0]==_loc_index[1]
            _loc_index = _loc_index[:, ~mask_self_loops]
            full_adj.append(_loc_index)
        full_adj = torch.cat(full_adj, dim=1)
        full_adj = torch.unique(full_adj, dim=1)
        
        data = Data(
                node_attrs=lat.node_attrs,
                positions=lat.positions,
                edge_attr=lat.edge_attr,
                edge_index=lat.edge_index,
                line_index=full_adj,
                unit_shifts=lat.unit_shifts,
                shifts=lat.shifts,
                rel_dens=lat.rel_dens,
                stiffness=lat.stiffness,
                compliance=lat.compliance,
                name = lat.name,
            )
        return data

def main() -> None:
    df = pd.read_csv('./adamw-hp-dim.csv', index_col=0)
    # num_hp_trial = int(os.environ['NUM_HP_TRIAL'])
    num_hp_trial = 0
    
    params = Namespace(
        hidden_dim=64,
        interaction_reduction='sum',
        global_reduction='min',
        message_passes=5,
        batch_size=100,
        max_num_epochs=200,
        optimizer='radam',
        lr_milestones=[150,180],
        lr=1e-3,
        amsgrad=True,
        weight_decay=1e-8,
        beta1=0.9,
        epsilon=1e-8,
        scheduler='multisteplr',
        num_workers=1,
    )

    # wandb_logger = WandbLogger(project="hyperparam_search", entity="ivan-grega")
    # wandb_logger.experiment.config["desc"]  = f"Overfit, CGCNN"

    # wandb_logger.experiment.log_code('.', include_fn=lambda x: os.path.basename(x) in [os.path.basename(__file__),'models.py','blocks.py'])

    el_tens = CartesianTensor('ijkl=jikl=ijlk=klij')

    train_dset = GLAMM_Dataset(
        root='./GLAMMDsetO_m',
        catalogue_path='./overfit_dset_60_train.lat',
        transform=RotateLat(),
        pre_transform=LineGraph(),
        dset_fname='train.pt',
        n_reldens=5,
        choose_reldens='half',
        graph_ft_format='cartesian_4',
    )
    print(train_dset)
    delattr(train_dset.data, 'compliance')

    train_dset.data.stiffness = train_dset.data.stiffness / train_dset.data.rel_dens.view(-1,1,1,1,1)
    normalization_factor = 2/torch.max(torch.abs(train_dset.data.stiffness))
    # wandb_logger.experiment.config["normalization_factor"]  = normalization_factor

    # train_dset.data.stiffness = el_tens.from_cartesian(train_dset.data.stiffness*normalization_factor).float()
    train_dset.data.stiffness = (train_dset.data.stiffness*normalization_factor).float()

   
    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=params.batch_size,
        shuffle=True,
        # num_workers=params.num_workers,
    )

    valid_dset = GLAMM_Dataset(
        root='./GLAMMDsetO_m',
        catalogue_path='./overfit_dset_40_val.lat',
        transform=RotateLat(),
        pre_transform=LineGraph(),
        dset_fname='validation.pt',
        n_reldens=5,
        choose_reldens='half',
        graph_ft_format='cartesian_4',
    )
    delattr(valid_dset.data, 'compliance')

    # convert to irreps
    valid_dset.data.stiffness = valid_dset.data.stiffness / valid_dset.data.rel_dens.view(-1,1,1,1,1)
    # valid_dset.data.stiffness = el_tens.from_cartesian(valid_dset.data.stiffness*normalization_factor).float()
    valid_dset.data.stiffness = (valid_dset.data.stiffness*normalization_factor).float()
    
    val_loader = DataLoader(
        dataset=valid_dset,
        batch_size=100,
        shuffle=False,
        # num_workers=params.num_workers,
    )
      
    model = LatticeGNN(params)
    # wandb_logger.watch(model, log="all", log_freq=200)


    callbacks = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(save_top_k=1, monitor='val_loss', save_last=True),
        TQDMProgressBar(),
        # PrintTableMetrics(['epoch','step','loss','train_err','val_err','lr','eta','samples_per_time'], every_n_steps=10),

    ]
    
    # params.max_num_steps = 100
    trainer = pl.Trainer(
        accelerator='auto',
        # max_steps=params.max_num_steps,
        max_epochs=params.max_num_epochs,
        check_val_every_n_epoch=1,
        # val_check_interval=0.2,
        callbacks=callbacks,
        # enable_progress_bar=False,                          
        gradient_clip_val=10,
        gradient_clip_algorithm='norm',
        # logger=wandb_logger,
        log_every_n_steps=3,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logging.info("Predicting")
    results = trainer.predict(
        model=model, 
        dataloaders=DataLoader(
            dataset=train_dset,
            batch_size=100,
            shuffle=False,
            drop_last=False,
            # num_workers=4,
            ),
        return_predictions=True
    )
    local_plot_evaluations(results, 'train_mcgnn_cartesian.png')
    val_results = trainer.predict(model=model, dataloaders=val_loader, return_predictions=True, ckpt_path='best')
    local_plot_evaluations(val_results, 'valid_mcgnn_cartesian.png')

if __name__=='__main__':
    main()
  
