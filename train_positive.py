# %%
from argparse import Namespace
import logging
import os

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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from e3nn import o3
from e3nn.io import CartesianTensor
import wandb            

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from data import elasticity_func
from gnn.models import PositiveGNN
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


def main() -> None:
    df = pd.read_csv('./adamw-hp-dim.csv', index_col=0)
    # num_hp_trial = int(os.environ['NUM_HP_TRIAL'])
    num_hp_trial = 0
        
    params = Namespace(
        lmax=3, # spherical harmonics
        hidden_irreps='32x0e+32x1o+32x2e',
        readout_irreps='16x0e+16x2e',
        interaction_reduction='sum',
        agg_norm_const=4.0,
        correlation=3,
        global_reduction='mean',
        message_passes=2,
        interaction_bias=True,
        batch_size=32,
        max_num_epochs=5,
        optimizer='adamw',
        lr=1e-4,
        amsgrad=True,
        weight_decay=1e-8,
        beta1=0.9,
        epsilon=1e-8,
        scheduler=None,
        func='exp',
        num_workers=4
    )

    # wandb_logger = WandbLogger(project="hyperparam_search", entity="ivan-grega")
    # wandb_logger.experiment.config["desc"]  = f"Overfit, Spectral"

    # wandb_logger.experiment.log_code('.', include_fn=lambda x: os.path.basename(x) in [os.path.basename(__file__),'models.py','blocks.py'])

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
    # wandb_logger.experiment.config["normalization_factor"]  = normalization_factor

    train_dset.data.stiffness = (train_dset.data.stiffness*normalization_factor).float()

   
    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )

    valid_dset = GLAMM_Dataset(
        root='./GLAMMDsetT',
        catalogue_path='./tiny_dset_1298_val.lat',
        transform=RotateLat(),
        dset_fname='validation.pt',
        n_reldens=5,
        choose_reldens='half',
        graph_ft_format='cartesian_4',
    )
    delattr(valid_dset.data, 'compliance')

    # convert to irreps
    valid_dset.data.stiffness = valid_dset.data.stiffness / valid_dset.data.rel_dens.view(-1,1,1,1,1)
    valid_dset.data.stiffness = (valid_dset.data.stiffness*normalization_factor).float()
    
    val_loader = DataLoader(
        dataset=valid_dset,
        batch_size=512,
        shuffle=False,
        num_workers=params.num_workers,
    )
      
    model = PositiveGNN(params)
    # wandb_logger.watch(model, log="all", log_freq=200)


    callbacks = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            # save_top_k=1, monitor='val_loss', save_last=True, 
            every_n_train_steps=1, save_weights_only=True,
            # save_top_k=2, monitor="global_step", mode="max",
            save_last=True
            ),
        TQDMProgressBar(),
        # PrintTableMetrics(['epoch','step','loss','train_err','val_err','lr','eta','samples_per_time'], every_n_steps=10),

    ]
    
    # params.max_num_steps = 100
    trainer = pl.Trainer(
        accelerator='auto',
        # max_steps=params.max_num_steps,
        max_epochs=params.max_num_epochs,
        check_val_every_n_epoch=100,
        val_check_interval=0.2,
        callbacks=callbacks,
        # enable_progress_bar=False,                          
        gradient_clip_val=1.0, #######
        gradient_clip_algorithm='norm',
        # logger=wandb_logger,
        log_every_n_steps=1,
        accumulate_grad_batches=10, ########
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__=='__main__':
    main()
  
