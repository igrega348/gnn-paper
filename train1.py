from argparse import Namespace
import logging
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
        rel_dens=lat.rel_dens,
        stiffness=torch.einsum('pj,ij->pi', lat.stiffness, R),
        name = lat.name
    )
    return transformed


def main() -> None:
    df = pd.read_csv('./adamw-hp-dim.csv', index_col=0)
    # num_hp_trial = int(os.environ['NUM_HP_TRIAL'])
    num_hp_trial = 0
    
    params = Namespace(
        lmax=3, # spherical harmonics
        hidden_irreps='64x0e+64x1o+64x2e',
        readout_irreps='32x0e+32x1o+32x2e',
        interaction_reduction='mean',
        agg_norm_const=16,
        correlation=3,
        global_reduction='mean',
        message_passes=2,
        interaction_bias=True,
        batch_size=100,
        max_num_epochs=20,
        optimizer='adamw',
        lr=df.loc[num_hp_trial, 'lr'],
        amsgrad=True,
        weight_decay=1e-4,
        beta1=(1-df.loc[num_hp_trial, 'onebeta1']),
        epsilon=1e-8,
        # nesterov=True,
        # momentum=0.9,
        scheduler=None,
        num_workers=4,
    )

    # wandb_logger = WandbLogger(project="hyperparam_search", entity="ivan-grega")
    # wandb_logger.experiment.config["desc"]  = f"Fixed dataset, tiny."
    # wandb_logger.experiment.log_code('.')

    el_tens = CartesianTensor('ijkl=jikl=ijlk=klij')

    train_dset = GLAMM_Dataset(
        root='./GLAMMDsetT',
        catalogue_path='C:/temp/gnn-paper-data/tiny_dset_7000_train.lat',
        transform=rotate_lat,                             
        dset_fname='train.pt',
        n_reldens=5,
        choose_reldens='half',
        graph_ft_format='cartesian_4',
    )
    print(train_dset)
    delattr(train_dset.data, 'compliance')

    train_dset.data.stiffness = train_dset.data.stiffness / train_dset.data.rel_dens.view(-1,1,1,1,1)
    normalization_factor = 2/torch.max(torch.abs(train_dset.data.stiffness))
    # normalization_factor = 3.147787432038804
    # wandb_logger.experiment.config["normalization_factor"]  = normalization_factor
    # convert to irreps
    train_dset.data.stiffness = el_tens.from_cartesian(train_dset.data.stiffness*normalization_factor).float()

   
    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )

    valid_dset = GLAMM_Dataset(
        root='./GLAMMDsetT',
        catalogue_path='C:/temp/gnn-paper-data/tiny_dset_1298_val.lat',
        transform=rotate_lat,                             
        dset_fname='validation.pt',
        n_reldens=5,
        choose_reldens='half',
        graph_ft_format='cartesian_4',
    )
    delattr(valid_dset.data, 'compliance')

    # convert to irreps
    valid_dset.data.stiffness = valid_dset.data.stiffness / valid_dset.data.rel_dens.view(-1,1,1,1,1)
    valid_dset.data.stiffness = el_tens.from_cartesian(valid_dset.data.stiffness*normalization_factor).float()
    
    val_loader = DataLoader(
        dataset=valid_dset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
    )

    model = LatticeGNN(params)
    # wandb_logger.watch(model, log="all", log_freq=200)


    callbacks = [
        ModelSummary(max_depth=3),
        ModelCheckpoint(save_top_k=1, monitor='val_loss', save_last=True),
        TQDMProgressBar(refresh_rate=5),
        # PrintTableMetrics(['epoch','step','loss','train_err','val_err','lr','eta','samples_per_time'], every_n_steps=400),

    ]
    
    # params.max_num_steps = 100
    trainer = pl.Trainer(
        accelerator='auto',
        # max_steps=params.max_num_steps,
        max_epochs=params.max_num_epochs,
        check_val_every_n_epoch=1,
        val_check_interval=0.2,
        callbacks=callbacks,
        # enable_progress_bar=False,                          
        gradient_clip_val=10,
        gradient_clip_algorithm='norm',
        # logger=wandb_logger,
        log_every_n_steps=150,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # logging.info("Predicting")
    # results = trainer.predict(
        # model=model, 
        # dataloaders=DataLoader(
            # dataset=train_dset,
            # batch_size=100,
            # shuffle=False,
            # drop_last=True,
            # num_workers=4,
            # ),
        # return_predictions=True
    # )
    # upload_evaluations(results, 'train')
    # val_results = trainer.predict(model=model, dataloaders=val_loader, return_predictions=True, ckpt_path='best')
    # upload_evaluations(val_results, 'valid')

if __name__=='__main__':
    main()
  
