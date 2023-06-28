# %%
import os
import sys
sys.path.append(os.path.abspath('../../'))
from argparse import Namespace
import logging
import traceback
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any, Tuple
import shutil
import json
import time

from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelSummary, 
    ModelCheckpoint, 
    TQDMProgressBar
) 
from pytorch_lightning.lite import LightningLite
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from e3nn import o3
from e3nn.io import CartesianTensor
import seaborn as sns
sns.set_context('notebook')
import matplotlib.pyplot as plt
import wandb
# from torch.utils.tensorboard import SummaryWriter

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from data import elasticity_func
from gnn.model_torch import PositiveLiteGNN
from gnn.callbacks import SimpleTableMetrics, PrintTableMetrics
# %%
class LightningWrappedModel(pl.LightningModule):
    _time_metrics = {}
    
    def __init__(self, params: Namespace, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.params = params
        self.automatic_optimization = False
        self.model = PositiveLiteGNN(params)
        self.save_hyperparameters(params)

    def configure_optimizers(self):
        params = self.params
        optim = torch.optim.AdamW(params=self.model.parameters(), lr=params.lr, 
            betas=(params.beta1,0.999), eps=params.epsilon,
            amsgrad=params.amsgrad, weight_decay=params.weight_decay,)
        return optim

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        grad_acc_steps = self.params.grad_acc_steps
        # calculate loss
        output = self.model(batch)
        rows, cols = torch.triu_indices(6,6)
        true_4 = elasticity_func.stiffness_Mandel_to_cart_4(batch['stiffness'])
        pred_4 = elasticity_func.stiffness_Mandel_to_cart_4(output['stiffness'])
        directions = torch.randn(100, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        stiff_true = torch.einsum('...ijkl,pi,pj,pk,pl->...p', true_4, directions, directions, directions, directions)
        stiff_pred = torch.einsum('...ijkl,pi,pj,pk,pl->...p', pred_4, directions, directions, directions, directions)
        stiff_loss = 0.3*torch.nn.functional.mse_loss(100*stiff_pred, 100*stiff_true) / grad_acc_steps
        true = batch['stiffness'][:, rows, cols]
        pred = output['stiffness'][:, rows, cols]
        matrix_loss = 0.7*torch.nn.functional.mse_loss(100*pred, 100*true) / grad_acc_steps
        loss = stiff_loss + matrix_loss

        self.manual_backward(loss)

        if (batch_idx + 1) % grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
    
        self.log('loss', loss, batch_size=batch.num_graphs, prog_bar=True, logger=True)
        self.log('stiff_loss', stiff_loss, batch_size=batch.num_graphs, prog_bar=True, logger=True)
        self.log('matrix_loss', matrix_loss, batch_size=batch.num_graphs, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        rows, cols = torch.triu_indices(6,6)
        true_4 = elasticity_func.stiffness_Mandel_to_cart_4(batch['stiffness'])
        pred_4 = elasticity_func.stiffness_Mandel_to_cart_4(output['stiffness'])
        directions = torch.randn(100, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        stiff_true = torch.einsum('...ijkl,pi,pj,pk,pl->...p', true_4, directions, directions, directions, directions)
        stiff_pred = torch.einsum('...ijkl,pi,pj,pk,pl->...p', pred_4, directions, directions, directions, directions)
        stiff_loss = 0.3*torch.nn.functional.mse_loss(100*stiff_pred, 100*stiff_true)
        true = batch['stiffness'][:, rows, cols]
        pred = output['stiffness'][:, rows, cols]
        matrix_loss = 0.7*torch.nn.functional.mse_loss(100*pred, 100*true)
        loss = stiff_loss + matrix_loss
        self.log('val_loss', loss, prog_bar=True, batch_size=batch.num_graphs, logger=True)
        self.log('val_stiff_loss', stiff_loss, prog_bar=False, batch_size=batch.num_graphs, logger=True)
        self.log('val_matrix_loss', matrix_loss, prog_bar=False, batch_size=batch.num_graphs, logger=True)
        
    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Tuple:
        """Returns (prediction, true)"""
        return self.model(batch), batch
    
    def on_epoch_end(self) -> None:
        if (self.trainer.current_epoch + 1) % self.params.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(self.logger.experiment.dir, f'epoch-{self.trainer.current_epoch}.ckpt')
            self.trainer.save_checkpoint(checkpoint_path, weights_only=True)

    def on_train_epoch_start(self) -> None:
        self._time_metrics['_last_step'] = self.trainer.global_step
        self._time_metrics['_last_time'] = time.time()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        step = self.trainer.global_step
        steps_done = step - self._time_metrics['_last_step']
        time_now = time.time()
        time_taken = time_now - self._time_metrics['_last_time']
        steps_per_sec = steps_done / time_taken
        self._time_metrics['_last_step'] = step
        self._time_metrics['_last_time'] = time_now
        self.log('steps_per_time', steps_per_sec, prog_bar=False, logger=True)

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


def main():
    df = pd.read_csv('../adamw-hp-dim.csv', index_col=0)
    num_hp_trial = 0# int(os.environ['NUM_HP_TRIAL'])

    desc = "Trying out LightningModule"

    params = Namespace(
        lmax=3, # spherical harmonics
        hidden_irreps='32x0e+32x1o+32x2e',
        readout_irreps='16x0e+16x2e',
        interaction_reduction='sum',
        agg_norm_const=4.0,
        correlation=3,
        global_reduction='mean',
        message_passes=3,
        interaction_bias=True,
        batch_size=512,
        max_num_epochs=100,
        max_num_steps=float('inf'),
        max_steps_per_epoch=200,
        max_valid_steps=2,
        save_every_n_epochs=5,
        check_valid_every_n_epochs=10,
        optimizer='adamw',
        lr=df.loc[num_hp_trial, 'lr'],
        amsgrad=True,
        weight_decay=1e-8,
        beta1=(1-df.loc[num_hp_trial, 'onebeta1']),
        epsilon=1e-8,
        matrix_func='square',
        num_workers=4,
        grad_acc_steps=1
    )
    params.desc = desc

    dt = datetime.now()
    run_name = dt.strftime("%Y-%m-%d_%H%M%S")
    log_dir = f'./{run_name}'
    os.makedirs(log_dir, exist_ok=True)
    print(log_dir)
    params.log_dir = log_dir

    ############# setup data ##############
    train_dset = GLAMM_Dataset(
        root='../../GLAMMDsetT',
        catalogue_path='../../GLAMMDsetT/raw/tiny_dset_7000_train.lat',
        transform=RotateLat(),
        dset_fname='train.pt',
        n_reldens=10,
        choose_reldens='first',
        graph_ft_format='cartesian_4',
    )
    print(train_dset)
    delattr(train_dset.data, 'compliance')

    train_dset.data.stiffness = train_dset.data.stiffness / train_dset.data.rel_dens.view(-1,1,1,1,1)
    normalization_factor = 2/torch.max(torch.abs(train_dset.data.stiffness))
    params.normalization_factor = float(normalization_factor)

    train_dset.data.stiffness = (train_dset.data.stiffness*normalization_factor).float()

    valid_dset = GLAMM_Dataset(
        root='../../GLAMMDsetT',
        catalogue_path='../../GLAMMDsetT/raw/tiny_dset_1298_val.lat',
        transform=RotateLat(),
        dset_fname='validation.pt',
        n_reldens=3,
        choose_reldens='last',
        graph_ft_format='cartesian_4',
    )
    print(valid_dset)
    delattr(valid_dset.data, 'compliance')

    valid_dset.data.stiffness = valid_dset.data.stiffness / valid_dset.data.rel_dens.view(-1,1,1,1,1)
    valid_dset.data.stiffness = (valid_dset.data.stiffness*normalization_factor).float()

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )

    ############# setup model ##############
    lightning_model = LightningWrappedModel(params)

    ############# setup trainer ##############
    wandb_logger = WandbLogger(project="memory-leak", entity="ivan-grega", save_dir=params.log_dir)
    callbacks = [
        ModelSummary(max_depth=3),
    ]
    trainer = pl.Trainer(
        accelerator='auto',
        default_root_dir=params.log_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=params.max_num_epochs,
        max_steps=params.max_num_steps,
        max_time="00:23:30:00",
        limit_train_batches=params.max_steps_per_epoch,
        limit_val_batches=params.max_valid_steps,
        check_val_every_n_epoch=params.check_valid_every_n_epochs,
        log_every_n_steps=10,
    )

    ############# run training ##############
    trainer.fit(lightning_model, train_loader, valid_loader)

    ############# save checkpoint ##############
    checkpoint_path = os.path.join(trainer.log_dir, 'final.ckpt')
    trainer.save_checkpoint(checkpoint_path, weights_only=True)

    ############# save params ##############
    params_path = os.path.join(trainer.log_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(vars(params), f)

if __name__=='__main__':
    main()