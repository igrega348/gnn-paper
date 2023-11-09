# %%
import os
import sys
sys.path.append(os.path.abspath('../../'))
from argparse import Namespace
import json
import time
from typing import Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelSummary, 
    ModelCheckpoint, 
    TQDMProgressBar,
    EarlyStopping
) 
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.seed import seed_everything
from torch_geometric.loader import DataLoader
from e3nn import o3

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from .model_torch import PositiveLiteGNN
from data import elasticity_func
from gnn.mace import get_edge_vectors_and_lengths
from gnn.callbacks import PrintTableMetrics
# %%
class LightningWrappedModel(pl.LightningModule):
    _time_metrics = {}
    
    def __init__(self, model: torch.nn.Module, params: Namespace, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(params, dict):
            params = Namespace(**params)
        self.params = params
        self.model = model(params)
       
        self.save_hyperparameters(params)

    def configure_optimizers(self):
        params = self.params
        optim = torch.optim.AdamW(params=self.model.parameters(), lr=params.lr, 
            betas=(params.beta1,0.999), eps=params.epsilon,
            amsgrad=params.amsgrad, weight_decay=params.weight_decay,)
        return optim

    def training_step(self, batch, batch_idx):
        directions = torch.randn(250, 3, dtype=torch.float32, device=batch.positions.device)
        directions = directions / directions.norm(dim=-1, keepdim=True)
    
        output = self.model(batch)

        true_stiffness = batch['stiffness']
        pred_stiffness = output['stiffness']

        rows, cols = torch.triu_indices(6,6)
       
        target = true_stiffness[:, rows, cols]
        predicted = pred_stiffness[:, rows, cols]
        stiffness_loss = torch.nn.functional.l1_loss(predicted, target, reduction='none').mean(dim=1) # [N]
        
        true_stiff_4 = elasticity_func.stiffness_Mandel_to_cart_4(true_stiffness)
        pred_stiff_4 = elasticity_func.stiffness_Mandel_to_cart_4(pred_stiffness)
        
        stiff_dir_true = torch.einsum('...ijkl,pi,pj,pk,pl->...p', true_stiff_4, directions, directions, directions, directions) # [N, 250]    
        stiff_dir_pred = torch.einsum('...ijkl,pi,pj,pk,pl->...p', pred_stiff_4, directions, directions, directions, directions)
        stiff_dir_loss = torch.nn.functional.l1_loss(stiff_dir_pred, stiff_dir_true, reduction='none').mean(dim=1) # [N]
        mean_stiffness = stiff_dir_true.mean(dim=1) # [N]

        stiff_dir_loss_mean = stiff_dir_loss.mean()
        stiffness_loss_mean = stiffness_loss.mean()

        loss = stiffness_loss
        if self.params.use_dir_loss:
            loss += stiff_dir_loss
        loss = 100*(loss / mean_stiffness).mean() # [1]
    
        self.log('loss', loss, batch_size=batch.num_graphs, logger=True)
        self.log('stiff_dir_loss', stiff_dir_loss_mean, batch_size=batch.num_graphs, logger=True, prog_bar=True)
        self.log('stiffness_loss', stiffness_loss_mean, batch_size=batch.num_graphs, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        directions = torch.randn(250, 3, dtype=torch.float32, device=batch.positions.device)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        
        output = self.model(batch)
        true_stiffness = batch['stiffness']
        pred_stiffness = output['stiffness']

        rows, cols = torch.triu_indices(6,6)
       
        target = true_stiffness[:, rows, cols]
        predicted = pred_stiffness[:, rows, cols]
        stiffness_loss = torch.nn.functional.l1_loss(predicted, target)
        
        
        true_stiff_4 = elasticity_func.stiffness_Mandel_to_cart_4(true_stiffness)
        pred_stiff_4 = elasticity_func.stiffness_Mandel_to_cart_4(pred_stiffness)
        stiff_dir_true = torch.einsum('...ijkl,pi,pj,pk,pl->...p', true_stiff_4, directions, directions, directions, directions)       
        stiff_dir_pred = torch.einsum('...ijkl,pi,pj,pk,pl->...p', pred_stiff_4, directions, directions, directions, directions)
        stiff_dir_loss = torch.nn.functional.l1_loss(stiff_dir_pred, stiff_dir_true)
       
        loss = stiffness_loss
    
        self.log('val_loss', loss, batch_size=batch.num_graphs, logger=True, prog_bar=True, sync_dist=True)
        self.log('val_stiff_dir_loss', stiff_dir_loss, batch_size=batch.num_graphs, logger=True, sync_dist=True)
        self.log('val_stiffness_loss', stiffness_loss, batch_size=batch.num_graphs, logger=True, sync_dist=True)
        return loss
        
    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Tuple:
        """Returns (prediction, true)"""
        return self.model(batch), batch
    
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
        # check if loss is nan
        loss = outputs['loss']
        if torch.isnan(loss):
            self.trainer.should_stop = True
            rank_zero_info('Loss is NaN. Stopping training')


class RotateLat:
    def __init__(self, rotate=True):
        self.rotate = rotate

    def __call__(self, lat: Data, Q: Optional[Tensor] = None):
        if self.rotate:
            if Q is None:
                Q = o3.rand_matrix()
            C = torch.einsum('...ijkl,ai,bj,ck,dl->...abcd', lat.stiffness, Q, Q, Q, Q)
            S = torch.einsum('...ijkl,ai,bj,ck,dl->...abcd', lat.compliance, Q, Q, Q, Q)
            pos = torch.einsum('ij,...j->...i', Q, lat.positions)
            shifts = torch.einsum('ij,...j->...i', Q, lat.shifts)
        else:
            C = lat.stiffness
            S = lat.compliance
            pos = lat.positions
            shifts = lat.shifts
            
        C_mand = elasticity_func.stiffness_cart_4_to_Mandel(C)
        S_mand = elasticity_func.stiffness_cart_4_to_Mandel(S)
        transformed = Data(
            node_attrs=lat.node_attrs,
            edge_attr=lat.edge_attr,
            edge_index=lat.edge_index,
            positions = pos,
            shifts = shifts,
            rel_dens=lat.rel_dens,
            stiffness=C_mand,
            compliance=S_mand,
            name = lat.name
        )
        return transformed


def obtain_errors(results, tag: str):
    target = torch.cat([x[1]['stiffness'] for x in results]) # [num_graphs, 6, 6]
    prediction = torch.cat([x[0]['stiffness'] for x in results]) # [num_graphs, 6, 6]
    names = np.concatenate([x[1]['name'] for x in results])
    rel_dens = torch.cat([x[1]['rel_dens'] for x in results]).numpy()
    directions = torch.randn(250, 3)
    directions = directions / directions.norm(dim=1, keepdim=True)
    loss = torch.nn.functional.l1_loss(prediction, target, reduction='none').mean(dim=(1,2)).numpy()
    target_4 = elasticity_func.stiffness_Mandel_to_cart_4(target)
    prediction_4 = elasticity_func.stiffness_Mandel_to_cart_4(prediction)
    c = torch.einsum('...ijkl,pi,pj,pk,pl->...p', target_4, directions, directions, directions, directions).numpy()
    c_pred = torch.einsum('...ijkl,pi,pj,pk,pl->...p', prediction_4, directions, directions, directions, directions).numpy()
    dir_loss = np.abs(c - c_pred).mean(axis=1)
    mean_stiffness = c.mean(axis=1)
    # eigenvalues
    target_eig = [x for x in torch.linalg.eigvalsh(target).numpy()]
    try:
        predicted_eig = [x for x in torch.linalg.eigvalsh(prediction).numpy()]
    except:
        predicted_eig = np.nan
    return pd.DataFrame({'name':names, 'rel_dens':rel_dens, 'mean_stiffness':mean_stiffness, 'loss':loss, 'dir_loss':dir_loss, 'tag':tag, 'target_eig':target_eig, 'predicted_eig':predicted_eig})

def aggr_errors(df_errors):
    params = {}
    if df_errors['loss'].isna().sum() > 0:
        return params
    df_errors['rel_loss'] = df_errors['loss'] / df_errors['mean_stiffness']
    df_errors['rel_dir_loss'] = df_errors['dir_loss'] / df_errors['mean_stiffness']
    df_errors['min_pred_eig'] = df_errors['predicted_eig'].map(np.min)
    df_errors['min_target_eig'] = df_errors['target_eig'].map(np.min)
    # eigenvalue loss will be calculated as loss between the two volumes calculated from eigenvalues
    predicted_volumes = df_errors['predicted_eig'].map(np.prod)
    target_volumes = df_errors['target_eig'].map(np.prod)
    df_errors['eig_loss'] = np.abs(predicted_volumes - target_volumes)
    df_errors['rel_eig_loss'] = df_errors['eig_loss'] / target_volumes

    means = df_errors[['tag','loss','rel_loss','dir_loss','rel_dir_loss','eig_loss','rel_eig_loss']].groupby(['tag']).mean()
    for tag in means.index:
        for col in means.columns:
            params[f'{col}_{tag}'] = means.loc[tag, col]

    mins = df_errors[['tag','min_pred_eig','min_target_eig']].groupby(['tag']).min()
    for tag in mins.index:
        for col in mins.columns:
            params[f'{col}_{tag}'] = mins.loc[tag, col]
    
    prop_eig_negative = (df_errors['min_pred_eig']<0).groupby(df_errors['tag']).sum() / df_errors['tag'].value_counts()
    for tag in prop_eig_negative.index:
        params[f'prop_eig_negative_{tag}'] = prop_eig_negative.loc[tag]
    
    return params

def load_datasets(tag: str, which: str = '0imp', parent: str = '../../datasets', reldens_norm: bool = False, rotate: bool = True):
    assert which in ['0imp_quarter', '0imp_half', '0imp', '1imp', '2imp', '4imp']
    if tag == 'test':
        root = os.path.join(parent, which)
        dset_file = 'test_cat.lat'
        processed_fname = 'test.pt'
    elif tag == 'train':
        root = os.path.join(parent, which)
        dset_file = 'training_cat.lat'
        processed_fname = 'train.pt'
    elif tag == 'valid':
        root = os.path.join(parent, which)
        dset_file = 'validation_cat.lat'
        processed_fname = 'validation.pt'
    rank_zero_info(f'Loading dataset {tag} from {root}')
    dset = GLAMM_Dataset(
        root=root,
        catalogue_path=os.path.join(root, 'raw', dset_file),
        transform=RotateLat(rotate=rotate),
        dset_fname=processed_fname,
        n_reldens=3,
        choose_reldens='last',
        graph_ft_format='cartesian_4',
    )
    rank_zero_info(dset)

    # scaling and normalization
    if reldens_norm:
        normalization_factor = 100 / dset.data.rel_dens.view(-1,1,1,1,1)
    else:
        normalization_factor = 10000 # increased again because we're targeting relative densities on the order of 0.01

    dset.data.stiffness = (dset.data.stiffness * normalization_factor).float()
    dset.data.compliance = (dset.data.compliance / normalization_factor).float()

    return dset

def main():
    df = pd.read_csv('./mace-hparams-180.csv', index_col=0)
    num_hp_trial = int(os.environ['NUM_HP_TRIAL'])

    desc = "Exp-180. Stiffness training. MACE+ve. 4imp"
    rank_zero_info(desc)
    seed_everything(num_hp_trial, workers=True)

    params = Namespace(
        # network
        lmax=4,
        hidden_irreps='+'.join([f'{df.loc[num_hp_trial, "hidden_irreps"]}x{i}e' if i%2==0 else f'{df.loc[num_hp_trial, "hidden_irreps"]}x{i}o' for i in range(0,5)]),
        readout_irreps='+'.join([f'{df.loc[num_hp_trial, "readout_irreps"]}x{i}e' if i%2==0 else f'{df.loc[num_hp_trial, "readout_irreps"]}x{i}o' for i in range(0,5)]),
        num_edge_bases=int(df.loc[num_hp_trial, 'num_edge_bases']),
        interaction_reduction='sum',
        interaction_bias=True,
        agg_norm_const=2.0,
        inter_MLP_dim=64,
        inter_MLP_layers=3,
        correlation=3,
        global_reduction='mean',
        message_passes=int(df.loc[num_hp_trial, 'message_passes']),
        # dataset
        which='4imp',
        # training
        use_dir_loss=df.loc[num_hp_trial, 'use_dir_loss'],
        num_hp_trial=num_hp_trial,
        batch_size=64,
        valid_batch_size=64,
        log_every_n_steps=25,
        optimizer='adamw',
        lr=df.loc[num_hp_trial, 'lr'], 
        amsgrad=True,
        weight_decay=1e-8,
        beta1=0.9,
        epsilon=1e-8,
        num_workers=4,
    )
    params.desc = desc

    run_name = os.environ['SLURM_JOB_ID']
    log_dir = Path(f'./{run_name}')
    while log_dir.is_dir():
        run_name = str(int(run_name)+1)
        log_dir = Path(f'./{run_name}')
    log_dir.mkdir()
    rank_zero_info(log_dir)
    params.log_dir = str(log_dir)

    ############# setup data ##############
    train_dset = load_datasets(which=params.which, tag='train', reldens_norm=False)
    valid_dset = load_datasets(which='0imp', tag='valid', reldens_norm=False)

    max_edge_radius = train_dset.data.edge_attr.max().item()
    params.max_edge_radius = max_edge_radius
    # randomize the order of the dataset into loader
    train_loader = DataLoader(
        dataset=train_dset, 
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )

    valid_loader = DataLoader(
        dataset=valid_dset,
        batch_size=params.valid_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
    )

    ############# setup model ##############
    lightning_model = LightningWrappedModel(PositiveLiteGNN, params)

    ############# setup trainer ##############
    wandb_logger = WandbLogger(project="glamm-gnn-fresh", entity="ivan-grega", save_dir=params.log_dir, 
                               tags=['exp-180', 'stiffness', 'mace+ve','4imp'])
    callbacks = [
        ModelSummary(max_depth=3),
        ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.3f}', every_n_epochs=1, monitor='val_loss', save_top_k=1),
        PrintTableMetrics(['epoch','step','loss','val_loss'], every_n_steps=100000),
        EarlyStopping(monitor='val_loss', patience=50, verbose=True, mode='min', strict=False) 
    ]
    max_time = '00:01:27:00' if os.environ['SLURM_JOB_PARTITION']=='ampere' else '00:05:45:00'
    trainer = pl.Trainer(
        accelerator='auto',
        accumulate_grad_batches=4, # effective batch size 256
        gradient_clip_val=10.0,
        default_root_dir=params.log_dir,
        logger=wandb_logger,
        enable_progress_bar=False,
        callbacks=callbacks,
        max_steps=50000,
        max_time=max_time,
        val_check_interval=100,
        log_every_n_steps=params.log_every_n_steps,
        check_val_every_n_epoch=None,
    )

    ############# save params ##############
    if trainer.is_global_zero:
        params.use_dir_loss = bool(params.use_dir_loss)
        params_path = log_dir/f'params-{num_hp_trial}.json'
        params_path.write_text(json.dumps(vars(params), indent=2))

    ############# run training ##############
    trainer.fit(lightning_model, train_loader, valid_loader)

    ############# run testing ##############
    rank_zero_info('Testing')
    train_loader = DataLoader(
        dataset=train_dset, batch_size=params.valid_batch_size, 
        shuffle=False,)
    valid_loader = DataLoader(
        dataset=valid_dset, batch_size=params.valid_batch_size,
        shuffle=False,
    )
    test_dset = load_datasets(which='0imp', tag='test', reldens_norm=False)
    test_loader = DataLoader(
        dataset=test_dset, batch_size=params.valid_batch_size, 
        shuffle=False, 
    )
    train_results = trainer.predict(lightning_model, train_loader, return_predictions=True, ckpt_path='best')
    valid_results = trainer.predict(lightning_model, valid_loader, return_predictions=True, ckpt_path='best')
    test_results = trainer.predict(lightning_model, test_loader, return_predictions=True, ckpt_path='best')
    df_errors = pd.concat([obtain_errors(train_results, 'train'), obtain_errors(valid_results, 'valid'), obtain_errors(test_results, 'test')], axis=0, ignore_index=True)
    eval_params = aggr_errors(df_errors)
    pd.Series(eval_params, name=num_hp_trial).to_csv(log_dir/f'aggr_results-{num_hp_trial}-step={trainer.global_step}.csv')
 
    if eval_params['loss_test']>10:
        for f in log_dir.glob('**/epoch*.ckpt'):
            rank_zero_info(f'Test loss: {eval_params["loss_test"]}. Removing checkpoint {f}')
            f.unlink()

if __name__=='__main__':
    main()
