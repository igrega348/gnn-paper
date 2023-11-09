# %%
import os
import sys
sys.path.append(os.path.abspath('../../'))
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
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

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from data import elasticity_func
from model_torch import PositiveLiteGNN
from train_mace import LightningWrappedModel, RotateLat, load_datasets, obtain_errors, aggr_errors
# %%

def main():
    df = pd.read_csv('./mace-hparams-180.csv', index_col=0)

    num_hp_trial = int(os.environ['NUM_HP_TRIAL'])
    params_path = Path(f'./results/params-{num_hp_trial}.json')
    params = json.loads(params_path.read_text())
    params = Namespace(**params)
    log_dir = Path(params.log_dir)
    rank_zero_info(log_dir)
    ckpts = list(log_dir.glob('glamm-gnn-fresh/*/checkpoints/epoch*.ckpt'))
    if len(ckpts) == 0:
        rank_zero_info(f'No checkpoint for {num_hp_trial} found.')
        quit()
    else:
        ckpt_path = ckpts[-1]
    if len(list(log_dir.glob('aggr*')))>0:
        rank_zero_info('Found aggr. Stopping')
        quit()

    ############# setup model ##############
    lightning_model = LightningWrappedModel(PositiveLiteGNN, params)

    ############# setup trainer ##############
    trainer = pl.Trainer(
        accelerator='auto',
        default_root_dir=params.log_dir,
    )

    ############# run testing ##############
    rank_zero_info('Testing')
    test_dset = load_datasets(which='0imp', tag='test', reldens_norm=False, rotate=True)
    test_loader = DataLoader(
        dataset=test_dset, batch_size=64, 
        shuffle=False, 
    )
    test_results = trainer.predict(lightning_model, test_loader, return_predictions=True, ckpt_path=ckpt_path)
    df_errors = obtain_errors(test_results, 'test')
    eval_params = aggr_errors(df_errors)
    pd.Series(eval_params, name=num_hp_trial).to_csv(log_dir/f'aggr_results-{num_hp_trial}-step={trainer.global_step}.csv')

    
if __name__=='__main__':
    main()