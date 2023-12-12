# %%
import os
import sys
sys.path.append(os.path.abspath('../'))
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

def get_model():
    num_hp_trial = 60
    params_path = Path(__file__).parent.parent.parent / Path(f'gnn-fresh-exp/exp-180/results/params-{num_hp_trial}.json')
    params = json.loads(params_path.read_text())
    params = Namespace(**params)
    log_dir =Path(__file__).parent.parent.parent / Path(f'gnn-fresh-exp/exp-180') / Path(params.log_dir)
    ckpts = list(log_dir.glob('**/epoch*.ckpt'))
    ckpt_path = ckpts[-1]
   
    ############# setup model ##############
    lightning_model = LightningWrappedModel.load_from_checkpoint(ckpt_path, model=PositiveLiteGNN, params=params)
    trainer = pl.Trainer(
        accelerator='auto',
    )

    ############# run testing ##############
    rank_zero_info('Testing')
    test_dset = load_datasets(which='0imp', tag='test', parent='../../ICLR2024/datasets', reldens_norm=False, rotate=True)
    test_loader = DataLoader(
        dataset=test_dset[:5000], batch_size=64, 
        shuffle=False, 
    )
    test_results = trainer.predict(lightning_model, test_loader, return_predictions=True, ckpt_path=ckpt_path)

    return lightning_model
    
if __name__ == '__main__':
    get_model()