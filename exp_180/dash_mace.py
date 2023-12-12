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
from .model_torch import PositiveLiteGNN
from .train_mace import LightningWrappedModel, RotateLat, load_datasets, obtain_errors, aggr_errors
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

    return lightning_model
    