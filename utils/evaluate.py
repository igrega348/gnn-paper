import os.path as osp
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import InMemoryDataset
from . import plotting

def plot_metrics(
    run_dir: str, 
    result: list, 
    datamodule: LightningDataModule, 
    stage: str = 'predict'
    ):
    assert isinstance(result, list)
    if isinstance(result[0], list):
        # multiple dataloaders
        true = [tup[1] for dl in result for tup in dl]
        pred = [tup[0] for dl in result for tup in dl]
    else: 
        true = [tup[1] for tup in result]
        pred = [tup[0] for tup in result]
    true = torch.cat(true).numpy()
    pred = torch.cat(pred).numpy()

    if not hasattr(datamodule, 'train_val_dataset'):
        datamodule.setup()

    if stage=='predict':
        dataset,_ = InMemoryDataset.collate(datamodule.train_val_dataset) 
    elif stage=='train':
        dataset = datamodule.train_dataset
    elif stage=='val':
        dataset = datamodule.val_dataset
    elif stage=='test':
        dataset = datamodule.test_dataset
    else:
        raise ValueError(f'Stage `{stage}` not supported')

    plotting.parity_plot(true, pred, osp.join(run_dir, 'callbacks', 'plots', f'end_{stage}_parity.png'))
    plotting.surface_plot(true, pred, dataset, 
                nplot=3, fn=osp.join(run_dir, 'callbacks', 'plots', f'end_{stage}_surface.png'))
    
    del dataset