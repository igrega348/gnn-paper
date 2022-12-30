from typing import Union, Optional
import os
import os.path as osp
from runpy import run_path
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader, CombinedDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import random_split, ConcatDataset
from torch_geometric.transforms import BaseTransform, Constant
from typing import Optional
from .datasets import GLAMMDataset

class NodeFtFromGraphFt(BaseTransform):
    def __init__(self, set_from: Union[int,list]) -> None:
        self.set_from = set_from

    def __call__(self, data: Data):
        assert data.y.shape[0]==1
        data.x = data.y[0, self.set_from] * torch.ones_like(data.x)
        return data

class GLAMMDataModule(LightningDataModule):
    def __init__(self, run_params: dict):
        super().__init__()

        params = run_params['dataset']
        ddir = osp.realpath(params['data_dir'])
        ddir = osp.join(ddir, params['name'])
        
        self.data_dir = ddir
        num_collate = 1
        for key in ['num_lat','num_imperf','imperf_level','rot_per_lat']:
            if isinstance(params[key], list):
                num_collate = max(num_collate, len(params[key]))
     
        for key in ['num_lat','num_imperf','imperf_level','rot_per_lat']:
            if not isinstance(params[key], list):
                params[key] = [params[key]]*num_collate
            else:
                assert len(params[key])==num_collate
        self.num_collate = num_collate
        run_params['dataset'] = params

        if 'test' in run_params:
            num_collate_test = 1
            params = run_params['test']
            for key in ['num_lat','num_imperf','imperf_level','rot_per_lat']:
                if isinstance(params[key], list):
                    num_collate_test = max(num_collate_test, len(params[key]))
         
            for key in ['num_lat','num_imperf','imperf_level','rot_per_lat']:
                if not isinstance(params[key], list):
                    params[key] = [params[key]]*num_collate_test
                else:
                    assert len(params[key])==num_collate_test
            self.num_collate_test = num_collate_test
            run_params['test'] = params

        self.run_params = run_params

        if 'pre_transform' in params:
            if 'node_ft' in params['pre_transform']:
                node_ft_t = params['pre_transform']['node_ft']
                if len(list(node_ft_t.keys()))>1:
                    raise ValueError('Multiple keys set for nodal pre_transform')
                if 'from_val' in node_ft_t:
                    val = node_ft_t['from_val']
                    self.pre_transform = Constant(val, cat=False)
                elif 'from_graph_ft' in node_ft_t:
                    ind = node_ft_t['from_graph_ft']
                    self.pre_transform = NodeFtFromGraphFt(ind)
        else:
            self.pre_transform = None

    def prepare_data(self): 
        params = self.run_params['dataset']
        for i in range(self.num_collate):
            GLAMMDataset(self.data_dir, 
                pre_transform=self.pre_transform,
                stage='train',
                nlat=params['num_lat'][i], 
                nimperf=params['num_imperf'][i],
                imperf_level=params['imperf_level'][i],
                rot_per_lat=params['rot_per_lat'][i], 
                representation=params['representation'],
                node_pos=params['node_pos'],
                node_ft=params['node_ft_format'],
                edge_ft=params['edge_ft_format'])
        
        if 'test' in self.run_params:
            test_params = self.run_params['test']
            for i in range(self.num_collate_test):
                GLAMMDataset(self.data_dir, 
                    pre_transform=self.pre_transform,
                    stage=test_params['stage'],
                    nlat=test_params['num_lat'][i], 
                    nimperf=test_params['num_imperf'][i],
                    imperf_level=test_params['imperf_level'][i],
                    rot_per_lat=test_params['rot_per_lat'][i], 
                    representation=params['representation'],
                    node_pos=params['node_pos'],
                    node_ft=params['node_ft_format'],
                    edge_ft=params['edge_ft_format'])


    def setup(self, stage: Optional[str] = None) -> None:
        params = self.run_params['dataset']
        train_val_datasets = []
        for i in range(self.num_collate):
            dset = GLAMMDataset(
                self.data_dir, 
                pre_transform=self.pre_transform,
                stage='train',
                nlat=params['num_lat'][i], 
                nimperf=params['num_imperf'][i],
                imperf_level=params['imperf_level'][i],
                rot_per_lat=params['rot_per_lat'][i], 
                representation=params['representation'],
                node_pos=params['node_pos'],
                node_ft=params['node_ft_format'],
                edge_ft=params['edge_ft_format']) 
            if 'target' in params:
                dset.data.y = dset.data.y[:, params['target']]
            train_val_datasets.append(dset)

        if self.num_collate==1:
            train_val_dataset = train_val_datasets[0]
        else:
            # dlist = [d for dset in train_val_datasets for d in dset]
            # train_val_dataset,_ = InMemoryDataset.collate(dlist)
            train_val_dataset = ConcatDataset(train_val_datasets)
        assert train_val_dataset[0].x.shape[1]==params['num_node_ft']
        assert train_val_dataset[0].edge_attr.shape[1]==params['num_edge_ft']

        if 'test' in self.run_params:
            test_datasets = []
            test_params = self.run_params['test']
            for i in range(self.num_collate_test):
                dset = GLAMMDataset(
                    self.data_dir, 
                    pre_transform=self.pre_transform,
                    stage=test_params['stage'],
                    nlat=test_params['num_lat'][i], 
                    nimperf=test_params['num_imperf'][i],
                    imperf_level=test_params['imperf_level'][i],
                    rot_per_lat=test_params['rot_per_lat'][i], 
                    representation=params['representation'],
                    node_pos=params['node_pos'],
                    node_ft=params['node_ft_format'],
                    edge_ft=params['edge_ft_format']) 
                if 'target' in params:
                    dset.data.y = dset.data.y[:, params['target']]        
                test_datasets.append(dset)

            if self.num_collate_test==1:
                test_dataset = test_datasets[0]
            else:
                # dlist = [d for dset in test_datasets for d in dset]
                # test_dataset,_ = InMemoryDataset.collate(dlist)
                test_dataset = ConcatDataset(test_datasets)
            assert test_dataset[0].x.shape[1]==params['num_node_ft']
            assert test_dataset[0].edge_attr.shape[1]==params['num_edge_ft']

        
        if not hasattr(self, 'train_idx'):
            if params['split_mode']=='random':
                split_fn = self.idx_random_split
            elif params['split_mode']=='base_name':
                split_fn = self.idx_base_name_split
            elif params['split_mode']=='imperf':
                split_fn = self.idx_imperf_split
            else:
                raise ValueError(f'Invalid split `{params["split_ratio"]}`')
            # Split fn signature:
            # split_fn(train_val_dataset, params['split_ratio'])

            if isinstance(train_val_dataset, ConcatDataset):
                train_idx = []
                val_idx = []
                for dset in train_val_dataset.datasets:
                    ti, vi = split_fn(
                        dset, params['split_ratio']
                    )
                    train_idx.append(ti)
                    val_idx.append(vi)
                self.train_dataset = ConcatDataset(
                    [dset[ti] for dset, ti in zip(train_val_dataset.datasets, train_idx)]
                )
                self.val_dataset = ConcatDataset(
                    [dset[vi] for dset, vi in zip(train_val_dataset.datasets, val_idx)]
                )
            else:
                train_idx, val_idx = split_fn(
                    train_val_dataset, params['split_ratio']
                )
                self.train_dataset = train_val_dataset[train_idx]
                self.val_dataset = train_val_dataset[val_idx]

            self.train_idx = train_idx
            self.val_idx = val_idx

        # self.train_val_dataset = train_val_dataset
        if 'test' in self.run_params:
            self.test_dataset = test_dataset
        else:
            self.test_dataset = None
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                            batch_size=self.run_params['train']['batch_size'],
                            shuffle=True,
                            num_workers=self.run_params['train']['num_workers'],
                            pin_memory=True
                        )

    def val_dataloader(self):
        if ('use_test' in self.run_params['val']) and (self.run_params['val']['use_test']):
            dset = self.test_dataset
        else:
            dset = self.val_dataset
        return DataLoader(dset, 
                            batch_size=self.run_params['val']['batch_size'],
                            shuffle=False, 
                            num_workers=self.run_params['val']['num_workers'],
                            pin_memory=True
                        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                            batch_size=self.run_params['test']['batch_size'],
                            shuffle=False, 
                            num_workers=self.run_params['test']['num_workers'],
                            pin_memory=True
                        )

    def predict_dataloader(self):
        train_loader = DataLoader(self.train_dataset, 
                            batch_size=self.run_params['val']['batch_size'],
                            shuffle=False,
                            num_workers=self.run_params['val']['num_workers'],
                            pin_memory=True
                        )
        val_loader = DataLoader(self.val_dataset, 
                            batch_size=self.run_params['val']['batch_size'],
                            shuffle=False,
                            num_workers=self.run_params['val']['num_workers'],
                            pin_memory=True
                        )
        return CombinedLoader([train_loader, val_loader])
    
    @staticmethod
    def idx_random_split(train_val_dataset, ratio):
        Ndata = len(train_val_dataset)
        Ntrain = int(ratio*Ndata)
        train_val_idx = np.arange(Ndata)
        np.random.shuffle(train_val_idx)
        train_idx = torch.tensor(np.sort(train_val_idx[:Ntrain]), dtype=torch.long)
        val_idx = torch.tensor(np.sort(train_val_idx[Ntrain:]), dtype=torch.long)
        return train_idx, val_idx

    @staticmethod
    def idx_base_name_split(train_val_dataset, ratio):
        if isinstance(train_val_dataset, ConcatDataset):
            names = np.concatenate([dset.data.name for dset in train_val_dataset.datasets])
        else:
            names = np.array(train_val_dataset.data.name)
        base_names = np.array(['_'.join(name.split('_')[:3]) for name in names])
        uq_base_names = np.unique(base_names)
        Nnames = len(uq_base_names)
        Ntrain = int(ratio*Nnames)

        np.random.shuffle(uq_base_names)
        train_names = uq_base_names[:Ntrain]
        train_mask = np.isin(base_names, train_names)
        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(~train_mask)

        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)
        return train_idx, val_idx

    @staticmethod
    def idx_imperf_split(train_val_dataset, ratio):
        if isinstance(train_val_dataset, ConcatDataset):
            names = np.concatenate([dset.data.name for dset in train_val_dataset.datasets])
        else:
            names = np.array(train_val_dataset.data.name)
        base_names = np.array(['_'.join(name.split('_')[:3]) for name in names])
        uq_base_names = np.unique(base_names)

        train_idx = []
        val_idx = []

        for name in uq_base_names:
            idx_base_names = np.flatnonzero(base_names==name)
            impf_names = np.unique(names[idx_base_names])
            Nnames = len(impf_names)
            Ntrain = int(ratio*Nnames)
            np.random.shuffle(impf_names)
            impf_train = impf_names[:Ntrain]
            impf_val = impf_names[Ntrain:]
            idx_train = np.flatnonzero(np.isin(names, impf_train))
            idx_val = np.flatnonzero(np.isin(names, impf_val))
            train_idx.extend(idx_train)
            val_idx.extend(idx_val)

        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)
        return train_idx, val_idx
        
    def store_dset_names(self, folder: str):
        print('Storing datasets')
        names = {}
        if isinstance(self.train_dataset, ConcatDataset):
                train_names = np.concatenate(
                    [dset.data.name for dset in self.train_dataset.datasets]
                )
                val_names = np.concatenate(
                    [dset.data.name for dset in self.val_dataset.datasets]
                )
        else:
            train_names = np.array(self.train_dataset.data.names)
            val_names = np.array(self.val_dataset.data.names)
        names['train'] = train_names
        names['val'] = val_names
        
        if hasattr(self, 'test_dataset'):
            if isinstance(self.test_dataset, ConcatDataset):
                test_names = np.concatenate(
                    [dset.data.name for dset in self.test_dataset.datasets]
                )
            elif isinstance(self.test_dataset, InMemoryDataset):
                test_names = np.array(self.test_dataset.data.names)
            else:
                test_names = []
            names['test'] = test_names

        os.makedirs(folder, exist_ok=True)
        for key in names:
            arr = names[key]
            if len(arr)>0:
                fn = osp.join(folder, f'{key}_names.npy')
                np.save(fn, names[key])