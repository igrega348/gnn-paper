# %%
import os
import sys
# sys.path.append(os.path.abspath('../../'))
from argparse import Namespace
import logging
import traceback
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any
import shutil
import json

from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.lite import LightningLite
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from e3nn import o3
from e3nn.io import CartesianTensor
import wandb            
import seaborn as sns
sns.set_context('notebook')
import matplotlib.pyplot as plt
import wandb
# from torch.utils.tensorboard import SummaryWriter

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from data import elasticity_func
from gnn.model_torch import PositiveLiteGNN
from gnn.callbacks import SimpleTableMetrics
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
    
def plot_parity_plots(true, predicted, mode):
    fig = plt.figure(figsize=(6,6))
    rows, cols = np.triu_indices(6)
    for i in range(21):
        row = rows[i]
        col = cols[i]
        i_subplot = 6*row + col + 1
        ax = fig.add_subplot(6,6,i_subplot)
        x = true[:,i]
        y = predicted[:,i]
        error = np.mean(np.abs(x-y)/np.abs(x).max())
        if mode == 'hist':
            sns.histplot(x=x, y=y, ax=ax)
        elif mode == 'scatter':
            sns.scatterplot(x=x, y=y, ax=ax, s=5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(f'{error*100:.2g}%', xy=(0.5,0.9), xycoords='axes fraction', ha='center', va='top')
        ax.axline((0,0), slope=1, color='black', linestyle='--')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(x.min(), x.max())
    return fig

def plot_parity_one(true: np.ndarray, predicted: np.ndarray):
    fig = plt.figure(figsize=(6,6))
    sns.scatterplot(x=true, y=predicted, s=1)
    sns.histplot(x=true, y=predicted, stat='density', pthresh=.2, cmap='hsv', cbar=False, alpha=0.5)
    plt.axline((0,0), slope=1, color='k', linewidth=1, linestyle='--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.xlim(true.min(), true.max())
    plt.ylim(true.min(), true.max())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig

class Lite(LightningLite):
    stats = dict()
    step: int
    params: Namespace
    plotting_mode = 'save'

    # writer = SummaryWriter()

    def update_stats(self, log_metrics: dict) -> None:
        step = self.step
        if step in self.stats:
            self.stats[step].update(log_metrics)
        else:
            self.stats[step] = log_metrics
        # for k, v in log_metrics.items():
            # self.writer.add_scalar(k, v, step)
        wandb.log(log_metrics, step=step)

    def plot_predictions(self, true, pred, mode: str, name: Optional[str] = None):
        if len(true)<1 or len(pred)<1: return None
        true = torch.cat(true, dim=0).numpy()
        pred = torch.cat(pred, dim=0).numpy()
        fig = plot_parity_plots(true, pred, mode)
        plt.tight_layout()
        plot_name = f'plot_{self.step}'
        if isinstance(name, str) and ('save' in self.plotting_mode):
            plot_name = f'{name}_{self.step}'
            save_fn = os.path.join(self.params.log_dir, plot_name+'.png')
            fig.savefig(save_fn, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False)
        if 'show' in self.plotting_mode:
            plt.show()
        else:
            plt.close(fig)

    def plot_parity_one(self, true, pred, name: str = None):
        if len(true)<1 or len(pred)<1: return None
        true = torch.cat(true, dim=0).numpy()
        pred = torch.cat(pred, dim=0).numpy()
        fig = plot_parity_one(true, pred)
        plot_name = f'plot_{self.step}'
        if isinstance(name, str) and ('save' in self.plotting_mode):
            plot_name = f'{name}_{self.step}_dir'
            save_fn = os.path.join(self.params.log_dir, plot_name+'.png')
            fig.savefig(save_fn, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False)
        if 'show' in self.plotting_mode:
            plt.show()
        else:
            plt.close(fig)

    def eval(self, model, dataloader, mode: str = 'hist', name: str = None, max_steps: int = 100):
        model.eval()
        trues = []
        preds = []
        stiff_pred = []
        stiff_true = []
        rows, cols = np.triu_indices(6)
        with torch.no_grad():
            step = 0
            total = min(len(dataloader), max_steps)
            for batch in tqdm(dataloader, total=total, desc='Evaluating'):
                output = model(batch)
                # rel_dens = batch['rel_dens'].cpu()
                # scaler = rel_dens.view(-1,1)
                scaler = 1
                directions = torch.randn(100,3, device=batch['stiffness'].device)
                directions = directions / directions.norm(dim=-1, keepdim=True)
                C4_pred = elasticity_func.stiffness_Mandel_to_cart_4(output['stiffness'])
                C4_true = elasticity_func.stiffness_Mandel_to_cart_4(batch['stiffness'])
                stiff_pred.append(torch.einsum('...abcd,pa,pb,pc,pd->...p', C4_pred, directions, directions, directions, directions).flatten().cpu())
                stiff_true.append(torch.einsum('...abcd,pa,pb,pc,pd->...p', C4_true, directions, directions, directions, directions).flatten().cpu())


                true = batch['stiffness'][:, rows, cols]
                pred = output['stiffness'][:, rows, cols]
                trues.append(true.cpu()*scaler)
                preds.append(pred.cpu()*scaler)
                step += 1
                if step >= max_steps:
                    break
        self.plot_predictions(trues, preds, mode, name)
        self.plot_parity_one(stiff_true, stiff_pred, name+'_dir')

        return true, pred

    def run(self, params: Namespace):
        self.params = params
        model = PositiveLiteGNN(params)
        self.print('Setting optimizer AdamW')
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=params.lr, 
            betas=(params.beta1,0.999), eps=params.epsilon,
            amsgrad=params.amsgrad, weight_decay=params.weight_decay,
        )
        model, optimizer = self.setup(model, optimizer)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.3)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.print(f'Number of trainable parameters: {pytorch_total_params}')

        train_dset = GLAMM_Dataset(
            root='../../GLAMMDsetT',
            catalogue_path='../../GLAMMDsetT/raw/tiny_dset_7000_train.lat',
            transform=RotateLat(),
            dset_fname='train.pt',
            n_reldens=10,
            choose_reldens='first',
            graph_ft_format='cartesian_4',
        )
        self.print(train_dset)
        delattr(train_dset.data, 'compliance')

        train_dset.data.stiffness = train_dset.data.stiffness / train_dset.data.rel_dens.view(-1,1,1,1,1)
        normalization_factor = 2/torch.max(torch.abs(train_dset.data.stiffness))

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
        self.print(valid_dset)
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
        train_loader, valid_loader = self.setup_dataloaders(train_loader, valid_loader)

        self.step = 0
        stop = False
        grad_acc_steps = params.grad_acc_steps
        steps_per_epoch = min(len(train_loader), params.max_steps_per_epoch)
        total_steps = min(params.max_num_epochs*steps_per_epoch, params.max_num_steps)
        # pbar = tqdm(desc='Epoch ?', total=total_steps)
        pbar = SimpleTableMetrics(['step','loss','val_loss','lr','eta'], every_n_steps=5)
        optimizer.zero_grad()

        for epoch in range(params.max_num_epochs):
            stats_dict = {'epoch':epoch, 'step':self.step, 'max_steps':self.step + params.max_steps_per_epoch}
            model.train()
            optimizer.zero_grad()
            epoch_steps = 0
            pbar.set_description(f'Epoch {epoch}')
            trues = []
            preds = []
            for batch in train_loader:
                output = model(batch)
                rows, cols = torch.triu_indices(6,6)
                true_4 = elasticity_func.stiffness_Mandel_to_cart_4(batch['stiffness'])
                pred_4 = elasticity_func.stiffness_Mandel_to_cart_4(output['stiffness'])
                directions = torch.randn(100, 3, device=batch['stiffness'].device)
                directions = directions / directions.norm(dim=-1, keepdim=True)
                stiff_true = torch.einsum('...ijkl,pi,pj,pk,pl->...p', true_4, directions, directions, directions, directions)
                stiff_pred = torch.einsum('...ijkl,pi,pj,pk,pl->...p', pred_4, directions, directions, directions, directions)
                loss = 0.3*torch.nn.functional.mse_loss(100*stiff_pred, 100*stiff_true) / grad_acc_steps
                true = batch['stiffness'][:, rows, cols]
                pred = output['stiffness'][:, rows, cols]
                loss += 0.7*torch.nn.functional.mse_loss(100*pred, 100*true) / grad_acc_steps
                
                loss_valid = torch.isfinite(loss).item()
                # backprop and optimizer update
                if loss_valid:
                    self.backward(loss)
                else:
                    self.print('Loss is not valid, skipping backprop')
                if (self.step+1) % grad_acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                params_valid = torch.isfinite(next(iter(model.parameters()))).sum().item()
                grads_valid = True
                if isinstance(next(iter(model.parameters())).grad, torch.Tensor):
                    grads_valid = torch.isfinite(next(iter(model.parameters())).grad).sum().item()
                model_valid = bool(params_valid and grads_valid)             
                                
                # log
                loss_disp = loss.item()*grad_acc_steps
                stats_dict.update({'loss':loss_disp, 'step':self.step})                
                pbar.set_postfix(stats_dict)
                self.update_stats(stats_dict)
                if ((epoch+1) % params.save_every_n_epochs == 0) or ((epoch+1)==params.max_num_epochs):
                    trues.append(true.detach().cpu())
                    preds.append(pred.detach().cpu())
                
                pbar.update()
                self.step += 1
                epoch_steps += 1

                # stopping conditions
                if self.step >= params.max_num_steps:
                    stop = True
                    self.print('Max number of steps reached, stopping')
                    break
                if torch.isnan(loss).item():
                    stop = True
                    self.print('Loss is NaN, stopping')
                    break
                if epoch_steps >= params.max_steps_per_epoch:
                    break

            stats_dict = {'epoch':epoch}
            val_trues = []
            val_preds = []
            if (epoch+1) % params.check_valid_every_n_epochs == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss = []
                    epoch_steps = 0
                    for batch in valid_loader:
                        output = model(batch)
                        rows, cols = torch.triu_indices(6,6)
                        true = batch['stiffness'][:, rows, cols]
                        pred = output['stiffness'][:, rows, cols]
                        loss = torch.nn.functional.mse_loss(100*pred, 100*true)

                        valid_loss.append(loss.item())
                        
                        stats_dict['_val_loss'] = loss.item()
                        pbar.set_postfix(stats_dict)
                        val_trues.append(true.cpu())
                        val_preds.append(pred.cpu())

                        epoch_steps += 1
                        if epoch_steps >= params.max_valid_steps:
                            break
                    
                    valid_loss = np.mean(valid_loss)
                    stats_dict.update({'val_loss': valid_loss, 'lr': optimizer.param_groups[0]['lr']})
                    self.update_stats(stats_dict)
                    pbar.set_postfix(stats_dict)
            if stop:
                break
                
            # scheduler.step()

            # save model and plot predictions
            if ((epoch+1) % params.save_every_n_epochs == 0) or ((epoch+1)==params.max_num_epochs):
                self.save(model.state_dict(), os.path.join(params.log_dir, f'ckpt_{self.step:06d}.pt'))
                self.plot_predictions(trues, preds, 'scatter', 'train_parity')
                self.plot_predictions(val_trues, val_preds, 'scatter', 'valid_parity')

        # self.writer.flush()
        self.eval(model, train_loader, mode='hist', max_steps=4, name='eval_train')
        self.eval(model, valid_loader, mode='hist', max_steps=4, name='eval_valid')
        

def main() -> None:    
    df = pd.read_csv('../adamw-hp-dim.csv', index_col=0)
    num_hp_trial = int(os.environ['NUM_HP_TRIAL'])
    
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
        check_valid_every_n_epochs=1,
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
    wandb.init(project='glamm-gnn-fresh', entity='ivan-grega', config=params)
    wandb.config["desc"]  = f"Exp-2, rho rescaling, 3 message passes"

    dt = datetime.now()
    run_name = dt.strftime("%Y-%m-%d_%H%M%S")
    log_dir = f'./{run_name}'
    os.makedirs(log_dir, exist_ok=True)
    print(log_dir)
    params.log_dir = log_dir
    with open(os.path.join(log_dir, 'params.json'), 'w') as f:
        json.dump(vars(params), f, indent=4)
    # shutil.copy2(__file__, log_dir)
    # shutil.copy2('./gnn/model_torch.py', log_dir)

    lite = Lite(accelerator='auto', precision=32)
    try:
        lite.run(params)
        wandb.finish()
    except Exception as e:
        traceback.print_exc()
        wandb.finish(-1)
        pass
    finally:
        df_stats = pd.DataFrame(lite.stats).T
        df_stats.to_csv(os.path.join(params.log_dir, 'stats.csv'))

if __name__=='__main__':
    main()  
# %%
valid_dset = GLAMM_Dataset(
    root='./GLAMMDsetFull',
    catalogue_path='./GLAMMDsetFull/raw/fixed_dset_1298_val.lat',
    dset_fname='validation.pt',
    n_reldens=5,
    choose_reldens='half',
    graph_ft_format='cartesian_4',
)