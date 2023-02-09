import copy
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import wandb

class PrintTableMetrics(pl.Callback):
    def __init__(self, log_metrics: list) -> None:
        super().__init__()

        col_width = 10
        s = "{:<{}}"

        header = []
        for metric in log_metrics:
            header.append(metric)
        if 'epoch' not in header:
            header.insert(0, "epoch")
        
        n_cols = len(header)
        self.col_width = col_width
        total_width = col_width * n_cols + 3*n_cols
        self.total_width = total_width
        
        self.header = header

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        s = "{:<{}}"
        fields = [s.format(metric, self.col_width) for metric in self.header]
        line = " | ".join(fields) + "\n" + "-" * self.total_width
        rank_zero_info(line)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        line = "\n" + "-" * self.total_width
        rank_zero_info(line)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metrics_dict = copy.copy(trainer.callback_metrics)
        local_dict = {key: metrics_dict[key].detach().cpu().item() for key in metrics_dict.keys()}
        local_dict['epoch'] = trainer.current_epoch
        rank_zero_info(self._format_table(local_dict))

    
    def _format_table(self, metrics: dict) -> str:
        # Formatting
        s = "{:<{}}"
        fields = []
        for key in self.header:
            if key in metrics:
                if isinstance(metrics[key], float):
                    val = f'{metrics[key]:.6f}'
                else:
                    val = metrics[key]
                fields.append(s.format(val, self.col_width))
            else:
                fields.append(s.format('', self.col_width))
        line =  " | ".join(fields)
        return line

def upload_evaluations(results, name: str):
    outputs = torch.cat([tup[0]['stiffness'] for tup in results], dim=0).cpu().numpy()
    trues = torch.cat([tup[1]['stiffness'] for tup in results], dim=0).cpu().numpy()
    if 'train' in name:
        txt = ''
        ttl = 'Parity'
    elif 'val' in name:
        txt = 'val_'
        ttl = 'Val_parity'
    total_error = []
    for k in range(trues.shape[1]):
        data_table = np.column_stack((trues[:,k], outputs[:,k]))
        error = np.mean(np.abs(outputs[:,k] - trues[:,k]))/np.max(np.abs(trues[:,k]))
        total_error.append(error)
        table = wandb.Table(data=data_table, columns=[f'{txt}true{k}', f'{txt}pred{k}'])
        wandb.log({f"{ttl}{k}":wandb.plot.scatter(table=table, x=f"{txt}true{k}", y=f"{txt}pred{k}", title=f'#{k}, err={error*100:.2g}%')})
    max_error = max(total_error)
    avg_error = sum(total_error)/len(total_error)
    logging.info(f'{name}: average error={avg_error*100:.2g}%, max_error={max_error*100:.2g}%')
    wandb.log({f'{name}_avg_err':avg_error, f'{name}_max_err':max_error})

def log_matrix2(input: torch.Tensor, target: torch.Tensor, name: str):
    S = torch.zeros((6,13))

    imin = 0
    imax = 6
    for S_i in [input.clone(), target.clone()]:
        maxampl = torch.abs(S_i).max()
        S_i[torch.abs(S_i)<0.005*maxampl] = 0.0
        S[:, imin:imax] = S_i
        imin += 7
        imax += 7
    S[:,6] = torch.nan
    plt.imshow(S.numpy(), cmap='coolwarm')
    for i in range(6):
        for j in range(6):
            plt.text(j, i, f'{S[i,j]:.2g}', ha='center', va='center', fontsize=9)
        for j in range(7,13):
            plt.text(j, i, f'{S[i,j]:.2g}', ha='center', va='center', fontsize=9)

    error = torch.nn.functional.l1_loss(input, target)
    plt.text(6, 1.5, f'L1', ha='center', va='top')
    plt.text(6, 2, f'Error', ha='center', va='top')
    plt.text(6, 3, f'{error:.3f}', ha='center', va='bottom', fontsize=8)
    plt.xticks([])
    plt.yticks([])

    wandb.log({name: plt})

    plt.clf()