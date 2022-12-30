import os
import os.path as osp
from typing import Optional, Any

import copy
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities import rank_zero_info
from .plotting import parity_plot

class MyCallbacks(Callback):
    def __init__(self, run_params: dict) -> None:
        super().__init__()
        save_train_data = False
        if 'save_train_data' in run_params['callbacks']:
            save_train_data = run_params['callbacks']['save_train_data']
        self.save_train_data = save_train_data

        save_val_data = False
        if 'save_val_data' in run_params['callbacks']:
            save_val_data = run_params['callbacks']['save_val_data']
        self.save_val_data = save_val_data

        end_val_parity = False
        if 'end_val_parity' in run_params['callbacks']:
            end_val_parity = run_params['callbacks']['end_val_parity']
        self.end_val_parity = end_val_parity

        self.run_params = run_params

        self.train_pred = []
        self.train_true = []
        self.val_pred = []
        self.val_true = []

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        lines = []
        if self.save_val_data or self.save_train_data:
            lines.append('Raw data in npy format saved as two columns\n')
            lines.append('col0: true\t col1: pred\n')
            fn = f'README'
            dname = osp.join(trainer.default_root_dir, 'callbacks', 'data')
            os.makedirs(dname, exist_ok=True)
            fn = osp.join(dname, fn)
            with open(fn, 'w') as fout:
                fout.writelines(lines)


    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        pred, true = outputs
        if self.save_val_data or self.end_val_parity:
            val_pred = pred.detach().cpu().numpy()
            val_true = true.detach().cpu().numpy()
            self.val_pred.append(val_pred)
            self.val_true.append(val_true)
        
    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_pred = []
        self.val_true = []


    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.save_val_data:
            val_pred = np.concatenate(self.val_pred)
            val_true = np.concatenate(self.val_true)
            fn = f'{trainer.current_epoch}_val_data.npy'
            fn = osp.join(trainer.default_root_dir, 'callbacks', 'data', fn)
            np.save(fn, np.column_stack((val_true, val_pred)))

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.end_val_parity:
            val_pred = np.concatenate(self.val_pred)
            val_true = np.concatenate(self.val_true)
            fn = f'{trainer.current_epoch}_val_parity.png'
            fn = osp.join(trainer.default_root_dir, 'callbacks', 'plots', fn)
            parity_plot(val_true, val_pred, fn)


class PrintTableMetrics(Callback):
    def __init__(self, run_params: dict) -> None:
        super().__init__()

        col_width = 10
        s = "{:<{}}"

        header = []
        for metric in run_params['callbacks']['log_metrics']:
            header.append(metric)
        if 'epoch' not in header:
            header.insert(0, "epoch")
        
        n_cols = len(header)
        self.col_width = col_width
        total_width = col_width * n_cols + 3*n_cols
        self.total_width = total_width
        
        self.header = header
        self.run_params = run_params

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