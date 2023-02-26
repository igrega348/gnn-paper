from typing import Any, Optional, Tuple, Dict, Union
from argparse import Namespace
import time
import logging

import torch
from torch_geometric.data import Batch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from e3nn import o3

from .blocks import (
    RepeatNodeEmbedding,
    RadialEmbeddingBlock,
    FourierBasisEmbeddingBlock,
    GeneralLinearReadoutBlock,
    GeneralNonLinearReadoutBlock,
    TensorProductInteractionBlock,
    TensorProductResidualInteractionBlock,
    EquivariantProductBlock,
    OneTPReadoutBlock,
    GlobalSumHistoryPooling
)
from .mace import get_edge_vectors_and_lengths

class LatticeGNN(pl.LightningModule):
    _time_metrics: Dict

    def __init__(
        self,
        params: Namespace, 
        *args: Any, 
        **kwargs: Any
    ) -> "LatticeGNN":
        super().__init__(*args, **kwargs)
        self.params = params

        hidden_irreps = o3.Irreps(params.hidden_irreps)

        self.register_buffer('loss_weights', torch.ones((1,21)))

        self.node_ft_embedding = RepeatNodeEmbedding(int(hidden_irreps.count(o3.Irrep(0,1))))
        self.radial_embedding = FourierBasisEmbeddingBlock(3)
        node_ft_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim+3}x0e")
        edge_attr_irreps = o3.Irreps.spherical_harmonics(params.lmax)
        self.spherical_harmonics = o3.SphericalHarmonics(
            edge_attr_irreps,
            normalize=True, normalization='component'
        )
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (edge_attr_irreps * num_features).sort()[0].simplify()
        readout_irreps = o3.Irreps(params.readout_irreps)

        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        for i in range(params.message_passes):
            if i==0:
                inter = TensorProductInteractionBlock(
                    node_feats_irreps=node_ft_irreps,
                    edge_attrs_irreps=edge_attr_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    irreps_out=interaction_irreps,
                    agg_norm_const=params.agg_norm_const,
                    reduce=params.interaction_reduction,
                    bias=params.interaction_bias
                )
                prod = EquivariantProductBlock(
                    node_feats_irreps=inter.irreps_out,
                    target_irreps=hidden_irreps,
                    correlation=params.correlation,
                    use_sc=False
                )
                rout = None 
            else:
                inter = TensorProductResidualInteractionBlock(
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=edge_attr_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    irreps_out=interaction_irreps,
                    sc_irreps_out=hidden_irreps,
                    agg_norm_const=params.agg_norm_const,
                    reduce=params.interaction_reduction,
                    bias=params.interaction_bias
                )
                prod = EquivariantProductBlock(
                    node_feats_irreps=inter.irreps_out,
                    target_irreps=hidden_irreps,
                    correlation=params.correlation,
                    use_sc=True
                )
                rout = GeneralNonLinearReadoutBlock(
                    irreps_in=hidden_irreps,
                    irreps_out=readout_irreps,
                    hidden_irreps=readout_irreps,
                    gate=torch.nn.functional.silu,
                )
            self.interactions.append(inter)
            self.products.append(prod)
            self.readouts.append(rout)
        
        self.pooling = GlobalSumHistoryPooling(reduce=params.global_reduction)
        self.linear = o3.Linear(readout_irreps, readout_irreps, 
            internal_weights=True, 
            shared_weights=True,
            biases=True
        )
        self.fourth_order_expansion = OneTPReadoutBlock(
            irreps_in=readout_irreps,
            irreps_out=o3.Irreps('2x0e+2x2e+1x4e')
        )

        self.save_hyperparameters()
        self._time_metrics = {}

    def forward(
        self,
        batch: Batch
    ) -> Dict:
        
        num_graphs = batch.num_graphs
        edge_index = batch.edge_index
        node_ft = self.node_ft_embedding(batch.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch.positions, edge_index=edge_index, shifts=batch.shifts
        )
        edge_length_embedding = self.radial_embedding(lengths)
        edge_radii = batch.edge_attr
        edge_feats = torch.cat(
            (edge_length_embedding, edge_radii, 10*edge_radii.pow(2), 100*edge_radii.pow(3)), 
            dim=1
        )
        edge_attrs = self.spherical_harmonics(vectors)
        
        outputs = []
        for interaction, product, readout in zip(self.interactions, self.products, self.readouts):
            node_ft, sc = interaction(node_ft, edge_attrs, edge_feats, edge_index)
            node_ft = product(node_ft, sc)
            if readout is not None:
                outputs.append(readout(node_ft))
        outputs = torch.stack(outputs, dim=-1)

        graph_ft = self.pooling(outputs, batch.batch, num_graphs)
        graph_ft = self.linear(graph_ft)
        stiffness = self.fourth_order_expansion(graph_ft) # [num_]

        return {'stiffness': stiffness}

    def configure_optimizers(self):
        params = self.params

        if (params.optimizer).lower()=='adamw':
            rank_zero_info('Setting optimizer AdamW')
            optimizer = torch.optim.AdamW(
                params=self.parameters(), lr=params.lr, 
                betas=(params.beta1,0.999), eps=params.epsilon,
                amsgrad=params.amsgrad, weight_decay=params.weight_decay,
            )
        elif (params.optimizer).lower()=='nadam':
            rank_zero_info('Setting optimizer NAdam')
            optimizer = torch.optim.NAdam(
                params=self.parameters(), lr=params.lr, 
                weight_decay=params.weight_decay,
            )            
        elif (params.optimizer).lower()=='sgd':
            rank_zero_info('Setting optimizer SGD')
            optimizer = torch.optim.SGD(
                params=self.parameters(), lr=params.lr, 
                momentum=params.momentum, nesterov=params.nesterov,
                weight_decay=params.weight_decay,
            )

        if not hasattr(params, 'scheduler') or not isinstance(params.scheduler, str):
            lr_scheduler = None
        elif (params.scheduler).lower()=="linearlr":
            rank_zero_info('Setting scheduler LinearLR')
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer, start_factor=1, end_factor=0.1,
                total_iters=params.max_num_epochs
            )

        if lr_scheduler is not None:
            return {"optimizer": optimizer, 'lr_scheduler':{
                'scheduler': lr_scheduler,
                'interval':'epoch', 'frequency':1
            }}  
        else:
            return {"optimizer": optimizer}

    # def configure_optimizers(self):
       
    #     args = self.params

    #     decay_interactions = {}
    #     no_decay_interactions = {}
    #     for name, param in self.interactions.named_parameters():
    #         if "linear.weight" in name or "skip_tp_full.weight" in name:
    #             decay_interactions[name] = param
    #         else:
    #             no_decay_interactions[name] = param

    #     param_options = dict(
    #     params=[
    #             {
    #                 "name": "interactions_decay",
    #                 "params": list(decay_interactions.values()),
    #                 "weight_decay": args.weight_decay,
    #             },
    #             {
    #                 "name": "interactions_no_decay",
    #                 "params": list(no_decay_interactions.values()),
    #                 "weight_decay": 0.0,
    #             },
    #             {
    #                 "name": "products",
    #                 "params": self.products.parameters(),
    #                 "weight_decay": args.weight_decay,
    #             },
    #             {
    #                 "name": "readouts",
    #                 "params": self.readouts.parameters(),
    #                 "weight_decay": 0.0,
    #             },
    #             {
    #                 "name": "fourth_order_expansion",
    #                 "params": self.fourth_order_expansion.parameters(),
    #                 "weight_decay": 0.0,
    #             }
    #         ],
    #         lr=args.lr,
    #         amsgrad=args.amsgrad,
    #     )
   
    #     if args.optimizer == "adamw":
    #         optimizer = torch.optim.AdamW(**param_options)
    #     else:
    #         optimizer = torch.optim.Adam(**param_options)

    #     if args.scheduler == "ExponentialLR":
    #         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #             optimizer=optimizer, gamma=args.lr_scheduler_gamma
    #         )
    #     elif args.scheduler == "ReduceLROnPlateau":
    #         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer=optimizer,
    #             factor=args.lr_factor,
    #             patience=args.scheduler_patience,
    #         )
    #     elif args.scheduler == "CosineAnnealing":
    #         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #             optimizer=optimizer,
    #             T_0=20,
    #             eta_min=0.0001
    #         )
    #     else:
    #         raise RuntimeError(f"Unknown scheduler: '{args.scheduler}'")
        
    #     if args.scheduler == "ReduceLROnPlateau":
    #         return {"optimizer": optimizer, 
    #                     "lr_scheduler": {
    #                         "scheduler": lr_scheduler,
    #                         "monitor": "val_loss",
    #                     }
    #                 }
    #     else:
    #         return {"optimizer": optimizer, "scheduler": lr_scheduler}  

    def training_step(self, batch, batch_idx):
        output = self(batch)

        loss = torch.nn.functional.smooth_l1_loss(
            self.loss_weights*output['stiffness'], self.loss_weights*batch['stiffness'], beta=1.0
        )
        # calculate 'percentage' error for each row of the output
        vals, _ = batch['stiffness'].abs().max(dim=1, keepdim=True)
        error = torch.mean((output['stiffness']-batch['stiffness']).abs()/vals, dim=1)
        train_err = torch.mean(error)

        self.log("loss", loss, 
            prog_bar=False, batch_size=batch.num_graphs,
            # on_step=False, on_epoch=True
            )
        self.log('train_err', train_err, 
            prog_bar=False, batch_size=batch.num_graphs, sync_dist=True
        )
        self.log("lr", self.optimizers().param_groups[0]['lr'], 
                prog_bar=False, batch_size=batch.num_graphs, 
                on_epoch=True, on_step=False, sync_dist=True
                )

        return loss    

    def validation_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self(batch)

        loss = torch.nn.functional.smooth_l1_loss(
            output['stiffness'], batch['stiffness'], beta=1.0
        )
        # calculate 'percentage' error for each row of the output
        vals, _ = batch['stiffness'].abs().max(dim=1, keepdim=True)
        error = torch.mean((output['stiffness']-batch['stiffness']).abs()/vals, dim=1)
        val_err = torch.mean(error)

        self.log("val_loss", loss, 
            prog_bar=True, batch_size=batch.num_graphs, sync_dist=True
        )
        self.log('val_err', val_err, 
            prog_bar=True, batch_size=batch.num_graphs, sync_dist=True
        )
        return output['stiffness'], batch['stiffness']

    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Tuple:
        """Returns (prediction, true)"""
        return self(batch), batch

    def on_train_start(self) -> None:
        self._time_metrics['_train_start_time'] = time.time()
        self._time_metrics['_train_start_step'] = self.global_step

    def on_train_epoch_start(self) -> None:
        self._time_metrics['_train_epoch_start_time'] = time.time()
        self._time_metrics['_train_epoch_start_step'] = self.global_step

    def on_train_epoch_end(self) -> None:
        tn = time.time()
        epoch_now = self.current_epoch + 1
        step_now = self.global_step
        time_per_epoch = (tn - self._time_metrics['_train_start_time'])/epoch_now
        epoch_steps = step_now - self._time_metrics['_train_epoch_start_step']
        total_steps = step_now - self._time_metrics['_train_start_step']
        time_per_step_total = (tn - self._time_metrics['_train_start_time'])/total_steps
        time_per_step_epoch = (tn - self._time_metrics['_train_epoch_start_time'])/epoch_steps
        
        if self.trainer.max_epochs>0:
            max_epochs = self.trainer.max_epochs
            eta = (max_epochs-epoch_now)*time_per_epoch
        elif self.trainer.max_steps>0:
            max_steps = self.trainer.max_steps
            eta = (max_steps - step_now)*time_per_step_total

        self.log("eta", eta,
                prog_bar=True, sync_dist=True
                )
        self.log('step_per_time', 1/time_per_step_epoch,
                prog_bar=False, sync_dist=True
                )