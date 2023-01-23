import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from e3nn import o3
from e3nn.io import CartesianTensor

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from gnn.models import LatticeGNN

def rotate_lat(lat: Data):
    Q = o3.rand_matrix()
    irreps = o3.Irreps('2x0e+2x2e+1x4e')
    R = irreps.D_from_matrix(Q)
    transformed = Data(
        node_attrs=lat.node_attrs,
        edge_attr=lat.edge_attr,
        edge_index=lat.edge_index,
        positions = torch.einsum('ij,pj->pi', Q, lat.positions),
        shifts = torch.einsum('ij,pj->pi', Q, lat.shifts),
        # unit_shifts = lat.unit_shifts,
        rel_dens=lat.rel_dens,
        stiffness=torch.einsum('pj,ij->pi', lat.stiffness, R),
        name = lat.name
    )
    return transformed

def compute_avg_num_neighbors(
    data_loader: torch.utils.data.DataLoader,
) -> float:
    num_neighbors = []
    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    num_neighbors = torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    avg_num_neighbors = torch.mean(num_neighbors)
    return avg_num_neighbors

def main() -> None:
    wandb_logger = WandbLogger(project="gnn-paper-v0", entity="ivan-grega")
    # wandb_logger.experiment.log_code('.')

    train_dset = GLAMM_Dataset(
        root='C:/temp/gnn-paper-data/GLAMMDset',
        catalogue_path='C:/temp/gnn-paper-data/train.lat',
        dset_fname='train.pt',
        n_reldens=4,
        graph_ft_format='cartesian_4',
    )
    el_tens = CartesianTensor('ijkl=jikl=ijlk=klij')
    train_dset.data.stiffness = train_dset.data.stiffness / train_dset.data.rel_dens.view(-1,1,1,1,1)
    normalization_factor = 2/torch.max(torch.abs(train_dset.data.stiffness))
    train_dset.data.stiffness = el_tens.from_cartesian(train_dset.data.stiffness*normalization_factor).float()

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=50,
        shuffle=True,
        num_workers=4
    )

    val_dset = GLAMM_Dataset(
        root='C:/temp/gnn-paper-data/GLAMMDset',
        catalogue_path='C:/temp/gnn-paper-data/train.lat',
        dset_fname='valid.pt',
        n_reldens=4,
        choose_reldens='last',
        graph_ft_format='cartesian_4',
    )
    val_dset.data.stiffness = val_dset.data.stiffness / val_dset.data.rel_dens.view(-1,1,1,1,1)
    val_dset.data.stiffness = el_tens.from_cartesian(val_dset.data.stiffness*normalization_factor).float()

    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=100,
        shuffle=False,
        num_workers=4
    )


    # avg_num_neighbours = compute_avg_num_neighbors(train_loader)
    # print(f'Average number of neighbors: {avg_num_neighbours}')

    model = LatticeGNN(
        hidden_irreps='32x0e+32x1o+32x2e',
    )
    wandb_logger.watch(model, log="all", log_freq=50)


    callbacks = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(save_top_k=1, monitor='val_loss')
    ]
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=10,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        gradient_clip_val=10,
        gradient_clip_algorithm='norm',
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__=='__main__':
    main()