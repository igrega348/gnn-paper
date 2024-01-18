# %%
import os
import sys
from argparse import Namespace
import json
from pathlib import Path
import datetime

from tqdm import tqdm, trange
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
from torch_geometric.data import Batch, Data
import matplotlib.pyplot as plt

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from data import elasticity_func, Catalogue, Lattice
from utils import plotting, abaqus
from exp_180.model_torch import PositiveLiteGNN
from exp_180.train_mace import LightningWrappedModel, RotateLat, load_datasets, obtain_errors, aggr_errors

n_2_bn = lambda name: '_'.join(name.split('_')[:3])

# %%
num_hp_trial = 60
params_path = Path(__file__).parent.parent / Path(f'gnn-fresh-exp/exp-180/results/params-{num_hp_trial}.json')
params = json.loads(params_path.read_text())
params = Namespace(**params)
log_dir =Path(__file__).parent.parent / Path(f'gnn-fresh-exp/exp-180') / Path(params.log_dir)
ckpts = list(log_dir.glob('**/epoch*.ckpt'))
ckpt_path = ckpts[-1]

############# setup model ##############
lightning_model = LightningWrappedModel.load_from_checkpoint(ckpt_path, model=PositiveLiteGNN, params=params)
trainer = pl.Trainer(
    accelerator='auto',
)
# %%
############# run testing ##############
rank_zero_info('Testing')
test_dset = load_datasets(which='0imp', tag='test', parent='../ICLR2024/datasets', reldens_norm=False, rotate=False)
train_dset = load_datasets(which='0imp', tag='train', parent='../ICLR2024/datasets', reldens_norm=False, rotate=False)
# %%
train_loader = DataLoader(
    dataset=train_dset, batch_size=64, 
    shuffle=False, 
)
test_results = trainer.predict(lightning_model, train_loader, return_predictions=True, ckpt_path=ckpt_path)
# %% find best performing lattices
names = np.concatenate([res[1].name for res in test_results], axis=0)
rel_dens = np.concatenate([res[1].rel_dens for res in test_results], axis=0)
preds = np.concatenate([res[0]['stiffness'].detach().cpu().numpy() for res in test_results], axis=0)
targets = np.concatenate([res[1].stiffness.numpy() for res in test_results], axis=0)
# %%
target_4 = elasticity_func.stiffness_Mandel_to_cart_4(torch.from_numpy(targets))
dirs = torch.randn(200,3)
dirs = dirs/torch.norm(dirs, dim=1, keepdim=True)
dir_stiff = torch.einsum('...ijkl,pi,pj,pk,pl->...p', target_4, dirs, dirs, dirs, dirs)
# %% Calculate how much max-min stiffness is as percentage of mean
maxmin = torch.max(dir_stiff, dim=1).values - torch.min(dir_stiff, dim=1).values
mean = torch.mean(dir_stiff, dim=1)
rel = maxmin/mean
# %%
errors = np.square(preds-targets)
errors = np.mean(errors, axis=(1,2))
# %%
df = pd.DataFrame({'name': names, 'rel_dens': rel_dens, 'error': errors, 'variation': rel})
df
# %%
cat = Catalogue.from_file('../ICLR2024/datasets/0imp/raw/training_cat.lat', 0)
# %%
name = df[df['variation']>0.3].sort_values('error').head(50).iloc[9]['name']
df[df['variation']>0.3].sort_values('error').head(50)
# %%
# name = 'cub_Z04.4_E11721_p_0.0_7219241959159375425'
name = 'cub_Z04.4_E14064_p_0.0_4687573218853323851'
print(name)
inds = np.flatnonzero(names==name)
C4 = target_4[inds[0]].numpy()
plotting.plotly_stiffness_surf(C4)
# %%
data = cat[name]
lat = Lattice(
        name=name,
        nodal_positions=data['nodal_positions'],
        fundamental_edge_adjacency=data['fundamental_edge_adjacency'], 
        fundamental_tesselation_vecs=data['fundamental_tesselation_vecs'],
        lattice_constants=data['lattice_constants'],
    )
lat = lat.create_windowed()
lat.calculate_fundamental_representation()
plotting.plotly_unit_cell_3d(lat)
# %%
C2 = preds[inds[0]]
with np.printoptions(precision=3, suppress=True):
    print(C2)
C4 = elasticity_func.stiffness_Mandel_to_cart_4(torch.from_numpy(targets[inds[0]]))
plotting.plotly_stiffness_surf(C4).show()
fig, ax = make_polar_plot(C4.numpy())

C2 = np.copy(C2)
C2[1,1] -= 10

# C2 = np.copy(C2)
# C2[0,1] = 5
# C2[1,0] = 5
# C2[1,2] = 5
# C2[2,1] = 5

with np.printoptions(precision=3, suppress=True):
    print(C2)
C4 = elasticity_func.stiffness_Mandel_to_cart_4(torch.from_numpy(C2))
plotting.plotly_stiffness_surf(C4).show()
fig, ax = make_polar_plot(C4.numpy(), fig)
fig.savefig(f'C:/temp/optim/{n_2_bn(name)}_initial.svg')
# %%
batch = Batch.from_data_list([train_dset[inds[0]]])
# plot initial lattice
num_edges = batch.edge_index.shape[1]//2
lat = Lattice(
    name='initial',
    nodal_positions=batch.positions.numpy(),
    fundamental_edge_adjacency=batch.edge_index.numpy().T[:num_edges,:],
    fundamental_tesselation_vecs=np.concatenate([np.zeros((num_edges,3)), batch.shifts.numpy()[:num_edges,:]], axis=1)
)
lat = lat.create_windowed()
lat.calculate_fundamental_representation()
abaqus.write_abaqus_inp_normals(
        lat,
        strut_radii=np.atleast_2d(batch.edge_attr.min().item()),
        metadata={
            'Job name':'init',
            'Lattice name':name,
            'Base lattice':n_2_bn(name),
            'Date':datetime.datetime.now().strftime("%Y-%m-%d"), 
            'Relative densities': ', '.join([str(batch.rel_dens.item())]),
            'Unit cell volume':f'{lat.UC_volume:.5g}',
            'Description':f'Fresh dataset, varying strut radii',
            'Imperfection level':f'custom',
            'Hash':'2390434320',
        },
        fname=os.path.join('C:/temp/optim/', 'init.inp'),
        element_type='B31',
        mesh_params={'max_length':1, 'min_div':3}
    )

plotting.plotly_unit_cell_3d(lat).show()
# initial surface
C4 = elasticity_func.stiffness_Mandel_to_cart_4(batch.stiffness).squeeze()
plotting.plotly_stiffness_surf(C4.numpy()).show()
# target surface
C4 = elasticity_func.stiffness_Mandel_to_cart_4(torch.from_numpy(C2))
plotting.plotly_stiffness_surf(C4.numpy()).show()
init_positions = batch.positions.clone()
positions = batch.positions.clone() + torch.randn_like(batch.positions)*0.00001
# init_radii = batch.edge_attr.clone()
# radii = batch.edge_attr.clone()
pbar = trange(50)
for i in pbar:
    batch = Batch.from_data_list([train_dset[inds[0]]])
    batch.positions = positions.clone()
    # batch.edge_attr = radii.clone()
    # batch.edge_attr.requires_grad = True
    batch.positions.requires_grad = True
    lr = 0.0001
    output = lightning_model.model(batch)['stiffness']
    # if (i % 10) == 0:
        # print(output)
        # print(target)
    loss = torch.nn.functional.mse_loss(output, torch.from_numpy(C2).view(1,6,6))
    grad = torch.autograd.grad(loss, batch.positions)
    positions -= lr*grad[0]
    # grad = torch.autograd.grad(loss, batch.edge_attr)
    # radii -= lr*grad[0].mean()
    # print(loss)
    pbar.set_description(f'loss: {loss.item()}')

# plot final lattice
num_edges = batch.edge_index.shape[1]//2
lat = Lattice(
    name='final',
    nodal_positions=batch.positions.detach().numpy(),
    fundamental_edge_adjacency=batch.edge_index.numpy().T[:num_edges,:],
    fundamental_tesselation_vecs=np.concatenate([np.zeros((num_edges,3)), batch.shifts.numpy()[:num_edges,:]], axis=1)
)
lat = lat.create_windowed()
lat.calculate_fundamental_representation()
abaqus.write_abaqus_inp_normals(
        lat,
        strut_radii=np.atleast_2d(batch.edge_attr.mean().item()),
        metadata={
            'Job name':'optim',
            'Lattice name':name,
            'Base lattice':n_2_bn(name),
            'Date':datetime.datetime.now().strftime("%Y-%m-%d"), 
            'Relative densities': ', '.join([str(batch.rel_dens.item())]),
            'Unit cell volume':f'{lat.UC_volume:.5g}',
            'Description':f'Fresh dataset, varying strut radii',
            'Imperfection level':f'custom',
            'Hash':'2390434320',
        },
        fname=os.path.join('C:/temp/optim/', 'optim.inp'),
        element_type='B31',
        mesh_params={'max_length':1, 'min_div':3}
    )

plotting.plotly_unit_cell_3d(lat).show()
# final surface
C4 = elasticity_func.stiffness_Mandel_to_cart_4(output).squeeze()
plotting.plotly_stiffness_surf(C4.detach().numpy()).show()
# %%
fig, ax = make_polar_plot(C4.detach().numpy())
fig.savefig(f'C:/temp/optim/{n_2_bn(name)}_final.svg')
# %%
results = abaqus.run_abq_sim(['init', 'optim'], 'C:/temp/optim/')
# %%
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
with np.printoptions(precision=3, suppress=True):
    for i in range(2):
        Sv = next(iter(results[i]['compliance_tensors'].items()))[1]
        Cv = np.linalg.inv(Sv)
        C2_abq = elasticity_func.stiffness_Voigt_to_Mandel(Cv)*10000 * 49.061 / 20.485
        C4_abq = elasticity_func.stiffness_Mandel_to_cart_4(torch.from_numpy(C2_abq))
        # fig = plotting.plotly_stiffness_surf(C4_abq.numpy())

        # scene_dict = {'xaxis': {'showbackground': False}, 'yaxis': {'showbackground': False}, 'zaxis': {'showbackground': False}}
        # fig.update_layout(
        #     scene=scene_dict, scene2=scene_dict, scene3=scene_dict, scene4=scene_dict,
        # )
        # # set camera
        # camera = dict(
        #     eye=dict(x=2.0, y=1.0, z=1.25)
        # )
        # fig.update_layout(scene_camera=camera)
        # fig.write_image(f'C:/temp/optim/E14064_{i}.png', width=1200, height=1200)
        # fig.show()

        fig, ax = make_polar_plot(C4_abq.numpy(), fig)

        print(C2_abq)
fig.savefig('C:/temp/optim/abaqus.svg')
# %%
def export_to_blender(lat: Lattice):
    # for blender we need edge coordinates format
    edge_coords = lat._node_adj_to_ec(lat.transformed_node_coordinates, lat.edge_adjacency)
    # write to csv
    df = pd.DataFrame(edge_coords)
    df.to_csv(f'C:/temp/optim/edge_coords_{lat.name}.csv', header=False, index=False, float_format='%.6f')
# %%
export_to_blender(lat)
# %%
def make_polar_plot(C4: np.ndarray, fig = None):

    phi = np.linspace(0, 2*np.pi, 200)
    z = np.zeros_like(phi)
    x = np.cos(phi)
    y = np.sin(phi)
    direc = np.stack([x,y,z], axis=1)

    e = np.einsum('...ijkl,pi,pj,pk,pl->...p', C4, direc, direc, direc, direc)
    # polar plot
    if fig is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    else:
        ax = fig.axes[0]
    ax.plot(phi, e)
    # remove grid
    ax.grid(False)
    # set limits
    ax.set_ylim(0, 55)
    return fig, ax
# %%
