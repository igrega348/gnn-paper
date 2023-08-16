# %%
import sys
import os
from math import ceil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch
from tqdm import tqdm, trange
from e3nn import o3

from data.datasets import GLAMM_rhotens_Dataset as GLAMM_Dataset
from data import elasticity_func
# %%
class RotateLat:
    def __call__(self, lat: Data):
        Q = o3.rand_matrix()
        C_rot = torch.einsum('pijkl,ai,bj,ck,dl->pabcd', lat.stiffness, Q, Q, Q, Q)
        S_rot = torch.einsum('pijkl,ai,bj,ck,dl->pabcd', lat.compliance, Q, Q, Q, Q)
        C_mand = elasticity_func.stiffness_cart_4_to_Mandel(C_rot)
        S_mand = elasticity_func.stiffness_cart_4_to_Mandel(S_rot)
        transformed = Data(
            node_attrs=lat.node_attrs,
            edge_attr=lat.edge_attr,
            edge_index=lat.edge_index,
            positions = torch.einsum('ij,pj->pi', Q, lat.positions),
            shifts = torch.einsum('ij,pj->pi', Q, lat.shifts),
            rel_dens=lat.rel_dens,
            stiffness=C_mand,
            compliance=S_mand,
            name = lat.name
        )
        return transformed
    
def load_datasets(root: str, train_name: str, val_name: str):
    train_dset = GLAMM_Dataset(
        root=root,
        catalogue_path=os.path.join(root, 'raw', train_name),
        transform=RotateLat(),
        dset_fname='train.pt',
        n_reldens=10,
        choose_reldens='first',
        graph_ft_format='cartesian_4',
    )
    rank_zero_info(train_dset)

    valid_dset = GLAMM_Dataset(
        root=root,
        catalogue_path=os.path.join(root, 'raw' ,val_name),
        transform=RotateLat(),
        dset_fname='validation.pt',
        n_reldens=5,
        choose_reldens='half',
        graph_ft_format='cartesian_4',
    )
    rank_zero_info(valid_dset)

    # scaling and normalization
    normalization_factor = 100
    train_dset.data.stiffness =  (train_dset.data.stiffness / train_dset.data.rel_dens.view(-1,1,1,1,1) * normalization_factor).float()
    train_dset.data.compliance = (train_dset.data.compliance * train_dset.data.rel_dens.view(-1,1,1,1,1) / normalization_factor).float()
    valid_dset.data.stiffness =  (valid_dset.data.stiffness / valid_dset.data.rel_dens.view(-1,1,1,1,1) * normalization_factor).float()
    valid_dset.data.compliance = (valid_dset.data.compliance * valid_dset.data.rel_dens.view(-1,1,1,1,1) / normalization_factor).float()

    return train_dset, valid_dset
# %%
tiny_dset_train, tiny_dset_val = load_datasets('./GLAMMDsetT', 'tiny_dset_7000_train.lat', 'tiny_dset_1298_val.lat')
# %%
C = tiny_dset_train.data.stiffness
S = tiny_dset_train.data.compliance
directions = torch.randn(100,3)
directions = directions / directions.norm(dim=1, keepdim=True)
C_proj = torch.einsum('pijkl,qi,qj,qk,ql->pq', C, directions, directions, directions, directions)
S_proj = torch.einsum('pijkl,qi,qj,qk,ql->pq', S, directions, directions, directions, directions)
# %%
sns.histplot(C_proj.flatten().numpy(), bins=100, stat='density')
plt.show()
sns.histplot(S_proj.flatten().numpy(), bins=100, log_scale=True, stat='density')
plt.show()
# %%
full_dset_train, full_dset_val = load_datasets('E:/dset_0', 'full_dset_7000_train.lat', 'full_dset_620_val.lat')
# %% 
C_proj = []
S_proj = []
batch_size = 2**16
n_batches = ceil(len(full_dset_train) / batch_size)
directions = torch.randn(100,3).to('cuda')
directions = directions / directions.norm(dim=1, keepdim=True)
for batch_idx in trange(n_batches):
    C = full_dset_train.data.stiffness[batch_idx*batch_size:(batch_idx+1)*batch_size].to('cuda')
    S = full_dset_train.data.compliance[batch_idx*batch_size:(batch_idx+1)*batch_size].to('cuda')
    
    C_proj.append(torch.einsum('pijkl,qi,qj,qk,ql->pq', C, directions, directions, directions, directions).to('cpu'))
    S_proj.append(torch.einsum('pijkl,qi,qj,qk,ql->pq', S, directions, directions, directions, directions).to('cpu'))
    
C_proj = torch.cat(C_proj, dim=0)
S_proj = torch.cat(S_proj, dim=0)
# %%
sns.histplot(C_proj.flatten().numpy(), bins=100, stat='density')
plt.show()
sns.histplot(S_proj.flatten().numpy(), bins=100, log_scale=True, stat='density')
plt.show()
# %% How much overlap do we have in lattice names?
tiny_train_names = set(tiny_dset_train.data.name)
tiny_val_names = set(tiny_dset_val.data.name)
tiny_names = tiny_train_names.union(tiny_val_names)
tiny_base_names = set(['_'.join(name.split('_')[:3]) for name in tiny_names])

full_train_names = set(full_dset_train.data.name)
full_val_names = set(full_dset_val.data.name)
full_names = full_train_names.union(full_val_names)
full_base_names = set(['_'.join(name.split('_')[:3]) for name in full_names])

# intersection of base names
matching_base_names = list(tiny_base_names.intersection(full_base_names))
len(matching_base_names)
# %% Compare the lattices without imperfections for the two datasets
# we don't have matching relative densities, so we'll plot a fit line
random_base_name = np.random.choice(matching_base_names)

names = tiny_dset_train.data.name
base_names = ['_'.join(name.split('_')[:3]) for name in names]
imperfections = [float(name.split('_')[4]) for name in names]
ind = np.flatnonzero((np.array(base_names) == random_base_name) & (np.array(imperfections) == 0))
rel_dens_tiny = tiny_dset_train.data.rel_dens[ind].numpy()
print(rel_dens_tiny)
C_tiny = elasticity_func.stiffness_cart_4_to_Mandel(tiny_dset_train.data.stiffness[ind]).numpy()
S_tiny = elasticity_func.stiffness_cart_4_to_Mandel(tiny_dset_train.data.compliance[ind]).numpy()

names = full_dset_train.data.name
base_names = ['_'.join(name.split('_')[:3]) for name in names]
imperfections = [float(name.split('_')[4]) for name in names]
ind = np.flatnonzero((np.array(base_names) == random_base_name) & (np.array(imperfections) == 0))
rel_dens_full = full_dset_train.data.rel_dens[ind].numpy()
print(rel_dens_full)
C_full = elasticity_func.stiffness_cart_4_to_Mandel(full_dset_train.data.stiffness[ind]).numpy()
S_full = elasticity_func.stiffness_cart_4_to_Mandel(full_dset_train.data.compliance[ind]).numpy()
# %% Plot scatter plot as a function of relative density on log scale
rows, cols = np.triu_indices(6)
plt.plot(rel_dens_tiny, C_tiny[:,rows, cols], '+', label='tiny')
plt.plot(rel_dens_full, C_full[:,rows, cols], 'o', label='full')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
# %% Now similar but projections in 10 random directions
# note the y-axis variables are specific stiffness and compliance
# need to multiply or divide by relative density to get stiffness and compliance
os.makedirs('E:/dset_comparisons', exist_ok=True)
directions = torch.randn(10,3)
directions = directions / directions.norm(dim=1, keepdim=True)

random_base_name = np.random.choice(matching_base_names)

names = tiny_dset_train.data.name
base_names = ['_'.join(name.split('_')[:3]) for name in names]
imperfections = [float(name.split('_')[4]) for name in names]
ind = np.flatnonzero((np.array(base_names) == random_base_name) & (np.array(imperfections) == 0))
rel_dens_tiny = tiny_dset_train.data.rel_dens[ind].numpy()
print(rel_dens_tiny)
C_bar_tiny = torch.einsum('pijkl,qi,qj,qk,ql->pq', tiny_dset_train.data.stiffness[ind], directions, directions, directions, directions).numpy()
S_bar_tiny = torch.einsum('pijkl,qi,qj,qk,ql->pq', tiny_dset_train.data.compliance[ind], directions, directions, directions, directions).numpy()
C_tiny = C_bar_tiny * rel_dens_tiny[:,None]
S_tiny = S_bar_tiny / rel_dens_tiny[:,None]

names = full_dset_train.data.name
base_names = ['_'.join(name.split('_')[:3]) for name in names]
imperfections = [float(name.split('_')[4]) for name in names]
ind = np.flatnonzero((np.array(base_names) == random_base_name) & (np.array(imperfections) == 0))
rel_dens_full = full_dset_train.data.rel_dens[ind].numpy()
print(rel_dens_full)
C_bar_full = torch.einsum('pijkl,qi,qj,qk,ql->pq', full_dset_train.data.stiffness[ind], directions, directions, directions, directions).numpy()
S_bar_full = torch.einsum('pijkl,qi,qj,qk,ql->pq', full_dset_train.data.compliance[ind], directions, directions, directions, directions).numpy()
C_full = C_bar_full * rel_dens_full[:,None]
S_full = S_bar_full / rel_dens_full[:,None]

# % stiffness and compliance on 2 plots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.plot(rel_dens_tiny, C_tiny, '+', label='tiny')
ax1.plot(rel_dens_full, C_full, 'o', markerfacecolor='none', label='full')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title(f'Stiffness {random_base_name}')
ax1.set_xlabel('Relative density')
ax1.set_ylabel('Stiffness')
ax2.plot(rel_dens_tiny, S_tiny, '+', label='tiny')
ax2.plot(rel_dens_full, S_full, 'o', markerfacecolor='none', label='full')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title(f'Compliance {random_base_name}')
ax2.set_xlabel('Relative density')
ax2.set_ylabel('Compliance')
plt.savefig(f'E:/dset_comparisons/{random_base_name}.png', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='w')
plt.show()
# with normalized data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.plot(rel_dens_tiny, C_bar_tiny, '+', label='tiny')
ax1.plot(rel_dens_full, C_bar_full, 'o', markerfacecolor='none', label='full')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title(f'Stiffness {random_base_name}')
ax1.set_xlabel('Relative density')
ax1.set_ylabel('Specific stiffness')
ax2.plot(rel_dens_tiny, S_bar_tiny, '+', label='tiny')
ax2.plot(rel_dens_full, S_bar_full, 'o', markerfacecolor='none', label='full')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title(f'Compliance {random_base_name}')
ax2.set_xlabel('Relative density')
ax2.set_ylabel('Normalized compliance')
plt.savefig(f'E:/dset_comparisons/{random_base_name}_normalized.png', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='w')