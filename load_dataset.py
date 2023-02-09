# %%
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s @ %(module)s: %(funcName)s :: %(message)s', level=logging.INFO)
import numpy as np
from data.datasets import assemble_catalogue
from data import Catalogue
# %%
df = assemble_catalogue(
    num_base_lattices=9000,
    imperfection_levels=[0.0,0.02,0.05],
    num_imperf_realisations=10,
    input_dir='C:/temp/gnn-paper-data',
    choose_base='first',
    choose_imperf='first',
    # output_fn='C:/temp/gnn-paper-data/presentation.lat',
    return_df=True
)
# %%
all_base_names = df['base_name'].unique()
len(all_base_names)
np.random.shuffle(all_base_names)
train_base_names = all_base_names[:7000]
val_base_names = all_base_names[7000:]

selected_df = df.loc[df['base_name'].isin(train_base_names), :]
selected_lat_dict = selected_df.to_dict('index')
new_cat = Catalogue.from_dict(selected_lat_dict)
new_cat.to_file('C:/temp/gnn-paper-data/dset_7000_train.lat')

selected_df = df.loc[df['base_name'].isin(val_base_names), :]
selected_lat_dict = selected_df.to_dict('index')
new_cat = Catalogue.from_dict(selected_lat_dict)
new_cat.to_file('C:/temp/gnn-paper-data/dset_1295_val.lat')

# %%
# from data import GLAMM_rhotens_Dataset as GLAMM_Dataset
# dataset = GLAMM_Dataset(
#     root='C:/temp/gnn-paper-data/GLAMMDset',
#     catalogue_path='C:/temp/gnn-paper-data/train.lat',
#     dset_fname='train.pt',
#     n_reldens=10,
# )