# %%
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s @ %(module)s: %(funcName)s :: %(message)s', level=logging.INFO)
import numpy as np
from data.datasets import assemble_catalogue
from data import Catalogue
# %%
# if __name__=='__main__':
#     df = assemble_catalogue(
#         num_base_lattices=9000,
#         # imperfection_levels=[0.0],
#         imperfection_levels=[0.0,0.02,0.05],
#         num_imperf_realisations=1,
#         input_dir='C:/temp/gnn-paper-data',
#         choose_base='first',
#         choose_imperf='first',
#         return_df=True,
#         multiprocessing=8
#     )
#     # %%
#     dset_name = ''
#     #
#     all_base_names = df['base_name'].unique()
#     print(f'All base names: {len(all_base_names)}')
#     np.random.shuffle(all_base_names)
#     train_base_names = all_base_names[:7000]
#     val_base_names = all_base_names[7000:]
#     print(f'Train base names {len(train_base_names)}')
#     print(f'Validation base names {len(val_base_names)}')

#     selected_df = df.loc[df['base_name'].isin(train_base_names), :]
#     selected_lat_dict = selected_df.to_dict('index')
#     new_cat = Catalogue.from_dict(selected_lat_dict)
#     new_cat.to_file(f'C:/temp/gnn-paper-data/{dset_name}_dset_{len(train_base_names)}_train.lat')

#     selected_df = df.loc[df['base_name'].isin(val_base_names), :]
#     selected_lat_dict = selected_df.to_dict('index')
#     new_cat = Catalogue.from_dict(selected_lat_dict)
#     new_cat.to_file(f'C:/temp/gnn-paper-data/{dset_name}_dset_{len(val_base_names)}_val.lat')
# %%
if __name__=='__main__':
    df = assemble_catalogue(
        num_base_lattices=1,
        imperfection_levels=[0.0,0.01,0.02,0.03,0.04,0.05,0.07,0.1],
        num_imperf_realisations=10,
        input_dir='C:/temp/gnn-paper-data',
        choose_base='random',
        choose_imperf='first',
        return_df=False,
        output_fn='C:/temp/one_lat_imperf.lat',
        multiprocessing=6
    )