# %%
# import logging
# logging.basicConfig(format='%(asctime)s %(levelname)s @ %(module)s: %(funcName)s :: %(message)s', level=logging.INFO)
# from data.datasets import assemble_catalogue

# assemble_catalogue(
#     num_base_lattices=500,
#     imperfection_levels=[0.01,0.02],
#     num_imperf_realisations=7,
#     input_dir='C:/temp/gnn-paper-data',
#     choose_base='first',
#     choose_imperf='first',
#     output_fn='C:/temp/gnn-paper-data/train.lat',
#     return_df=False
# )
# assemble_catalogue(
#     num_base_lattices=500,
#     imperfection_levels=[0.0,0.01,0.02],
#     num_imperf_realisations=3,
#     input_dir='C:/temp/gnn-paper-data',
#     choose_base='first',
#     choose_imperf='last',
#     output_fn='C:/temp/gnn-paper-data/valid.lat',
#     return_df=False
# )
# %%
from data import GLAMM_rhotens_Dataset as GLAMM_Dataset
dataset = GLAMM_Dataset(
    root='C:/temp/gnn-paper-data/GLAMMDset',
    catalogue_path='C:/temp/gnn-paper-data/train.lat',
    dset_fname='train.pt',
    n_reldens=10,
)