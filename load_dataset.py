# %%
# from data import GLAMM_rhotens_Dataset as GLAMM_Dataset
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
from data.datasets import assemble_catalogue

base_name_tup = assemble_catalogue(
    num_base_lattices=10,
    imperfection_levels=[0.0],
    num_imperf_realisations=8,
    input_dir='C:/temp/gnn-paper-data',
    output_fn='',
    choose_base='random'
)
print(base_name_tup)