from utils import plotting
from data import Lattice, Catalogue
from data.lattice import WindowingException
import numpy as np
from tqdm import tqdm
from plotly.subplots import make_subplots


cat = Catalogue.from_file('./filtered_cat.lat', 0)
# tet_Z05.7_E2424
# names = ['cub_Z06.9_E14510']
# names = cat.names[7800:]
#### failed
# names = ['tet_Z06.5_E7537']
# names = ['cub_Z05.7_E12503']
# names = ['cub_Z05.1_E14181']
# names = ['cub_Z06.9_E11597']
####

for lat_data in cat:
    lat = Lattice(**lat_data)
    _ = lat.obtain_shift_vector()
    break