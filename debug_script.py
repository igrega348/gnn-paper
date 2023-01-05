from utils import plotting
from data import Lattice, Catalogue
from data.lattice import WindowingError
import numpy as np
from tqdm import tqdm
from plotly.subplots import make_subplots
from multiprocessing import Pool
from random import shuffle


# tet_Z05.7_E2424
# names = ['cub_Z06.9_E14510']
# names = cat.names[7800:]
#### failed
# names = ['tet_Z06.5_E7537']
# names = ['cub_Z05.7_E12503']
# names = ['cub_Z05.1_E14181']
# names = ['cub_Z06.9_E11597']
####

def process_lattice(lat_data):
    lat = Lattice(**lat_data)
    try:
        lat = lat.create_windowed()
        lat.calculate_fundamental_representation()
        return lat.to_dict()
    except WindowingError:
        return {}


def main():
    cat = Catalogue.from_file('./filt_wind.lat.lat', 0)
    print(cat)
    lat_data = [data for data in cat]
    shuffle(lat_data)

    with Pool(processes=6) as p:
        windowed = list(tqdm(p.imap_unordered(process_lattice, lat_data), total=len(lat_data), smoothing=0.1))
    
    selected = [data['name'] for data in windowed if 'name' in data]
    print(f'Keeping {len(selected)} lattices')


if __name__=='__main__':
    main()