# %%
from data import Lattice, Catalogue
from data.lattice import WindowingError
from tqdm import tqdm
from multiprocessing import Pool
from random import shuffle
from typing import Tuple
# %%
def try_window(data: dict) -> Tuple:
    lat = Lattice(**data)
    try:
        _, num_attempts = lat.obtain_shift_vector(max_num_attempts=3, return_attempts=True)
    except WindowingError:
        num_attempts = float('inf')

    return (lat.name, num_attempts)

def create_fundamental(lat_data: dict) -> dict:
    lat = Lattice(**lat_data)
    try:
        newlat = lat.create_windowed()
        newlat.calculate_fundamental_representation()
        return newlat.to_dict()
    except WindowingError:
        return {}

def main():
    cat = Catalogue.from_file('./filtered_cat.lat', 0)
    print(cat)
    lat_data = [data for data in cat]*10
    shuffle(lat_data)
    with Pool(processes=6) as p:
        attempts = list(tqdm(p.imap_unordered(try_window, lat_data), total=len(lat_data), smoothing=0.1))

    discarded = [tup[0] for tup in attempts if tup[1]>3]

    selected = list(set(cat.names) - set(discarded))
    print(f'Keeping {len(selected)} lattices')
    
    # check
    lat_data = [data for data in cat if data['name'] in selected]
    with Pool(processes=6) as p:
        windowed = list(tqdm(p.imap(create_fundamental, lat_data), total=len(lat_data), smoothing=0.1))

    
    data_dict = {data['name']:data for data in windowed if 'name' in data}
    print(f'Keeping {len(data_dict)} lattices')

    cat = Catalogue.from_dict(data_dict)
    cat.to_file('./filt_wind.lat')

if __name__=='__main__':
    main()