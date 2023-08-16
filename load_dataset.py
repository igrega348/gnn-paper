# %%
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s @ %(module)s: %(funcName)s :: %(message)s', level=logging.INFO)
import numpy as np
from data.datasets import assemble_catalogue
from data import Catalogue
# %%
if __name__=='__main__':
    with open('E:/best_base_names.txt', 'r') as f:
        best_base_names = f.readlines()
    choose_base = [name.strip() for name in best_base_names]
    # randomly select 7000 base lattices for training and remaining for validation
    train_base_names = np.random.choice(choose_base, size=7000, replace=False)
    val_base_names = list(set(choose_base) - set(train_base_names))
    print(f'Train base names {len(train_base_names)}')
    print(f'Validation base names {len(val_base_names)}')
   
    # df = assemble_catalogue(
    #     num_base_lattices=19000,
    #     imperfection_levels=[0.0,0.01,0.02,0.03,0.04,0.05,0.07,0.10],
    #     num_imperf_realisations=10,
    #     input_dir='E:/new_dset_0/',
    #     choose_base=train_base_names,
    #     choose_imperf='first',
    #     return_df=False,
    #     multiprocessing=8,
    #     output_fn=f'E:/new_dset_1/full_dset_{len(train_base_names)}_train.lat'
    # )
    
    # df1 = assemble_catalogue(
    #     num_base_lattices=19000,
    #     imperfection_levels=[0.0,0.01,0.02,0.03,0.04,0.05,0.07,0.10],
    #     num_imperf_realisations=10,
    #     input_dir='E:/new_dset_1/',
    #     choose_base=val_base_names,
    #     choose_imperf='first',
    #     return_df=False,
    #     multiprocessing=8,
    #     output_fn=f'E:/new_dset_1/full_dset_{len(val_base_names)}_val.lat'
    # )
    # df = df.append(df1)
    # load only the intersecting base lattices

    cat_0_lines = dict()
    for i in range(10):
        cat = Catalogue.from_file(f'E:/new_dset_0/cat_{i:02d}.lat', 0)
        print(cat)
        cat_lines = cat.lines
        # keep only the base lattices that are in the intersection
        cat_lines = {name:line for name, line in cat_lines.items() if '_'.join(name.split('_')[:3]) in choose_base}
        cat_0_lines.update(cat_lines)

    train_lines = {name:line for name, line in cat_0_lines.items() if '_'.join(name.split('_')[:3]) in train_base_names}
    val_lines = {name:line for name, line in cat_0_lines.items() if '_'.join(name.split('_')[:3]) in val_base_names}

    train_cat = Catalogue(data=train_lines, indexing=0)
    val_cat = Catalogue(data=val_lines, indexing=0)

    print('Saving catalogues')
    train_cat.to_file(f'E:/new_dset_0/full_dset_{len(train_base_names)}_train.lat')
    val_cat.to_file(f'E:/new_dset_0/full_dset_{len(val_base_names)}_val.lat')


    # %%
    # dset_name = 'full'
    # #
    # all_base_names = df['base_name'].unique()
    # print(f'All base names: {len(all_base_names)}')
    # np.random.shuffle(all_base_names)
    # train_base_names = all_base_names[:7000]
    # val_base_names = all_base_names[7000:]
    # print(f'Train base names {len(train_base_names)}')
    # print(f'Validation base names {len(val_base_names)}')

    # selected_df = df.loc[df['base_name'].isin(train_base_names), :]
    # selected_lat_dict = selected_df.to_dict('index')
    # new_cat = Catalogue.from_dict(selected_lat_dict)
    # new_cat.to_file(f'E:/{dset_name}_dset_{len(train_base_names)}_train.lat')

    # selected_df = df.loc[df['base_name'].isin(val_base_names), :]
    # selected_lat_dict = selected_df.to_dict('index')
    # new_cat = Catalogue.from_dict(selected_lat_dict)
    # new_cat.to_file(f'E:/{dset_name}_dset_{len(val_base_names)}_val.lat')