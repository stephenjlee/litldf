import os, sys, json, hashlib, pickle
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from functools import reduce

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

import cv2
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point
import tensorflow as tf

keras = tf.keras
EarlyStopping = keras.callbacks.EarlyStopping

from litldf.utils.cache import cached_with_io
from litldf.data.config_data_paths import data_paths # these have been removed from the project due to sensitive information contained
from litldf.data.data_config import return_input_data_paths # these have been removed from the project due to sensitive information contained

jpgs_map_dict = {}

@cached_with_io
def load_bldgs(path):
    gdf = gpd.read_file(path)
    return gdf

@cached_with_io
def load_meters(path):
    gdf = gpd.read_file(path)
    return gdf


@cached_with_io
def get_gdf(gdf_path):
    gdf = gpd.read_file(gdf_path)
    return gdf

def set_jpgs_map(jpgs_map_path):
    global jpgs_map_dict
    jpgs_map_dict = json.load(open(jpgs_map_path))

def concatenate_gdfs(gdfs):
    # Step 1: Identify common columns across all geospatial dataframes
    common_columns = reduce(lambda x, y: x.intersection(y.columns), gdfs, gdfs[0].columns)

    # Step 2: Filter each dataframe to only include these common columns
    filtered_gdfs = [gdf[common_columns] for gdf in gdfs]

    # Step 3: Concatenate all the filtered dataframes
    combined_gdf = gpd.GeoDataFrame(pd.concat(filtered_gdfs, ignore_index=True))
    return combined_gdf


@cached_with_io
def load_data_mh(
        data_frac=1.0,
        keys=None,
        datasets='["RWA_2018_scrub", "KEN_CI_2014_scrub"]'
):
    scaler = StandardScaler()  # scale all auxiliary data values
    datasets = json.loads(datasets)

    data = {}
    for i, set_name in enumerate(['train', 'val', 'test']):

        bldgs_to_meters_paths = [data_paths[dataset][f'bldgs_to_meters_path_{set_name}'] for dataset in datasets]
        bldgs_to_meters_list = [json.load(open(bldgs_to_meters_path)) for bldgs_to_meters_path in bldgs_to_meters_paths]
        bldgs_to_meters = [element for sublist in bldgs_to_meters_list for element in sublist]

        meters_to_bldgs_paths = [data_paths[dataset][f'meters_to_bldgs_path_{set_name}'] for dataset in datasets]
        meters_to_bldgs_list = [json.load(open(meters_to_bldgs_path)) for meters_to_bldgs_path in meters_to_bldgs_paths]
        meters_to_bldgs = [element for sublist in meters_to_bldgs_list for element in sublist]

        bldgs_paths = [data_paths[dataset][f'bldgs_path_{set_name}'] for dataset in datasets]
        bldgs_gdf_list = [load_bldgs(bldgs_path) for bldgs_path in bldgs_paths]
        bldgs_gdf = concatenate_gdfs(bldgs_gdf_list)

        meters_paths = [data_paths[dataset][f'meters_path_{set_name}'] for dataset in datasets]
        meters_gdf_list = [load_meters(meters_path) for meters_path in meters_paths]
        meters_gdf_list_to_load = [meters_gdf.rename(columns={data_paths[dataset]['cons_col_name']: 'cons',
                                                              data_paths[dataset]['meter_id_name']: 'meter_id'})
                                   for (meters_gdf, dataset) in zip(meters_gdf_list, datasets)]
        meters_gdf = concatenate_gdfs(meters_gdf_list_to_load)

        # Calculate the number of elements to keep
        n_plates = len(meters_to_bldgs)
        n_plates_to_keep = int(n_plates * data_frac)

        # Generate a set of unique indices
        indices = np.linspace(0, n_plates - 1, num=n_plates_to_keep, dtype=int)
        np.random.shuffle(indices)

        # Select elements from the lists based on these indices
        bldgs_to_meters = [bldgs_to_meters[i] for i in indices]
        meters_to_bldgs = [meters_to_bldgs[i] for i in indices]

        # get bldgs and meters in order
        bldgs_order = np.concatenate([list(b2m.keys()) for b2m in bldgs_to_meters])
        meters_order = np.concatenate([list(m2b.keys()) for m2b in meters_to_bldgs])

        # log log useful flat lists
        bldgs = bldgs_order
        meters = meters_order

        # make sure that the buildings are in the correct order
        bldgs_order_df = pd.DataFrame({'correct_order': bldgs_order})

        # reorder buildings and only include matching buildings
        bldgs_gdf = pd.merge(bldgs_order_df,
                             bldgs_gdf,
                             left_on='correct_order',
                             right_on='origin_origin_id',
                             how='left')

        # make sure that the meters are in the correct order
        meters_order_df = pd.DataFrame({'correct_order': meters_order})
        meters_gdf = pd.merge(meters_order_df,
                              meters_gdf,
                              left_on='correct_order',
                              right_on='meter_id',
                              how='left')

        # add checks
        if not np.all(meters_gdf['meter_id'] == meters_order):
            raise Exception('graph data structure keys and meter numbers do not add up!')
        if not np.all(bldgs_gdf['origin_origin_id'] == bldgs_order):
            raise Exception('graph data structure keys and building ids do not add up!')

        n = meters_gdf['cons'].values[:, np.newaxis]
        x = bldgs_gdf[json.loads(keys)]

        x = x.fillna(0)

        # only fit the scaler on the training
        # set, so that it's consistent for
        # all train, test, val sets.
        if i == 0:
            scaler = scaler.fit(x)

        x = scaler.transform(x)
        x_fit = [x, bldgs_gdf['origin_origin_id'].values]

        # save general for set
        data[f'n_{set_name}'] = n
        data[f'x_fit_{set_name}'] = x_fit
        data[f'meters_to_bldgs_{set_name}'] = meters_to_bldgs
        data[f'bldgs_to_meters_{set_name}'] = bldgs_to_meters
        data[f'bldgs_order_{set_name}'] = bldgs_order
        data[f'meters_order_{set_name}'] = meters_order
        data[f'bldgs_{set_name}'] = bldgs
        data[f'meters_{set_name}'] = meters

    return data, scaler

@cached_with_io
def load_cons_gdf(cons_path, rows):
    cons_gdf = gpd.read_file(cons_path, rows=rows)
    return cons_gdf

def load_image_data(origin_origin_ids, jpgs_path):
    cache_dir_path = os.path.join(os.environ.get("PROJECT_CACHE"), 'load_image_data')
    os.makedirs(cache_dir_path, exist_ok=True)

    origin_origin_ids_hash = hashlib.sha256(json.dumps(origin_origin_ids.tolist()).encode('utf-8')).hexdigest()
    pickle_file_path = Path(os.path.join(cache_dir_path, origin_origin_ids_hash + '.p'))

    if not pickle_file_path.is_file():

        target_rows = 224
        target_cols = 224
        target_dims = 3

        images = []

        n_missing = 0
        N = origin_origin_ids.size
        for i, origin_origin_id in enumerate(origin_origin_ids):

            jpg_filename = f'{origin_origin_id}.jpg'

            jpg_path = ''
            if jpg_filename in jpgs_map_dict:
                jpg_path = jpgs_map_dict[f'{origin_origin_id}.jpg']

            if os.path.isfile(jpg_path):
                # if the file exists, load it!

                outputImage = cv2.imread(jpg_path)

                current_rows = outputImage.shape[0]
                current_cols = outputImage.shape[1]

            else:
                # if the file does not exist,

                print(f'file {jpg_path} does not exist. Using an image of zeros instead.'
                      f"({float(n_missing) / float(i + 1) * 100.:.2f}%) of images "
                      f"missing. {N} total bldgs.")

                outputImage = np.zeros((target_rows, target_cols, target_dims), dtype="uint8")

                current_rows = target_rows
                current_cols = target_cols

                n_missing = n_missing + 1

            diff_rows = current_rows - target_rows
            diff_cols = current_cols - target_cols

            if ((diff_rows < 0) or (diff_cols < 0)):
                # if there are any fewer than the necessary number of rows

                n_missing = n_missing + 1

                if diff_rows < 0:
                    print(
                        f"im row dim {current_rows} below min {target_rows} in "
                        f"path {jpg_path}. {n_missing} of {i}, "
                        f"({float(n_missing) / float(i + 1) * 100.:.2f}%) of images "
                        f"missing. {N} total bldgs.")
                if diff_cols < 0:
                    print(
                        f"im col dim {current_cols} below min {target_cols} in "
                        f"path {jpg_path}. {n_missing} of {i}, "
                        f"({float(n_missing) / float(i + 1) * 100.:.2f}%) of images "
                        f"missing. {N} total bldgs.")

                outputImage = np.zeros((target_rows, target_cols, target_dims), dtype="uint8")
                print(f"sending an image of zeros instead")

            elif ((diff_rows == 0) and (diff_cols == 0)):
                print('all good!')

            else:

                # cropping the output image, ensuring that it aligns in the center of the image
                outputImage = outputImage[int(np.ceil(float(diff_rows) / 2.)):int(-np.floor(float(diff_rows) / 2.)),
                              int(np.ceil(float(diff_cols) / 2.)):int(-np.floor(float(diff_cols) / 2.)), :]

            ## visualize image for debugging purposes
            # import matplotlib.pyplot as plt
            # vizimage = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)
            # plt.imshow(vizimage)
            # plt.show()
            images.append(outputImage)
        images = np.array(images)

        pickle.dump(images, open(pickle_file_path, "wb"), protocol=4)

    else:

        images = pickle.load(open(pickle_file_path, "rb"))

    return images


@cached_with_io
def load_data_xval(num_folds,
                   fold_num_val,
                   fold_num_test,
                   train_on_all,
                   dataset,
                   keys,
                   save_geojson,
                   rows,
                   use_jpgs,
                   rng_seed):
    cons_path, cons_key, run_path = return_input_data_paths(dataset)

    keys = json.loads(keys)

    if save_geojson:
        cons_gdf = load_cons_gdf(cons_path, rows)
    else:
        cons_gdf = gpd.read_file(cons_path, rows=rows, ignore_geometry=True)

    output_gdf = cons_gdf[(~cons_gdf[cons_key].isna()) & (cons_gdf[cons_key] > 0.5)].reset_index().copy()
    del cons_gdf

    s = np.round(output_gdf[cons_key].values / 12.)  # processing monthly consumption out of annualized values
    x_all = output_gdf[keys].values
    origin_origin_ids = output_gdf['origin_origin_id'].values
    x_mean = x_all.mean(axis=0)
    x_std = x_all.std(axis=0)
    x = (x_all - x_mean) / x_std
    N = s.size

    ############################################################################
    # BEGIN K FOLD CROSS VAL
    ############################################################################

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=rng_seed)

    # stratified k-fold stratifies based on y; integer arguments are treated as
    # multi-class classification.  Let's do something more intelligent.
    n_bins = 5
    bins = -np.ones(n_bins + 1)
    bins[1:] = np.logspace(np.log10(np.min(s[s > 0]) - 0.5),
                           np.log10(np.max(s[s > 0]) + 1), n_bins)

    fold_inds = [item[1][1] for item in enumerate(skf.split(x, np.digitize(s, bins=bins)))]

    val_inds = fold_inds[fold_num_val]
    test_inds = fold_inds[fold_num_test]
    train_inds = np.setdiff1d(np.concatenate(fold_inds), np.concatenate([val_inds, test_inds]))

    # get the training and testing sets for this fold
    x_train = x[train_inds, :]
    x_val = x[val_inds, :]
    x_test = x[test_inds, :]

    s_train = s[train_inds]
    s_val = s[val_inds]
    s_test = s[test_inds]

    if use_jpgs:

        origin_origin_ids_train = origin_origin_ids[train_inds]
        origin_origin_ids_val = origin_origin_ids[val_inds]
        origin_origin_ids_test = origin_origin_ids[test_inds]

        x_train_fit = [x_train, origin_origin_ids_train]
        x_val_fit = [x_val, origin_origin_ids_val]
        x_test_fit = [x_test, origin_origin_ids_test]
        x_fit = [x, origin_origin_ids]

    else:

        x_train_fit = x_train
        x_test_fit = x_test
        x_val_fit = x_val
        x_fit = x

    data = {
        'x': x,
        'origin_origin_ids': origin_origin_ids,
        'x_all': x_all,
        'x_mean': x_mean,
        'x_std': x_std,
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        's': s,
        's_train': s_train,
        's_val': s_val,
        's_test': s_test,
        'x_train_fit': x_train_fit,
        'x_val_fit': x_val_fit,
        'x_test_fit': x_test_fit,
        'x_fit': x_fit,
        'train_inds': train_inds,
        'val_inds': val_inds,
        'test_inds': test_inds
    }

    if train_on_all:
        data['x_train'] = x
        data['s_train'] = s
        data['x_train_fit'] = x_fit
        data['train_inds'] = np.sort(np.concatenate([train_inds, val_inds, test_inds]))

    return data, output_gdf, keys, run_path
