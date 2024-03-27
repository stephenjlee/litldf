import os, sys, argparse, json, pickle, multiprocessing, random
import numpy as np
import geopandas as gpd

sys.path.append(os.environ.get("PROJECT_ROOT"))
sys.path.append(os.environ.get("LDF_ROOT"))
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

seed = 0
random.seed(seed)
np.random.seed(seed)
rng = np.random.default_rng()

def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Get shapefile in gridded form for country.')

    parser.add_argument('-cc', '--country_code',
                        default='RWA',
                        help='The ISO country code for the country being analyzed')
    parser.add_argument('-pps', '--par_pool_size', type=int, default=12,
                        help='parallel pool size for cuda processing')

    parser.add_argument('-dkp', '--dem_keys_path',
                        default=os.path.join(os.environ.get("PROJECT_DATA"),
                                             'trained_models', 'loc',
                                             'data_keys.json'), help='')
    parser.add_argument('-dsp', '--dem_scaler_path',
                        default=os.path.join(os.environ.get("PROJECT_DATA"),
                                             'trained_models', 'loc'
                                             'scaler.pkl'), help='')
    parser.add_argument('-dlp', '--dem_ldf_path',
                        default=os.path.join(os.environ.get("PROJECT_DATA"), 'loc'
                                             'trained_models'), help='')
    parser.add_argument('-cip', '--country_input_path',
                        default='', help='')
    parser.add_argument('-csp', '--country_shape_path',
                        default='', help='')
    parser.add_argument('-cop', '--country_output_path',
                        default='', help='')

    return parser.parse_args(args)


global_args = parse_args([])
global_args = global_args.__dict__

scaler_dem = pickle.load(open(global_args['dem_scaler_path'], 'rb'))
args_dem = json.load(open(os.path.join(global_args['dem_ldf_path'], 'args_all.json')))
data_keys_dem = json.loads(args_dem['keys'])

jpgs_path = os.path.join(os.environ.get("PROJECT_OUTEXT"), 'RWA-17')

def add_elecdem(tuple):
    (i,
     unique_file_id,
     country_input_path,
     country_shape_path,
     country_output_path,
     n_unique_file_ids,
     country_code) = tuple

    try:

        from ldf.models.model_ldf_scalar import LdfScalarModel
        from ldf.models.model_ldf_multimodal import LdfMultimodalModel

        if not args_dem['use_jpgs']:
            model_dem = LdfScalarModel(
                error_metric=args_dem['error_metric'],
                do_val=args_dem['do_val'],
                reg_mode=args_dem['reg_mode'],
                reg_val=args_dem['reg_val'],
                model_name=args_dem['model_name'],
                red_factor=args_dem['red_factor'],
                red_lr_on_plateau=args_dem['red_lr_on_plateau'],
                red_patience=args_dem['red_patience'],
                red_min_lr=args_dem['red_min_lr'],
                es_patience=args_dem['es_patience'],
                train_epochs=args_dem['train_epochs'],
                batch_size=args_dem['batch_size'],
                output_dir=r'/tmp',
                gpu=0,
                learning_rate_training=args_dem['learning_rate_training'],
                load_json=os.path.join(global_args['dem_ldf_path'], 'model_opt.json'),
                load_weights=os.path.join(global_args['dem_ldf_path'], 'model_opt.h5')
            )
            model_dem.setup_model(load_model=True)
        else:

            model_dem = LdfMultimodalModel(
                error_metric=args_dem['error_metric'],
                do_val=args_dem['do_val'],
                reg_mode=args_dem['reg_mode'],
                reg_val=args_dem['reg_val'],
                gpu=0,
                model_name=args_dem['model_name'],
                red_factor=args_dem['red_factor'],
                red_lr_on_plateau=args_dem['red_lr_on_plateau'],
                red_patience=args_dem['red_patience'],
                red_min_lr=args_dem['red_min_lr'],
                es_patience=args_dem['es_patience'],
                train_epochs=args_dem['train_epochs'],
                batch_size=args_dem['batch_size'],
                output_dir=r'/tmp',
                learning_rate_training=args_dem['learning_rate_training'],
                jpgs_path=jpgs_path,
                freeze_basemodel_layers=args_dem['freeze_basemodel_layers'],
                load_json=os.path.join(global_args['dem_ldf_path'], 'model_opt.json'), #
                load_weights=os.path.join(global_args['dem_ldf_path'], 'model_opt.h5') #
            )
            model_dem.setup_model(load_model=True)

        density_geoms_geojson_path = os.path.join(country_input_path, f"{unique_file_id}_geoms.geojson")

        bldgs_gdf = gpd.read_file(density_geoms_geojson_path)
        bldgs_gdf["origin_origin_id"] = bldgs_gdf["origin"] + '_' + bldgs_gdf["origin_id"]

        print('loaded bldgs')

        # run inference
        x_orig_dem = bldgs_gdf[data_keys_dem].values
        x_dem = scaler_dem.transform(x_orig_dem)

        if not args_dem['use_jpgs']:
            mean_preds_dem, \
                std_preds_dem, \
                preds_params_dem, \
                _ = \
                model_dem.predict(x_dem)
        else:
            mean_preds_dem, \
                std_preds_dem, \
                preds_params_dem, \
                _ = \
                model_dem.predict([x_dem, bldgs_gdf['origin_origin_id'].values])

        a_alls_dem = preds_params_dem['a_all']  # inclusive of priors
        b_alls_dem = preds_params_dem['b_all']  # inclusive of priors

        print(f'finished running demand for {unique_file_id}')


        # update and save buildings file
        bldgs_gdf['cons (kWh/month)'] = mean_preds_dem
        bldgs_gdf['std cons (kWh/month)'] = std_preds_dem
        bldgs_gdf['a_alls_dem'] = a_alls_dem
        bldgs_gdf['b_alls_dem'] = b_alls_dem

        try:
            bldgs_gdf.to_file(
                os.path.join(country_output_path, f"{unique_file_id}_geoms.geojson"),
                driver='GeoJSON')
            print(f'wrote files for grid cell: {unique_file_id}')

        except:
            print(f'Cannot write dataframe to file! It might be empty! Failed on grid cell: {unique_file_id}')
            with open(os.path.join(country_output_path, f"{unique_file_id}_geoms.err"), 'w') as f:
                f.write('Cannot write dataframe to file! It might be empty!')

    except:
        print(f'Something went wrong with {unique_file_id}. Debug for more info! \n')


def add_elecdem_by_grid_cell(country_input_path,
                             country_shape_path,
                             country_output_path,
                             par_pool_size,
                             country_code):
    all_files = np.array([file for file in os.listdir(country_input_path) if file.endswith("_geoms.geojson")])
    # density_geoms.geojson
    all_files_underscore_index = np.char.find(all_files, '_')
    all_files_ids = np.char.ljust(all_files, all_files_underscore_index)
    unique_file_ids = np.unique(all_files_ids)
    n_unique_file_ids = unique_file_ids.size

    par_list = []
    for i, unique_file_id in enumerate(unique_file_ids):

        elecdem_geoms_csv_path = os.path.join(country_output_path, f"{unique_file_id}_geoms.csv")
        elecdem_geoms_geojson_path = os.path.join(country_output_path, f"{unique_file_id}_geoms.geojson")

        # if the files already exist, skip it. i.e. only run if there's a missing file
        if not (os.path.exists(elecdem_geoms_csv_path) and os.path.exists(elecdem_geoms_geojson_path)):
            print(f'added {unique_file_id} to the list for generating density geometries.')
            par_list.append((i,
                             unique_file_id,
                             country_input_path,
                             country_shape_path,
                             country_output_path,
                             n_unique_file_ids,
                             country_code))

    with multiprocessing.Pool(par_pool_size) as p:
        print(p.map(add_elecdem, par_list, chunksize=1))
    # add_elecdem(par_list[0])


###################################################
def main(args):
    args = parse_args(args)
    args = args.__dict__

    os.makedirs(args['country_output_path'], exist_ok=True)

    print(f"running add_elecdem_by_grid_cell")
    add_elecdem_by_grid_cell(args['country_input_path'],
                             args['country_shape_path'],
                             args['country_output_path'],
                             args['par_pool_size'],
                             args['country_code'])

    print('done!')


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
