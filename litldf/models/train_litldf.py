import os, sys, time, argparse, json, pickle, copy, hashlib
import multiprocessing

print(multiprocessing.cpu_count())
import numpy as np
import random

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))
sys.path.append(os.environ.get("LDF_ROOT"))

seed = 0
random.seed(seed)
np.random.seed(seed)
rng = np.random.default_rng()

from ldf.models.model_ldf_multimodal import LdfMultimodalModel
from ldf.models.model_ldf_scalar import LdfScalarModel
from litldf.data.load_data import load_data_mh, set_jpgs_map

import ldf.utils.utils_general as ug
import litldf.models.demand_allocation_mh as da


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run sampler to infer electrification status.')

    # GENERAL ARGS
    parser.add_argument('-od', '--output_dir',
                        default=os.path.join(os.environ.get("PROJECT_OUT"), 'train_litldf'),
                        help='the output directory')
    parser.add_argument('-ss', '--save_samps', type=ug.str2bool, nargs='?',
                        const=True, default=True, help='whether to save samples.')
    parser.add_argument('-ps', '--pool_size', type=int, default=20, help='')
    parser.add_argument('-npp', '--n_postpred_to_plot', type=int, default=100,
                        help='number of posterior predictive plots.')
    parser.add_argument('-sgj', '--save_geojson', default=False,
                        help='Whether (if True) to save a geojson, or (if False) just a csv. Saving a geojson requires significantly more memory.')

    # DATA ARGS
    parser.add_argument('-df', '--data_frac',
                        default=1.0,
                        type=float,
                        help='')
    parser.add_argument('-k', '--keys',
                        default='["area_in_meters", "n_bldgs_1km_away", "lulc2017_crops_N11", "lulc2017_built_area_N1", "lulc2017_rangeland_N1", "lulc2017_crops_N1", "lulc2017_built_area_N11", "lulc2017_rangeland_N11", "lulc2017_crops_N51", "lulc2017_built_area_N51", "lulc2017_rangeland_N51", "ntl2018_N1", "ntl2018_N11", "ntl2018_N51", "ookla_fixed_20200101_avg_d_kbps", "ookla_fixed_20200101_devices", "ookla_mobile_20200101_avg_d_kbps", "ookla_mobile_20200101_devices"]',
                        help='What keys to use from geojsons')
    parser.add_argument('-uj', '--use_jpgs', type=bool, default=False,
                        help='whether or not to use images when running the model')
    parser.add_argument('-jp', '--jpgs_map_path', default=os.path.join(os.environ.get("PROJECT_OUT"),
                                                                       'jpg_maps',
                                                                       'RWA-17_KEN-CI-17.json'),
                        help='the path to the jpegs maps folder that tells you where different jpegs are')

    parser.add_argument('-ds', '--datasets', default='["RWA_2018_scrub_2", "KEN_CI_2014_scrub_3"]',
                        help='the name of the dataset to run. See data_config.py for options')

    # PGM ARGS
    parser.add_argument('-mep', '--mh_es_patience', type=int, default=50,
                        help='early stopping patience, default -1 (no early stopping)')
    parser.add_argument('-nmi', '--n_mh_iters_for_train', type=int, default=2000, help='')
    parser.add_argument('-nie', '--n_mh_iters_for_eval', type=int, default=100, help='')
    parser.add_argument('-nse', '--n_subsample_iters_for_eval', type=int, default=5, help='')
    parser.add_argument('-ti', '--nn_train_interval', type=int, default=10, help='')
    parser.add_argument('-am', '--alpha_m', type=float, default=1.0, help='')
    parser.add_argument('-at', '--alpha_t', type=float, default=1., help='')
    parser.add_argument('-ab', '--alpha_b', type=float, default=1., help='')
    parser.add_argument('-aai', '--a_all_init', type=float, default=33.3, help='')
    parser.add_argument('-bai', '--b_all_init', type=float, default=1., help='')

    # NN ARGS
    parser.add_argument('-mt', '--model_type',
                        default='ldf_scalar',
                        help='options: ldf_scalar, ldf_multi_input')
    parser.add_argument('-lrt', '--learning_rate_training', type=float, default=0.001,
                        help='the learning rate to apply during training')
    parser.add_argument('-tr', '--train_epochs', type=int, default=1,
                        help='the number epochs to train the nn on')
    parser.add_argument('-pms', '--params', default='{"a0": 0.0001, "b0": 0.0001}',
                        help='the param settings to use')
    parser.add_argument('-lm', '--load_model', type=ug.str2bool, nargs='?',
                        const=True, default=True, help='whether to load a saved model, or run from scratch.')
    parser.add_argument('-lj', '--load_json',
                        default=f'',
                        help='if load_model is True: model file to load')
    parser.add_argument('-lw', '--load_weights',
                        default=f'',
                        help='if load_weights is True: weights file to load. Note that this model needs to be '
                             'compatible with the rest of the args given.')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='1: use the GPU; 0: just use CPUs')
    parser.add_argument('-bs', '--batch_size', type=int, default=512,
                        help='batch size for stochastic gradient descent')
    parser.add_argument('-ep', '--es_patience', type=int, default=100,
                        help='early stopping patience, default -1 (no early stopping)')
    parser.add_argument('-nrr', '--n_rand_restarts', type=int, default=1,
                        help='number of random restarts for model training')
    parser.add_argument('-do', '--do_val', type=float, default=0.02,
                        help='the dropout value to employ at the hidden layers')
    parser.add_argument('-rm', '--reg_mode', default='L2',
                        help='the regularization mode to employ. Available options are L2 and L1')
    parser.add_argument('-rv', '--reg_val', type=float, default=0.01,
                        help='the regularization value to employ with the regularization mode specified by --reg_mode')

    parser.add_argument('-rlr', '--red_lr_on_plateau', type=bool, default=False,
                        help='whether or not to use ReduceLROnPlateau')
    parser.add_argument('-rflr', '--red_factor', type=float, default=0.5,
                        help='ReduceLROnPlateau factor')
    parser.add_argument('-rplr', '--red_patience', type=float, default=5,
                        help='ReduceLROnPlateau patience')
    parser.add_argument('-rmlr', '--red_min_lr', type=float, default=0.0000000001,
                        help='ReduceLROnPlateau minimum learning rate')
    parser.add_argument('-em', '--error_metric', default='gapo',
                        help='the name of the eror metric set to use. Current options include: gapo')
    parser.add_argument('-m', '--model_name', default='SHALLOW',
                        help='the name of the model to run. Current options include: '
                             'SHALLOW, '
                             'LARGER, '
                             'XLARGE, '
                             'LARGER_RESNET')
    parser.add_argument('-fbl', '--freeze_basemodel_layers', type=bool, default=True,
                        help='Whether or not to freeze basemodel layers when using '
                             'a pre-trained convnet model')

    return parser.parse_args(args)


def update_data(args, data, data_keys):
    if args['model_type'] == 'ldf_scalar':
        data[data_keys['x_fit_train']] = np.array(data[data_keys['x_fit_train']][0])
        data[data_keys['x_fit_val']] = np.array(data[data_keys['x_fit_val']][0])
        data[data_keys['x_fit_test']] = np.array(data[data_keys['x_fit_test']][0])

    return data


def get_model(args, data, data_keys, hash):
    model = None

    if args['model_type'] == 'ldf_scalar':

        # setup ldf model
        model = LdfScalarModel(
            hash=hash,
            error_metric=args['error_metric'],
            do_val=args['do_val'],
            reg_mode=args['reg_mode'],
            reg_val=args['reg_val'],
            model_name=args['model_name'],
            red_factor=args['red_factor'],
            red_lr_on_plateau=args['red_lr_on_plateau'],
            red_patience=args['red_patience'],
            red_min_lr=args['red_min_lr'],
            es_patience=args['es_patience'],
            train_epochs=args['train_epochs'],
            batch_size=args['batch_size'],
            output_dir=args['output_dir'],
            learning_rate_training=args['learning_rate_training'],
            load_json=args['load_json'],
            load_weights=args['load_weights']
        )
        model.setup_model(X=data[data_keys['x_fit_train']],
                          load_model=args['load_model'])


    elif args['model_type'] == 'ldf_multi_input':

        if args['use_jpgs']:
            set_jpgs_map(args['jpgs_map_path'])
            print('asdf')

        # setup ldf model
        model = LdfMultimodalModel(
            hash=hash,
            save_geojson=args['save_geojson'],
            error_metric=args['error_metric'],
            do_val=args['do_val'],
            reg_mode=args['reg_mode'],
            reg_val=args['reg_val'],
            gpu=args['gpu'],
            model_name=args['model_name'],
            red_factor=args['red_factor'],
            red_lr_on_plateau=args['red_lr_on_plateau'],
            red_patience=args['red_patience'],
            red_min_lr=args['red_min_lr'],
            es_patience=args['es_patience'],
            train_epochs=args['train_epochs'],
            batch_size=args['batch_size'],
            output_dir=args['output_dir'],
            learning_rate_training=args['learning_rate_training'],
            jpgs_map_path=args['jpgs_map_path'],
            freeze_basemodel_layers=args['freeze_basemodel_layers'],
            load_json=args['load_json'],
            load_weights=args['load_weights']
        )
        model.setup_model(X=data[data_keys['x_fit_train']],
                          load_model=args['load_model'])

    return model


def get_data_keys():
    data_keys = {}
    for set_name in ['train', 'val', 'test']:
        data_keys[f'bldgs_to_meters_{set_name}'] = f'bldgs_to_meters_{set_name}'
        data_keys[f'meters_to_bldgs_{set_name}'] = f'meters_to_bldgs_{set_name}'
        data_keys[f'meters_{set_name}'] = f'meters_{set_name}'
        data_keys[f'n_{set_name}'] = f'n_{set_name}'
        data_keys[f'x_fit_{set_name}'] = f'x_fit_{set_name}'

    return data_keys


# @cached_with_io
def demand_allocation_mh(args):
    # Create a hashlib object and update it with the dictionary string
    hash_object = hashlib.md5(str(args).encode())

    # Get the hexadecimal representation of the hash and take the first 7 characters
    hash = hash_object.hexdigest()[:7]

    # param configs
    dirmult_args = {
        'n_mh_iters_for_train': args['n_mh_iters_for_train'],
        'n_mh_iters_for_eval': args['n_mh_iters_for_eval'],
        'alpha_m': args['alpha_m'],
        'alpha_t': args['alpha_t'],
        'alpha_b': args['alpha_b'],
        'a_all_init': args['a_all_init'],
        'b_all_init': args['b_all_init'],
    }

    # load data
    data, scaler = load_data_mh(
        data_frac=args['data_frac'],
        keys=args['keys'],
        datasets=args['datasets'])

    data_keys = get_data_keys()

    start_time = time.time()

    samps_tuples_train = da.generate_data(dirmult_args,
                                          data[data_keys['bldgs_to_meters_train']],
                                          data[data_keys['meters_to_bldgs_train']],
                                          data[data_keys['meters_train']],
                                          data[data_keys['n_train']]
                                          )
    samps_tuples_val = da.generate_data(dirmult_args,
                                        data[data_keys['bldgs_to_meters_val']],
                                        data[data_keys['meters_to_bldgs_val']],
                                        data[data_keys['meters_val']],
                                        data[data_keys['n_val']]
                                        )
    samps_tuples_test = da.generate_data(dirmult_args,
                                         data['bldgs_to_meters_test'],
                                         data['meters_to_bldgs_test'],
                                         data['meters_test'],
                                         data['n_test']
                                         )

    os.makedirs(args['output_dir'], exist_ok=True)

    data = update_data(args, data, data_keys)
    model = get_model(args, data, data_keys, hash)

    output_dir = model.get_output_dir()

    # save scaler and data keys
    pickle.dump(scaler, open(os.path.join(output_dir, 'scaler.pkl'), 'wb'))
    with open(os.path.join(output_dir, 'data_keys.json'), "w") as outfile:
        json.dump(data_keys, outfile)

    # save all args
    with open(os.path.join(output_dir, 'args_all.json'), "w") as outfile:
        json.dump(args, outfile)

    # #####################################
    # TRAINING PHASE
    # #####################################
    # init
    error_hist_keras = []
    error_hist_train = []
    error_hist_val = []
    error_hist_test = []
    json_path, weights_path = None, None
    es_model_iter = 0
    es_model_train_loss = None
    es_model_val_loss = None
    es_model_test_loss = None
    samps_tuples_train_opt = None
    samps_tuples_val_opt = None
    samps_tuples_test_opt = None
    samps_train, samps_val, samps_test = None, None, None
    # run sampler
    for s in range(dirmult_args['n_mh_iters_for_train']):

        label = f"mh train iter {s} of {dirmult_args['n_mh_iters_for_train']}"

        samps_tuples_train, samps_train = \
            da.sample_all(samps_tuples_train, args['pool_size'], label)
        samps_tuples_val, samps_val = \
            da.sample_all(samps_tuples_val, args['pool_size'], label)
        samps_tuples_test, samps_test = \
            da.sample_all(samps_tuples_test, args['pool_size'], label)

        if s % args['nn_train_interval'] == 0:
            print(f'---------s: {s}---------')

            # unravel y's
            y_unraveled_train = da.get_y_unraveled(samps_train)

            # train model and get new params
            hist = model.fine_tune_model(X=data[data_keys['x_fit_train']],
                                         y=y_unraveled_train,
                                         batch_size=args['batch_size'],
                                         use_jpgs=args['use_jpgs'],
                                         train_epochs=args['train_epochs'],
                                         use_multiprocessing=True,
                                         break_out_val=False)

            # update_preds_params
            samps_tuples_train = da.update_preds_params(model, samps_tuples_train, data[data_keys['x_fit_train']])
            samps_tuples_val = da.update_preds_params(model, samps_tuples_val, data[data_keys['x_fit_val']])
            samps_tuples_test = da.update_preds_params(model, samps_tuples_test, data['x_fit_test'])

            # log to keras
            error_hist_keras.append(hist)

        # compute nlls
        nll_train, _ = da.get_nll_from_samps(samps_train)
        nll_val, _ = da.get_nll_from_samps(samps_val)
        nll_test, _ = da.get_nll_from_samps(samps_test)

        # if val loss is the lowest we've seen yet,
        if ((es_model_val_loss is None) or (nll_val < es_model_val_loss)) and (s > 20):
            json_path, weights_path = model.save_model('opt')
            es_model_iter = s
            es_model_train_loss = nll_train
            es_model_val_loss = nll_val
            es_model_test_loss = nll_test
            samps_tuples_train_opt = copy.deepcopy(samps_tuples_train)
            samps_tuples_val_opt = copy.deepcopy(samps_tuples_val)
            samps_tuples_test_opt = copy.deepcopy(samps_tuples_test)

        # log to hist
        error_hist_train.append(nll_train)
        error_hist_val.append(nll_val)
        error_hist_test.append(nll_test)

        # early stopping on mh
        if (((s - es_model_iter) > args['mh_es_patience']) and
                (args['mh_es_patience'] > -1)):
            print('early stopping on mh')
            break

    # #####################################
    # EVALUATION PHASE
    # #####################################
    # init
    samps_eval_train = []
    samps_eval_val = []
    samps_eval_test = []
    if es_model_iter > 0:  # if we saved samples from early stopping, load them to replace the latest samples
        samps_tuples_train = samps_tuples_train_opt
        samps_tuples_val = samps_tuples_val_opt
        samps_tuples_test = samps_tuples_test_opt
    # run sampler, evaluation phase
    for s in range(dirmult_args['n_mh_iters_for_eval']):

        label = f"mh eval iter {s} of {dirmult_args['n_mh_iters_for_eval']}"

        samps_tuples_train, samps_train = \
            da.sample_all(samps_tuples_train, args['pool_size'], label)
        samps_tuples_val, samps_val = \
            da.sample_all(samps_tuples_val, args['pool_size'], label)
        samps_tuples_test, samps_test = \
            da.sample_all(samps_tuples_test, args['pool_size'], label)

        if (s % args['n_subsample_iters_for_eval']) == 0:
            # log to samps
            samps_eval_train.append(samps_train)
            samps_eval_val.append(samps_val)
            samps_eval_test.append(samps_test)

    _, nll_train_by_bldg = da.get_nll_from_samps(samps_train)
    _, nll_val_by_bldg = da.get_nll_from_samps(samps_val)
    _, nll_test_by_bldg = da.get_nll_from_samps(samps_test)

    # #####################################
    # PLOTTING PHASE
    # #####################################
    da.plot_learning_curves(error_hist_train,
                            error_hist_val,
                            error_hist_test,
                            output_dir,
                            es_model_iter)

    da.plot_learning_curves_keras(error_hist_keras,
                                  output_dir,
                                  'ldf')

    # #####################################
    # SAVE RESULTS PHASE
    # #####################################
    if args['save_samps']:
        results = {'samps_eval_train': samps_eval_train,
                   'samps_eval_val': samps_eval_val,
                   'samps_eval_test': samps_eval_test,
                   'samps_tuples_train': samps_tuples_train,
                   'samps_tuples_val': samps_tuples_val,
                   'samps_tuples_test': samps_tuples_test,
                   'error_hist_keras': error_hist_keras,
                   'error_hist_train': error_hist_train,
                   'error_hist_val': error_hist_val,
                   'error_hist_test': error_hist_test,
                   'model_paths': (json_path, weights_path),
                   'es_model_train_loss': es_model_train_loss,
                   'es_model_val_loss': es_model_val_loss,
                   'es_model_test_loss': es_model_test_loss,
                   'es_model_sample_iter': es_model_iter,
                   'output_dir': output_dir}
        pickle.dump(results, open(os.path.join(results['output_dir'], 'results.p'), "wb"))
    # save train test val nll as a json
    if es_model_iter > 0:
        results_json = {
            'train': es_model_train_loss,
            'val': es_model_val_loss,
            'test': es_model_test_loss
        }
    else:
        results_json = {
            'train': results['error_hist_train'][-1],
            'val': results['error_hist_val'][-1],
            'test': results['error_hist_test'][-1]
        }
    results_json['time'] = time.time() - start_time
    with open(os.path.join(results['output_dir'], 'results.json'), "w") as outfile:
        json.dump(results_json, outfile)

    # #####################################
    # EVALUATION PHASE
    # #####################################
    # save error metrics
    da.save_error_metrics(output_dir,
                          samps_tuples_train,
                          samps_eval_train,
                          nll_train_by_bldg,
                          n_postpred_to_plot=-1,
                          fold='train')
    da.save_error_metrics(output_dir,
                          samps_tuples_val,
                          samps_eval_val,
                          nll_val_by_bldg,
                          n_postpred_to_plot=-1,
                          fold='val')
    da.save_error_metrics(output_dir,
                          samps_tuples_test,
                          samps_eval_test,
                          nll_test_by_bldg,
                          n_postpred_to_plot=args['n_postpred_to_plot'],
                          fold='test')

    return results


def main(args):
    args = parse_args(args)
    args = args.__dict__

    results = \
        demand_allocation_mh(args)


def run_with_args_subset(args_subset):
    args = parse_args([])
    args = args.__dict__

    for args_elem in args_subset.keys():
        args[args_elem] = args_subset[args_elem]

    results = \
        demand_allocation_mh(args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
