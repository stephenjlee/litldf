import os, sys, time, argparse, json, copy, pickle
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))
sys.path.append(os.environ.get("LDF_ROOT"))

import matplotlib.pyplot as plt
import numpy as np

import ldf.utils.utils_stat as us
import ldf.utils.utils_general as ug
from ldf.models.model_ldf_scalar import LdfScalarModel
from litldf.utils.cache import cached_with_io

import litldf.models.demand_allocation_mh as da
from litldf.visualization.plot_simplexes_for_synthetic_experiments import plot_simplex_for_a013, create_gif_with_name_match

np.random.seed(1)
rng = np.random.default_rng(1)


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run sampler to infer electrification status.')

    # GENERAL ARGS
    parser.add_argument('-od', '--output_dir',
                        default=os.path.join(os.environ.get("PROJECT_OUT"), 'a013'),
                        help='the output directory')
    parser.add_argument('-ss', '--save_samps', type=ug.str2bool, nargs='?',
                        const=True, default=True, help='whether to save samples.')
    parser.add_argument('-ps', '--pool_size', type=int, default=24, help='')
    parser.add_argument('-npp', '--n_postpred_to_plot', type=int, default=100,
                        help='number of posterior predictive plots.')

    # SIMULATION ARGS
    parser.add_argument('-tt', '--tx_true', default='sinegapo', help='options: sinegapo, sinebin')
    parser.add_argument('-mt', '--model_type', default='ldf', help='options: ldf, sineopt')

    # PGM ARGS
    parser.add_argument('-np', '--num_plates', type=int, default=2000, help='')
    parser.add_argument('-mep', '--mh_es_patience', type=int, default=200,
                        help='early stopping patience, default -1 (no early stopping)')
    parser.add_argument('-nit', '--n_mh_iters_for_train', type=int, default=20, help='')
    parser.add_argument('-nie', '--n_mh_iters_for_eval', type=int, default=10, help='')
    parser.add_argument('-nse', '--n_subsample_iters_for_eval', type=int, default=5, help='')
    parser.add_argument('-ti', '--nn_train_interval', type=int, default=5, help='')
    parser.add_argument('-am', '--alpha_m', type=float, default=1., help='')
    parser.add_argument('-at', '--alpha_t', type=float, default=1., help='')
    parser.add_argument('-ab', '--alpha_b', type=float, default=1., help='')
    parser.add_argument('-aai', '--a_all_init', type=float, default=0.0001, help='')
    parser.add_argument('-bai', '--b_all_init', type=float, default=0.0001, help='')

    # NN ARGS
    parser.add_argument('-lrt', '--learning_rate_training', type=float, default=0.0001,
                        help='the learning rate to apply during training')
    parser.add_argument('-tr', '--train_epochs', type=int, default=10,
                        help='the number epochs to train the nn on')
    parser.add_argument('-pms', '--params', default='{"a0": 0.0001, "b0": 0.0001}',
                        help='the param settings to use')
    parser.add_argument('-lm', '--load_model', type=ug.str2bool, nargs='?',
                        const=True, default=False, help='whether to load a saved model, or run from scratch.')
    parser.add_argument('-lj', '--load_json',
                        default=f'',
                        help='if load_model is True: model file to load')
    parser.add_argument('-lw', '--load_weights',
                        default=f'',
                        help='if load_weights is True: weights file to load. Note that this model needs to be '
                             'compatible with the rest of the args given.')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='1: use the GPU; 0: just use CPUs')
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help='batch size for stochastic gradient descent')
    parser.add_argument('-ep', '--es_patience', type=int, default=100,
                        help='early stopping patience, default -1 (no early stopping)')
    parser.add_argument('-nrr', '--n_rand_restarts', type=int, default=1,
                        help='number of random restarts for model training')
    parser.add_argument('-do', '--do_val', type=float, default=0.01,
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
                        help='the name of the eror metric set to use. Current options include: gapo, norm')
    parser.add_argument('-m', '--model_name', default='LARGER',
                        help='the name of the model to run. Current options include: SHALLOW, LARGER, XLARGE')

    return parser.parse_args(args)


class SineOpt:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.res = None
        self.amplitude = 500.
        self.periodicity_factor = 0.1
        self.shift_vert = 1000.
        self.const_b = 100.

    def fine_tune_model(self, X=None, y=None):

        from scipy.optimize import minimize, rosen, rosen_der

        def fun(x):
            amplitude = x[0]
            periodicity_factor = x[1]
            shift_vert = x[2]
            const_b = x[3]

            nll = -1. * us.gammapoisson_logpmf(
                y,
                amplitude * np.cos(periodicity_factor * X) + shift_vert,
                const_b
            )

            return np.mean(nll)

        init = copy.deepcopy((self.amplitude, self.periodicity_factor, self.shift_vert, self.const_b))

        self.res = minimize(fun, init, method='Nelder-Mead')

        print(self.res)

        if self.res.success:
            self.amplitude = self.res.x[0]
            self.periodicity_factor = self.res.x[1]
            self.shift_vert = self.res.x[2]
            self.const_b = self.res.x[3]

        else:
            print('fail')

        return self.res['fun']

    def predict(self, X=None):

        a_all = self.amplitude * np.cos(self.periodicity_factor * X.flatten()) + self.shift_vert
        n = a_all.size
        b_all = np.repeat(self.const_b, n)

        preds_params_temp = {
            'a_all': a_all,
            'b_all': b_all
        }

        return None, None, preds_params_temp, None

    def save_model(self, s=None):

        model_json = {
            'res_fun': self.res.fun,
            'amplitude': self.amplitude,
            'periodicity_factor': self.periodicity_factor,
            'shift_vert': self.shift_vert,
            'const_b': self.const_b
        }

        json_path = os.path.join(self.output_dir, f'sine_model_{s}.json')
        with open(json_path, 'w') as json_file:
            json_file.write(json.dumps(model_json))

        return json_path, None


def get_model(args, x, output_dir):
    model = None

    if args['model_type'] == 'sineopt':

        model = SineOpt(
            output_dir=output_dir
        )

    elif args['model_type'] == 'ldf':

        # setup ldf model
        model = LdfScalarModel(
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
            output_dir=output_dir,
            learning_rate_training=args['learning_rate_training'],
            load_json=args['load_json'],
            load_weights=args['load_weights']
        )
        model.setup_model(X=x,
                          load_model=args['load_model'])

    return model


class TxSineGapo:
    def __init__(self):
        self.amplitude = 800.
        self.periodicity_factor = 0.125
        self.shift_vert = 2000.
        self.const_b = 50.

    def y_secret_given_x(self, x):

        a_all_true = self.amplitude * np.cos(self.periodicity_factor * x) + self.shift_vert

        y_secret = us.gammapoisson_sample(
            a_all_true,
            self.const_b
        )
        return y_secret

    def plot(self, output_dir):

        x = np.linspace(0, 100, 101)

        a_all_true = self.amplitude * np.cos(self.periodicity_factor * x) + self.shift_vert
        b_all_true = np.repeat(self.const_b, repeats=x.size)
        mesh = []
        for a_elem, b_elem in zip(a_all_true, b_all_true):
            pmfs = []
            for y in x:
                pmfs.append(np.exp(us.gammapoisson_logpmf(y, a_elem, b_elem)))
            mesh.append(pmfs)
        mesh = np.array(mesh)
        plt.figure()
        plt.rcParams.update({'font.size': 10})
        plt.rcParams['axes.grid'] = False
        plt.imshow(mesh.T, cmap='viridis', origin='lower')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, 'heatmap_gapo_true.png'))
        plt.close('all')


class TxSineBin:
    def __init__(self):
        self.amplitude = 800.
        self.periodicity_factor = 0.125
        self.shift_vert = 2000.
        self.const_b = 100.

    def y_secret_given_x(self, x):

        a_all_true = np.cos(self.periodicity_factor * x)
        a_all_true = (a_all_true + 1.) / (
                2. * 1.5) + 1. / 6.  # normalize

        y_secret = us.bin_sample(self.const_b,
                                 a_all_true)

        return y_secret

    def plot(self, output_dir):

        x = np.linspace(0, 100, 101)

        a_all_true = np.cos(self.periodicity_factor * x)
        a_all_true = (a_all_true + 1.) / (
                2. * 1.5) + 1. / 6.  # normalize
        b_all_true = np.repeat(self.const_b, repeats=x.size)
        mesh = []
        for a_elem, b_elem in zip(a_all_true, b_all_true):
            pmfs = []
            for y in x:
                pmfs.append(np.exp(us.binom_logpmf(y, b_elem, a_elem)))
            mesh.append(pmfs)
        mesh = np.array(mesh)
        plt.figure()
        plt.rcParams.update({'font.size': 10})
        plt.rcParams['axes.grid'] = False
        plt.imshow(mesh.T, cmap='viridis', origin='lower')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, 'heatmap_bin_true.png'))
        plt.close('all')


def get_tx(tx_true):
    tx = None

    if tx_true == 'sinegapo':
        tx = TxSineGapo()

    if tx_true == 'sinebin':
        tx = TxSineBin()

    return tx


def y_secret_given_x(x, dirmult_args):
    # defines y_secret and plots the distribution

    tx = get_tx(dirmult_args['tx_true'])

    y_secret = tx.y_secret_given_x(x)

    return y_secret


def gen_samps_and_obs_inner(graph_data_inner, dirmult_args, output_dir):
    x = np.floor(np.random.uniform(10, 100, size=graph_data_inner['n_bldgs'])).astype(int)[:, np.newaxis]
    y_secret = y_secret_given_x(x, dirmult_args)

    m = []
    for meter in graph_data_inner['meters']:
        n_connected = len(graph_data_inner['meters_to_bldgs'][meter])
        m.append(np.repeat(1. / n_connected, repeats=n_connected))

    n = []
    # loop through each meter
    for key_temp in graph_data_inner['meters_to_bldgs'].keys():
        bldgs_set_temp = graph_data_inner['meters_to_bldgs'][key_temp]
        key_tot = []
        # loop through each set of buildings associated with a meter
        for bldg_temp in bldgs_set_temp:
            # get each building's consumption
            y_secret_temp = y_secret[graph_data_inner['bldgs'].index(bldg_temp)]
            # dividing building's consumption by the appropriate amount for
            # shared meters
            n_connected_meters = float(len(graph_data_inner['bldgs_to_meters'][bldg_temp]))
            y_secret_temp = (y_secret_temp / n_connected_meters).astype(int)
            key_tot.append(y_secret_temp)
        # sum up all consumption assigned to the meter
        key_tot = np.sum(key_tot)
        # append to values of n. One element of n for each meter m
        n.append(key_tot)
    n = np.array(n)

    # init y's
    y = []
    # loop through buildings. One value of y per building
    for bldg_temp in graph_data_inner['bldgs_to_meters'].keys():
        matching_meters = graph_data_inner['bldgs_to_meters'][bldg_temp]
        y_temp = []
        # loop through all meters connected to the building. For each meter, add
        # its contribution to the building's total consumption based on values of n and m
        for meter_temp in matching_meters:
            bldg_index_specific_to_meter = graph_data_inner['meters_to_bldgs'][meter_temp].index(bldg_temp)
            meter_index = graph_data_inner['meters'].index(meter_temp)
            y_temp.append(np.round(n[meter_index] * m[meter_index][bldg_index_specific_to_meter]).astype(int))
        y.append(y_temp)

    samps = {
        'm': m,
        'y': y,
        'a_all': np.repeat(dirmult_args['a_all_init'], repeats=graph_data_inner['n_bldgs']),
        'b_all': np.repeat(dirmult_args['b_all_init'], repeats=graph_data_inner['n_bldgs'])
    }
    obs = {
        'n': n,
        'x': x
    }

    return samps, obs


def gen_samps_and_obs_inner_viz(graph_data_inner, dirmult_args, output_dir):
    # x = np.floor(np.random.uniform(10, 100, size=graph_data_inner['n_bldgs'])).astype(int)[:, np.newaxis]
    x = np.array([[25], [50], [75]])

    y_secret = y_secret_given_x(x, dirmult_args)

    m = []
    for meter in graph_data_inner['meters']:
        n_connected = len(graph_data_inner['meters_to_bldgs'][meter])
        m.append(np.repeat(1. / n_connected, repeats=n_connected))

    n = []
    # loop through each meter
    for key_temp in graph_data_inner['meters_to_bldgs'].keys():
        bldgs_set_temp = graph_data_inner['meters_to_bldgs'][key_temp]
        key_tot = []
        # loop through each set of buildings associated with a meter
        for bldg_temp in bldgs_set_temp:
            # get each building's consumption
            y_secret_temp = y_secret[graph_data_inner['bldgs'].index(bldg_temp)]
            # dividing building's consumption by the appropriate amount for
            # shared meters
            n_connected_meters = float(len(graph_data_inner['bldgs_to_meters'][bldg_temp]))
            y_secret_temp = (y_secret_temp / n_connected_meters).astype(int)
            key_tot.append(y_secret_temp)
        # sum up all consumption assigned to the meter
        key_tot = np.sum(key_tot)
        # append to values of n. One element of n for each meter m
        n.append(key_tot)
    n = np.array(n)

    # init y's
    y = []
    # loop through buildings. One value of y per building
    for bldg_temp in graph_data_inner['bldgs_to_meters'].keys():
        matching_meters = graph_data_inner['bldgs_to_meters'][bldg_temp]
        y_temp = []
        # loop through all meters connected to the building. For each meter, add
        # its contribution to the building's total consumption based on values of n and m
        for meter_temp in matching_meters:
            bldg_index_specific_to_meter = graph_data_inner['meters_to_bldgs'][meter_temp].index(bldg_temp)
            meter_index = graph_data_inner['meters'].index(meter_temp)
            y_temp.append(np.round(n[meter_index] * m[meter_index][bldg_index_specific_to_meter]).astype(int))
        y.append(y_temp)

    samps = {
        'm': m,
        'y': y,
        'a_all': np.repeat(dirmult_args['a_all_init'], repeats=graph_data_inner['n_bldgs']),
        'b_all': np.repeat(dirmult_args['b_all_init'], repeats=graph_data_inner['n_bldgs'])
    }
    obs = {
        'n': n,
        'x': x
    }

    return samps, obs


def get_graph_data_structures(dirmult_args):
    # setup
    # expand bldgs_to_meters_template to k samples
    bldgs_to_meters = []
    for k in range(dirmult_args['K']):
        bldgs_to_meters_for_k = {}
        for key in dirmult_args['bldgs_to_meters_template'].keys():
            new_key = key + f'_{k}'
            new_list = [elem + f'_{k}' for elem in dirmult_args['bldgs_to_meters_template'][key]]
            bldgs_to_meters_for_k[new_key] = new_list
        bldgs_to_meters.append(bldgs_to_meters_for_k)

    # expand meters_to_bldgs_template to k samples
    meters_to_bldgs = []
    for k in range(dirmult_args['K']):
        meters_to_bldgs_for_k = {}
        for key in dirmult_args['meters_to_bldgs_template'].keys():
            new_key = key + f'_{k}'
            new_list = [elem + f'_{k}' for elem in dirmult_args['meters_to_bldgs_template'][key]]
            meters_to_bldgs_for_k[new_key] = new_list
        meters_to_bldgs.append(meters_to_bldgs_for_k)

    # collect into final graph data structure
    graph_data = []
    for bldgs_to_meters_for_k, meters_to_bldgs_for_k in zip(bldgs_to_meters, meters_to_bldgs):
        bldgs_for_k = list(bldgs_to_meters_for_k.keys())
        n_bldgs_for_k = len(bldgs_for_k)
        meters_for_k = list(meters_to_bldgs_for_k)

        graph_data.append({
            'bldgs_to_meters': bldgs_to_meters_for_k,
            'meters_to_bldgs': meters_to_bldgs_for_k,
            'bldgs': bldgs_for_k,
            'n_bldgs': n_bldgs_for_k,
            'meters': meters_for_k
        })

    return graph_data


def get_graph_data_structures_viz(dirmult_args, n_iters=1):
    # setup
    # expand bldgs_to_meters_template to k samples
    bldgs_to_meters = []
    for k in range(n_iters):
        bldgs_to_meters_for_k = {}
        for key in dirmult_args['bldgs_to_meters_template'].keys():
            new_key = key + f'_{k}'
            new_list = [elem + f'_{k}' for elem in dirmult_args['bldgs_to_meters_template'][key]]
            bldgs_to_meters_for_k[new_key] = new_list
        bldgs_to_meters.append(bldgs_to_meters_for_k)

    # expand meters_to_bldgs_template to k samples
    meters_to_bldgs = []
    for k in range(n_iters):
        meters_to_bldgs_for_k = {}
        for key in dirmult_args['meters_to_bldgs_template'].keys():
            new_key = key + f'_{k}'
            new_list = [elem + f'_{k}' for elem in dirmult_args['meters_to_bldgs_template'][key]]
            meters_to_bldgs_for_k[new_key] = new_list
        meters_to_bldgs.append(meters_to_bldgs_for_k)

    # collect into final graph data structure
    graph_data = []
    for bldgs_to_meters_for_k, meters_to_bldgs_for_k in zip(bldgs_to_meters, meters_to_bldgs):
        bldgs_for_k = list(bldgs_to_meters_for_k.keys())
        n_bldgs_for_k = len(bldgs_for_k)
        meters_for_k = list(meters_to_bldgs_for_k)

        graph_data.append({
            'bldgs_to_meters': bldgs_to_meters_for_k,
            'meters_to_bldgs': meters_to_bldgs_for_k,
            'bldgs': bldgs_for_k,
            'n_bldgs': n_bldgs_for_k,
            'meters': meters_for_k
        })

    return graph_data


def gen_samps_and_obs(graph_data, dirmult_args, output_dir):
    samps = []
    obs = []
    for graph_data_inner in graph_data:
        samps_inner, obs_inner = gen_samps_and_obs_inner(graph_data_inner, dirmult_args, output_dir)
        samps.append(samps_inner)
        obs.append(obs_inner)

    return samps, obs


def gen_samps_and_obs_viz(graph_data, dirmult_args, output_dir):
    samps = []
    obs = []
    for graph_data_inner in graph_data:
        samps_inner, obs_inner = gen_samps_and_obs_inner_viz(graph_data_inner, dirmult_args, output_dir)
        samps.append(samps_inner)
        obs.append(obs_inner)

    return samps, obs


def generate_data(dirmult_args, output_dir):
    # expand templates to k samples
    graph_data = \
        get_graph_data_structures(dirmult_args)

    # generate initial samples and observations
    samps, \
        obs = \
        gen_samps_and_obs(graph_data, dirmult_args, output_dir)

    # wrap samples, args, obs, and graph data into tuples, one for each plate
    samps_tuples = da.wrap_by_coloring_problem(dirmult_args, samps, obs, graph_data)

    x = np.concatenate([obs_inner['x'] for obs_inner in obs])

    return samps_tuples, x


def generate_data_viz(dirmult_args, output_dir):
    # expand templates to k samples
    graph_data = \
        get_graph_data_structures_viz(dirmult_args, n_iters=1)

    # generate initial samples and observations
    samps, \
        obs = \
        gen_samps_and_obs_viz(graph_data, dirmult_args, output_dir)

    n_samps_viz = 1000

    samps = [samps[0] for _ in range(n_samps_viz)]
    obs = [obs[0] for _ in range(n_samps_viz)]
    graph_data = \
        get_graph_data_structures_viz(dirmult_args, n_iters=n_samps_viz)

    # wrap samples, args, obs, and graph data into tuples, one for each plate
    samps_tuples = da.wrap_by_coloring_problem(dirmult_args, samps, obs, graph_data)

    x = np.concatenate([obs_inner['x'] for obs_inner in obs])

    return samps_tuples, x


def plot_heatmap(model, output_dir, s):
    x_eval = np.linspace(0, 100, 101)[:, np.newaxis]
    _, _, preds_params_temp, _ = model.predict(x_eval)
    a_all = preds_params_temp['a_all']
    b_all = preds_params_temp['b_all']
    x_eval = x_eval.flatten()
    mesh = []
    for a_plot, b_plot, x_plot in zip(a_all, b_all, x_eval):
        pmfs = []
        for y_plot in np.linspace(0, 100, 101):
            pmfs.append(np.exp(us.gammapoisson_logpmf(y_plot, a_plot, b_plot)))
        mesh.append(pmfs)
    mesh = np.array(mesh)
    plt.figure()
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['axes.grid'] = False
    plt.imshow(mesh.T, cmap='viridis', origin='lower')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, f'heatmap_empirical_{s}.png'))
    plt.close('all')

    return


# @cached_with_io
def demand_allocation_mh(args):
    # param configs
    dirmult_args = {
        'tx_true': args['tx_true'],
        'K': args['num_plates'],
        'n_mh_iters_for_train': args['n_mh_iters_for_train'],
        'n_mh_iters_for_eval': args['n_mh_iters_for_eval'],
        'alpha_m': args['alpha_m'],
        'alpha_t': args['alpha_t'],
        'alpha_b': args['alpha_b'],
        'a_all_init': args['a_all_init'],
        'b_all_init': args['b_all_init'],
        'meters_to_bldgs_template': {'a': ['1', '2'],
                                     'b': ['2', '3']},
        'bldgs_to_meters_template': {'1': ['a'],
                                     '2': ['a', 'b'],
                                     '3': ['b']}
    }

    # make output directory
    output_dir = os.path.join(args['output_dir'], f"{time.time()}_{args['tx_true']}_{args['model_type']}")
    os.makedirs(output_dir, exist_ok=True)

    # save args
    with open(os.path.join(output_dir, 'dirmult_args.json'), "w") as outfile:
        json.dump(dirmult_args, outfile)
    with open(os.path.join(output_dir, 'args.json'), "w") as outfile:
        json.dump(args, outfile)

    # plot true transformation
    tx = get_tx(dirmult_args['tx_true'])
    tx.plot(output_dir)

    # generate tuples for mcmc
    samps_tuples_train, x_train = generate_data(dirmult_args, output_dir)
    samps_tuples_val, x_val = generate_data(dirmult_args, output_dir)
    samps_tuples_test, x_test = generate_data(dirmult_args, output_dir)
    samps_tuples_viz_0, x_viz_0 = generate_data_viz(dirmult_args, output_dir)
    samps_tuples_viz_1, x_viz_1 = generate_data_viz(dirmult_args, output_dir)

    # init model
    model = get_model(args, x_train, output_dir)

    # init timer
    start_time = time.time()

    # #####################################
    # TRAINING PHASE
    # #####################################
    # init logging
    error_hist_keras = []
    model_paths_hist = []

    error_hist_train = []
    error_hist_val = []
    error_hist_test = []
    error_hist_viz_0 = []
    error_hist_viz_1 = []

    json_path, weights_path = None, None
    es_model_iter = 0
    es_model_train_loss = None
    es_model_val_loss = None
    es_model_test_loss = None
    es_model_viz_loss_0 = None
    es_model_viz_loss_1 = None

    samps_tuples_train_opt = None
    samps_tuples_val_opt = None
    samps_tuples_test_opt = None
    samps_tuples_viz_opt_0 = None
    samps_tuples_viz_opt_1 = None

    samps_train, samps_val, samps_test, samps_viz_0, samps_viz_1 = None, None, None, None, None
    samps_tuples_hist_viz_0 = []
    samps_tuples_hist_viz_1 = []

    # run sampler, training phase
    for s in range(dirmult_args['n_mh_iters_for_train']):

        label = f"mh train iter {s} of {dirmult_args['n_mh_iters_for_train']}"

        samps_tuples_train, samps_train = \
            da.sample_all(samps_tuples_train, args['pool_size'], label)
        samps_tuples_val, samps_val = \
            da.sample_all(samps_tuples_val, args['pool_size'], label)
        samps_tuples_test, samps_test = \
            da.sample_all(samps_tuples_test, args['pool_size'], label)

        samps_tuples_viz_0, samps_viz_0 = \
            da.sample_all(samps_tuples_viz_0, args['pool_size'], label)
        samps_tuples_viz_1, samps_viz_1 = \
            da.sample_all(samps_tuples_viz_1, args['pool_size'], label)

        samps_tuples_hist_viz_0.append(copy.deepcopy(samps_tuples_viz_0))
        samps_tuples_hist_viz_1.append(copy.deepcopy(samps_tuples_viz_1))

        if s % args['nn_train_interval'] == 0:
            print(f'---------s: {s}---------')

            # unravel y's
            y_unraveled_train = da.get_y_unraveled(samps_train)

            # train model and get new params
            hist = model.fine_tune_model(X=x_train,
                                         y=y_unraveled_train)

            # update_preds_params
            samps_tuples_train = da.update_preds_params(model, samps_tuples_train, x_train)
            samps_tuples_val = da.update_preds_params(model, samps_tuples_val, x_val)
            samps_tuples_test = da.update_preds_params(model, samps_tuples_test, x_test)
            samps_tuples_viz_0 = da.update_preds_params(model, samps_tuples_viz_0, x_viz_0)
            samps_tuples_viz_1 = da.update_preds_params(model, samps_tuples_viz_1, x_viz_1)

            # plot_heatmap
            plot_heatmap(model, output_dir, s)

            # log to keras
            error_hist_keras.append(hist)

        # compute nlls
        nll_train, _ = da.get_nll_from_samps(samps_train)
        nll_val, _ = da.get_nll_from_samps(samps_val)
        nll_test, _ = da.get_nll_from_samps(samps_test)
        nll_viz_0, _ = da.get_nll_from_samps(samps_viz_0)
        nll_viz_1, _ = da.get_nll_from_samps(samps_viz_1)

        # if val loss is the lowest we've seen yet,
        if ((es_model_val_loss is None) or (nll_val < es_model_val_loss)) and (s > 20):
            json_path, weights_path = model.save_model('opt')
            es_model_iter = s
            es_model_train_loss = nll_train
            es_model_val_loss = nll_val
            es_model_test_loss = nll_test
            es_model_viz_loss_0 = nll_viz_0
            es_model_viz_loss_1 = nll_viz_1

            samps_tuples_train_opt = copy.deepcopy(samps_tuples_train)
            samps_tuples_val_opt = copy.deepcopy(samps_tuples_val)
            samps_tuples_test_opt = copy.deepcopy(samps_tuples_test)
            samps_tuples_viz_opt_0 = copy.deepcopy(samps_tuples_viz_0)
            samps_tuples_viz_opt_1 = copy.deepcopy(samps_tuples_viz_1)

        # log to hist
        error_hist_train.append(nll_train)
        error_hist_val.append(nll_val)
        error_hist_test.append(nll_test)
        error_hist_viz_0.append(nll_viz_0)
        error_hist_viz_1.append(nll_viz_1)

        # early stopping on mh
        if (((s - es_model_iter) > args['mh_es_patience']) and
                (args['mh_es_patience'] > -1)):
            print('early stopping on mh')
            break

    print(
        f"demand_allocation_mh --- {(time.time() - start_time)} ---")

    # #####################################
    # EVALUATION PHASE
    # #####################################
    # init
    samps_eval_train = []
    samps_eval_val = []
    samps_eval_test = []
    samps_eval_viz_0 = []
    samps_eval_viz_1 = []
    if es_model_iter > 0:  # if we saved samples from early stopping, load them to replace the latest samples
        samps_tuples_train = samps_tuples_train_opt
        samps_tuples_val = samps_tuples_val_opt
        samps_tuples_test = samps_tuples_test_opt
        samps_tuples_viz_0 = samps_tuples_viz_opt_0
        samps_tuples_viz_1 = samps_tuples_viz_opt_1

    # run sampler, evaluation phase
    for s in range(dirmult_args['n_mh_iters_for_eval']):

        label = f"mh eval iter {s} of {dirmult_args['n_mh_iters_for_eval']}"

        samps_tuples_train, samps_train = \
            da.sample_all(samps_tuples_train, args['pool_size'], label)
        samps_tuples_val, samps_val = \
            da.sample_all(samps_tuples_val, args['pool_size'], label)
        samps_tuples_test, samps_test = \
            da.sample_all(samps_tuples_test, args['pool_size'], label)
        samps_tuples_viz_0, samps_viz_0 = \
            da.sample_all(samps_tuples_viz_0, args['pool_size'], label)
        samps_tuples_viz_1, samps_viz_1 = \
            da.sample_all(samps_tuples_viz_1, args['pool_size'], label)

        if (s % args['n_subsample_iters_for_eval']) == 0:
            # log to samps
            samps_eval_train.append(samps_train)
            samps_eval_val.append(samps_val)
            samps_eval_test.append(samps_test)
            samps_eval_viz_0.append(samps_viz_0)
            samps_eval_viz_1.append(samps_viz_1)

    _, nll_train_by_bldg = da.get_nll_from_samps(samps_train)
    _, nll_val_by_bldg = da.get_nll_from_samps(samps_val)
    _, nll_test_by_bldg = da.get_nll_from_samps(samps_test)
    _, nll_viz_by_bldg_0 = da.get_nll_from_samps(samps_viz_0)
    _, nll_viz_by_bldg_1 = da.get_nll_from_samps(samps_viz_1)

    # #####################################
    # PLOTTING PHASE
    # #####################################
    da.plot_learning_curves(error_hist_train,
                            error_hist_val,
                            error_hist_test,
                            output_dir,
                            -1)

    da.plot_learning_curves_keras(error_hist_keras,
                                  output_dir,
                                  args['model_type'])

    plot_marginals(samps_eval_train,
                   output_dir)

    if args['save_samps']:
        # save all samples
        results = {'samps_eval_train': samps_eval_train,
                   'samps_eval_val': samps_eval_val,
                   'samps_eval_test': samps_eval_test,
                   'samps_eval_viz_0': samps_eval_viz_0,
                   'samps_eval_viz_1': samps_eval_viz_1,
                   'samps_tuples_train': samps_tuples_train,
                   'samps_tuples_val': samps_tuples_val,
                   'samps_tuples_test': samps_tuples_test,
                   'samps_tuples_viz_0': samps_tuples_viz_0,
                   'samps_tuples_viz_1': samps_tuples_viz_1,
                   'error_hist_keras': error_hist_keras,
                   'error_hist_train': error_hist_train,
                   'error_hist_val': error_hist_val,
                   'error_hist_test': error_hist_test,
                   'error_hist_viz_0': error_hist_viz_0,
                   'error_hist_viz_1': error_hist_viz_1,
                   'model_paths': (json_path, weights_path),
                   'es_model_train_loss': es_model_train_loss,
                   'es_model_val_loss': es_model_val_loss,
                   'es_model_test_loss': es_model_test_loss,
                   'es_model_viz_loss_0': es_model_viz_loss_0,
                   'es_model_viz_loss_1': es_model_viz_loss_1,
                   'es_model_sample_iter': es_model_iter,
                   'samps_tuples_hist_viz_0': samps_tuples_hist_viz_0,
                   'samps_tuples_hist_viz_1': samps_tuples_hist_viz_1,
                   'model_paths_hist': model_paths_hist,
                   'output_dir': output_dir}
        pickle.dump(results, open(os.path.join(results['output_dir'], 'results.p'), "wb"))

        # save train test val nll as a json
    if es_model_iter > 0:
        results_json = {
            'train': es_model_train_loss,
            'val': es_model_val_loss,
            'test': es_model_test_loss,
            'viz_0': es_model_viz_loss_0,
            'viz_1': es_model_viz_loss_1
        }
    else:
        results_json = {
            'train': results['error_hist_train'][-1],
            'val': results['error_hist_val'][-1],
            'test': results['error_hist_test'][-1],
            'viz_0': results['error_hist_viz_1'][-1],
            'viz_1': results['error_hist_viz_0'][-1]
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
                          n_postpred_to_plot=args['n_postpred_to_plot'],
                          fold='train')
    da.save_error_metrics(output_dir,
                          samps_tuples_val,
                          samps_eval_val,
                          nll_val_by_bldg,
                          n_postpred_to_plot=args['n_postpred_to_plot'],
                          fold='val')
    da.save_error_metrics(output_dir,
                          samps_tuples_test,
                          samps_eval_test,
                          nll_test_by_bldg,
                          n_postpred_to_plot=args['n_postpred_to_plot'],
                          fold='test')
    da.save_error_metrics(output_dir,
                          samps_tuples_viz_0,
                          samps_eval_viz_0,
                          nll_viz_by_bldg_0,
                          n_postpred_to_plot=args['n_postpred_to_plot'],
                          fold='viz_0')
    da.save_error_metrics(output_dir,
                          samps_tuples_viz_1,
                          samps_eval_viz_1,
                          nll_viz_by_bldg_1,
                          n_postpred_to_plot=args['n_postpred_to_plot'],
                          fold='viz_1')

    return results


def plot_marginals(samps_eval_train, output_dir):

    y_temp_hist = np.array([np.sum(samps[0]['y'][0]) for samps in samps_eval_train])
    y_0_2_hist = np.array([np.sum(samps[0]['y'][1]) for samps in samps_eval_train])
    y_0_3_hist = np.array([np.sum(samps[0]['y'][2]) for samps in samps_eval_train])
    m_0_A_hist = np.array([samps[0]['m'][0][0] for samps in samps_eval_train])
    m_0_B_hist = np.array([samps[0]['m'][1][0] for samps in samps_eval_train])

    # visualize
    bin_array = list(range(101))

    # marginal histogram
    plt.figure()
    plt.hist(y_temp_hist, bins=bin_array, density=True)
    plt.title('y_temp_hist mh')
    plt.savefig(os.path.join(output_dir, 'y_temp_hist.png'))
    plt.close('all')

    # marginal histogram
    plt.figure()
    plt.hist(y_0_2_hist, bins=bin_array, density=True)
    plt.title('y_0_2_hist mh')
    plt.savefig(os.path.join(output_dir, 'y_0_2_hist.png'))
    plt.close('all')

    # marginal histogram
    plt.figure()
    plt.hist(y_0_3_hist, bins=bin_array, density=True)
    plt.title('y_0_3_hist mh')
    plt.savefig(os.path.join(output_dir, 'y_0_3_hist.png'))
    plt.close('all')

    plt.figure()
    plt.plot(list(range(len(y_temp_hist))), y_temp_hist)
    plt.title('y_temp_hist_v_time')
    plt.savefig(os.path.join(output_dir, 'y_temp_hist_v_time.png'))
    plt.close('all')

    plt.figure()
    plt.plot(list(range(len(y_0_2_hist))), y_0_2_hist)
    plt.title('y_0_2_hist_v_time')
    plt.savefig(os.path.join(output_dir, 'y_0_2_hist_v_time.png'))
    plt.close('all')

    plt.figure()
    plt.plot(list(range(len(y_0_3_hist))), y_0_3_hist)
    plt.title('y_0_3_hist_v_time')
    plt.savefig(os.path.join(output_dir, 'y_0_3_hist_v_time.png'))
    plt.close('all')

    bin_array_0_1 = np.linspace(0., 1., 101)

    # marginal histogram
    plt.figure()
    plt.hist(m_0_A_hist, bins=bin_array_0_1, density=True)
    plt.title('m_0_A_hist mh')
    plt.savefig(os.path.join(output_dir, 'm_0_A_hist.png'))
    plt.close('all')

    # marginal histogram
    plt.figure()
    plt.hist(m_0_B_hist, bins=bin_array_0_1, density=True)
    plt.title('m_0_B_hist mh')
    plt.savefig(os.path.join(output_dir, 'm_0_B_hist.png'))
    plt.close('all')


def run_with_args_subset(args_subset):
    args = parse_args([])
    args = args.__dict__

    for args_elem in args_subset.keys():
        args[args_elem] = args_subset[args_elem]

    results = \
        demand_allocation_mh(args)


def main(args):
    args = parse_args(args)
    args = args.__dict__

    results = \
        demand_allocation_mh(args)

    samps_tuples_hist_viz_0 = results['samps_tuples_hist_viz_0'][1:]
    samps_tuples_hist_viz_1 = results['samps_tuples_hist_viz_1'][1:]
    # iter = 0

    for iter in range(len(samps_tuples_hist_viz_0)):

        print(f'{iter} of {len(samps_tuples_hist_viz_0)}')

        a1, a2, a3 = samps_tuples_hist_viz_0[iter][0][1]['a_all']
        b1, b2, b3 = samps_tuples_hist_viz_0[iter][0][1]['b_all']

        n_A, n_B = samps_tuples_hist_viz_0[iter][0][2]['n']
        n_A_2, n_B_2 = samps_tuples_hist_viz_1[iter][0][2]['n']

        y_samps_0 = np.array([[np.sum(ys) for ys in samps[1]['y']] for samps in samps_tuples_hist_viz_0[iter]])
        y_samps_1 = np.array([[np.sum(ys) for ys in samps[1]['y']] for samps in samps_tuples_hist_viz_1[iter]])

        for azim in [-50, -30, 120, 140]:
            # only plotting feasible region
            plot_simplex_for_a013(azim=azim,
                                  n_A=n_A,
                                  n_B=n_B,
                                  n_A_2=n_A_2,
                                  n_B_2=n_B_2,
                                  a1=a1,
                                  a2=a2,
                                  a3=a3,
                                  b1=b1,
                                  b2=b2,
                                  b3=b3,
                                  y_samps_0=y_samps_0,
                                  y_samps_1=y_samps_1,
                                  output_folder_path=results['output_dir'],
                                  fig_prefix=iter,
                                  subfolder=str(azim))

        # only plotting feasible region
        plot_simplex_for_a013(azim=iter % 360,
                              n_A=n_A,
                              n_B=n_B,
                              n_A_2=n_A_2,
                              n_B_2=n_B_2,
                              a1=a1,
                              a2=a2,
                              a3=a3,
                              b1=b1,
                              b2=b2,
                              b3=b3,
                              y_samps_0=y_samps_0,
                              y_samps_1=y_samps_1,
                              output_folder_path=results['output_dir'],
                              fig_prefix=iter,
                              subfolder='rotate')

    for subfolder in ['-50', '-30', '120', '140', 'rotate']:
        input_folder_path = os.path.join(results['output_dir'], subfolder)
        output_gif_path = os.path.join(input_folder_path, 'animation.gif')
        try:
            create_gif_with_name_match(subfolder,
                                       input_folder_path,
                                       output_gif_path,
                                       font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
        except:
            print('issues with create_gif_with_name_match()')


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
