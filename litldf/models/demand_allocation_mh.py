import os, sys, json, time, copy
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))
sys.path.append(os.environ.get("LDF_ROOT"))

from multiprocessing import Pool
import numpy as np
import scipy

from matplotlib import pyplot as plt
import ldf.utils.utils_stat as us

rng = np.random.default_rng()


def proposal_logpdf(
        samps_left,
        samps_right,
):
    dirichlet_m_logdfs = []
    for m_left, m_right in zip(samps_left['m'], samps_right['m']):

        if np.any(m_right == 0.):
            m_right = (m_right + np.finfo(np.float32).eps)

        if np.any(m_left == 0.):
            m_left = (m_left + np.finfo(np.float32).eps)

        m_right = m_right / np.sum(m_right)
        m_left = m_left / np.sum(m_left)
        m_logpdfs = scipy.stats.dirichlet.logpdf(m_left, 100. * m_right)

        dirichlet_m_logdfs.append(m_logpdfs)

    return np.sum(dirichlet_m_logdfs)


def ys_ms_given_others_proportional_logpmf(
        samps,
        dirmult_args
):
    beta_m_logdfs = []
    for m in samps['m']:
        if np.any(m <= 0.):
            m = (m + np.finfo(np.float32).eps)
        m = m / np.sum(m)
        m_logpdfs = scipy.stats.dirichlet.logpdf(m, np.repeat(dirmult_args['alpha_m'], repeats=m.size))

        beta_m_logdfs.append(m_logpdfs)

    gapo_y_logpmfs = []
    for y, a_all, b_all in zip(samps['y'], samps['a_all'], samps['b_all']):
        y_sum = np.sum(y)
        y_logpmf = us.gammapoisson_logpmf(y_sum, a_all, b_all)
        gapo_y_logpmfs.append(y_logpmf)

    return np.sum(beta_m_logdfs) + np.sum(gapo_y_logpmfs)


def generate_proposals_all(samps,
                           obs,
                           graph_data):
    samps_proposed = copy.deepcopy(samps)

    # sample m
    for i, m in enumerate(samps['m']):
        if np.any(m <= 0.):
            m = (m + np.finfo(np.float32).eps)
        m = m / np.sum(m)
        samps_proposed['m'][i] = np.random.dirichlet(100. * m)

    # sample ys loop through each meter. We'll be
    # sampling from the multinomial at the meter-level!
    for i, meter_temp in enumerate(graph_data['meters']):

        n_temp = obs['n'][i]
        m_temp = samps_proposed['m'][i]
        y_samps = np.random.multinomial(n_temp, m_temp)

        # now we need to find matching buildings, and for each one,
        # update the lower-level y value
        matching_bldgs = graph_data['meters_to_bldgs'][meter_temp]
        for matching_bldg, y_samp in zip(matching_bldgs, y_samps):
            matching_bldg_index = graph_data['bldgs'].index(matching_bldg)

            meter_list_relative_to_bldg = graph_data['bldgs_to_meters'][matching_bldg]
            meter_index_relative_to_bldg = meter_list_relative_to_bldg.index(meter_temp)

            samps_proposed['y'][matching_bldg_index][meter_index_relative_to_bldg] = y_samp

    return samps_proposed


def sample_ys_ms(tuple):
    rand_seed = int(os.getpid() * 1000. + int(str(int(time.time() * 10000.))[-7:]))
    np.random.seed(rand_seed)

    dirmult_args, samps, obs, graph_data = tuple

    samps_proposed = \
        generate_proposals_all(samps,
                               obs,
                               graph_data)

    numer_logpmf = ys_ms_given_others_proportional_logpmf(
        samps_proposed,
        dirmult_args
    )

    numer_prop_logpdf = proposal_logpdf(samps, samps_proposed)

    denom_logpmf = ys_ms_given_others_proportional_logpmf(
        samps,
        dirmult_args
    )

    denom_prop_logpdf = proposal_logpdf(samps_proposed, samps)

    log_r = numer_logpmf + numer_prop_logpdf - denom_logpmf - denom_prop_logpdf

    log_u = np.log(np.random.uniform(low=0.0, high=1.0))

    if log_u < log_r:
        samps = copy.deepcopy(samps_proposed)
        accept = True
    else:
        accept = False

    return samps, accept


def wrap_by_coloring_problem(dirmult_args, samps, obs, graph_data):
    # wrap samples, args, obs, and graph data into tuples, one for each plate
    samps_tuples = []
    for samps_inner, obs_inner, graph_data_inner in zip(samps, obs, graph_data):
        samps_tuples.append([copy.deepcopy(dirmult_args),
                             copy.deepcopy(samps_inner),
                             copy.deepcopy(obs_inner),
                             copy.deepcopy(graph_data_inner)])

    return samps_tuples


def get_y_unraveled(samps):
    # unravel y's
    y_unraveled = []
    for samps_inner in samps:
        for y_temp in samps_inner['y']:
            y_unraveled.append(float(np.sum(y_temp)))
    y_unraveled = np.array(y_unraveled)[:, np.newaxis]

    return y_unraveled


def sample_all(samps_tuples, pool_size, label):
    print('running parpool')
    with Pool(pool_size) as p:
        samps = p.map(sample_ys_ms, samps_tuples)
    print('ending parpool')

    accepts = [samp[1] for samp in samps]
    samps = [samp[0] for samp in samps]
    print(f'{label}: acceptance ratio: {np.mean(accepts)}')

    for i, samp in enumerate(samps):
        samps_tuples[i][1] = copy.deepcopy(samp)

    return samps_tuples, samps


def update_preds_params(model, samps_tuples, data_x_fit):

    _, _, preds_params_temp, _ = model.predict(data_x_fit)
    a_alls = preds_params_temp['a_all']
    b_alls = preds_params_temp['b_all']

    # roll params back into samps for next iters
    assign_ind = 0
    for j, samps_tuple in enumerate(samps_tuples):
        for k in range(samps_tuples[j][1]['a_all'].size):
            samps_tuples[j][1]['a_all'][k] = a_alls[assign_ind]
            samps_tuples[j][1]['b_all'][k] = b_alls[assign_ind]
            assign_ind = assign_ind + 1

    return samps_tuples


def get_nll_from_samps(samps):
    y_by_iter = []
    a_all_iter = []
    b_all_iter = []
    # loop through subgraphs
    for samp_by_subgraph in samps:

        y_by_subgraph = []
        # loop through individual lists of y, corresponding to indiviudal buildings
        for y_temp in samp_by_subgraph['y']:
            y_by_subgraph.append(np.sum(y_temp))

        y_by_iter.append(y_by_subgraph)
        a_all_iter.append(samp_by_subgraph['a_all'])
        b_all_iter.append(samp_by_subgraph['b_all'])

    y_by_iter = np.concatenate(y_by_iter)
    a_all_iter = np.concatenate(a_all_iter)
    b_all_iter = np.concatenate(b_all_iter)

    nlls = -1. * us.gammapoisson_logpmf(y_by_iter, a_all_iter, b_all_iter)
    nll = np.mean(nlls)

    return nll, nlls


def plot_learning_curves(error_hist_train,
                         error_hist_val,
                         error_hist_test,
                         output_dir,
                         es_model_iter):
    plt.figure()
    x_temp = list(range(len(error_hist_train)))
    plt.plot(x_temp, error_hist_train, label='Training')
    plt.plot(x_temp, error_hist_val, label='Validation')
    plt.plot(x_temp, error_hist_test, label='Test')
    if es_model_iter > 0:
        plt.axvline(x=es_model_iter, color='m', label='Early Stopping Iteration')
    plt.xlabel('Sampling iteration')
    plt.ylabel('Mean NLL')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close('all')


def plot_learning_curves_keras(error_hist_keras,
                               output_dir,
                               model_type):
    if len(error_hist_keras) > 0:

        if model_type == 'ldf':
            error_hist_keras_loss = np.concatenate([error_hist['loss'] for error_hist in error_hist_keras])
        else:
            error_hist_keras_loss = error_hist_keras

        plt.figure()
        x_temp_keras = list(range(len(error_hist_keras_loss)))
        plt.plot(x_temp_keras, error_hist_keras_loss, label='Training')
        plt.xlabel('Keras iteration')
        plt.ylabel('Mean NLL')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'learning_curves_keras.png'))
        plt.close('all')


def plot_posterior_predictive(output_dir,
                              empiric_mean,
                              y_temp_hist,
                              a_all,
                              b_all,
                              bldg_name,
                              plot=True):
    # empiric_mean = 10

    in_95_cr = False
    ppf_0250 = us.gammapoisson_ppf(0.0250, a_all, b_all)
    ppf_9750 = us.gammapoisson_ppf(0.9750, a_all, b_all)

    if (empiric_mean >= ppf_0250) and (empiric_mean <= ppf_9750):
        in_95_cr = True

    if plot:

        # plt.figure()
        # plt.hist(y_temp_hist)
        # plt.show()

        print(f'plotting post pred for {bldg_name}')

        ppf_0001 = us.gammapoisson_ppf(0.0001, a_all, b_all)
        ppf_9999 = us.gammapoisson_ppf(0.9999, a_all, b_all)

        min_domain = np.min([ppf_0001, empiric_mean])
        max_domain = np.max([ppf_9999, empiric_mean])

        bin_array_left = np.linspace(min_domain - 2, ppf_0250, int(ppf_0250 - min_domain + 1)).tolist()
        bin_array_middle = np.linspace(ppf_0250, ppf_9750, int(ppf_9750 - ppf_0250 + 1)).tolist()
        bin_array_right = np.linspace(ppf_9750, max_domain + 2, int(max_domain + 2 - ppf_9750 + 1)).tolist()

        # # marginal histogram
        # plt.figure()
        # plt.hist(y_temp_hist)

        plt.figure()

        pmfs_left = []
        for x in bin_array_left:
            pmfs_left.append(np.exp(us.gammapoisson_logpmf(x, a_all, b_all)))
        pmfs_middle = []
        for x in bin_array_middle:
            pmfs_middle.append(np.exp(us.gammapoisson_logpmf(x, a_all, b_all)))
        pmfs_right = []
        for x in bin_array_right:
            pmfs_right.append(np.exp(us.gammapoisson_logpmf(x, a_all, b_all)))

        # Convert the lists to numpy arrays for vectorized operations
        bin_array_left = np.array(bin_array_left)
        bin_array_middle = np.array(bin_array_middle)
        bin_array_right = np.array(bin_array_right)

        # Create a new set of x values for the fill function, offset by half a step.
        delta_x_left = (bin_array_left[1] - bin_array_left[0]) / 2 if len(bin_array_left) > 1 else 0
        delta_x_middle = (bin_array_middle[1] - bin_array_middle[0]) / 2 if len(bin_array_middle) > 1 else 0
        delta_x_right = (bin_array_right[1] - bin_array_right[0]) / 2 if len(bin_array_right) > 1 else 0

        fill_x_left = bin_array_left + delta_x_left
        fill_x_middle = bin_array_middle + delta_x_middle
        fill_x_right = bin_array_right + delta_x_right

        plt.fill_between(fill_x_left, pmfs_left, alpha=0.2, color='red', step='pre',
                         label='$p(y | x;\phi)$, outside 95% CI')
        plt.fill_between(fill_x_middle, pmfs_middle, alpha=0.2, color='green', step='pre',
                         label='$p(y | x;\phi)$, inside 95% CI')
        plt.fill_between(fill_x_right, pmfs_right, alpha=0.2, color='red', step='pre')

        plt.step(bin_array_left, pmfs_left, alpha=0.6, color='gray', where='mid')
        plt.step(bin_array_middle, pmfs_middle, alpha=0.6, color='gray', where='mid')
        plt.step(bin_array_right, pmfs_right, alpha=0.6, color='gray', where='mid')

        if (empiric_mean >= ppf_0250) and (empiric_mean <= ppf_9750):
            plt.axvline(x=empiric_mean, color='green', linewidth=3, label='$E[y | x,n;\phi]$')
        else:
            plt.axvline(x=empiric_mean, color='red', linewidth=3, label='$E[y | x,n;\phi]$')

        # Define the bin edges for the histogram
        bins = np.arange(y_temp_hist.min(), y_temp_hist.max() + 2) - 0.5

        # Adding histogram as a step plot with color fill and border
        plt.hist(y_temp_hist, bins=bins, density=True, histtype='stepfilled', edgecolor='gray',
                 facecolor='skyblue', alpha=0.4, label='$p(y | x,n;\phi)$')

        plt.legend()
        plt.xlabel(f'y (building id: {bldg_name})')
        plt.ylabel('probability mass')
        # plt.show()

        plt.savefig(os.path.join(output_dir, f'{bldg_name}_with_hist.png'))
        plt.close('all')

        print(f'saved plot for {bldg_name}')

    return in_95_cr


def plot_cdf_histogram(output_dir, cdfs):
    print('plotting cdf histogram')

    # plot cdfs
    plt.figure()
    plt.hist(cdfs, bins=50)
    plt.xlabel('$P(Y<E[y|x,n;\phi]|x;\phi)$')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f'cdfs.png'))
    plt.close('all')

def plot_cdf_sampled_histogram(output_dir, cdfs_sampled):
    print('plotting cdf sampled histogram')

    # plot cdfs_sampled
    plt.figure()
    plt.hist(cdfs_sampled, bins=50)
    plt.xlabel('$P(Y=t|x;\phi), t \sim P(y|x,n;\phi)$')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f'cdfs_sampled.png'))
    plt.close('all')

def save_error_metrics(output_dir,
                       samps_tuples,
                       samps_eval,
                       nll_by_bldg,
                       n_postpred_to_plot=-1,
                       fold=None):
    print('running save_error_metrics')

    # metrics to log
    cdfs = []
    cdfs_sampled = []

    means = []
    preds = []
    in_95_crs = []
    from sklearn.metrics import mean_squared_error

    sums_estimated_bldg_cons_on_plate = []
    sums_ns_on_plate = []
    bldg_count_for_plotting = 0

    # loop through every plate
    for plate_i, tuple_plate in enumerate(samps_tuples):
        graph_data_plate = tuple_plate[3]
        obs_plate = tuple_plate[2]

        # loop through every building on each plate for the
        # purpose of summing all estimated building-level consumption
        sum_estimated_bldgs_on_plate = []
        for bldg_i, bldg in enumerate(graph_data_plate['bldgs']):
            y_temp_a_all = np.unique([np.sum(samps[plate_i]['a_all'][bldg_i]) for samps in samps_eval])
            y_temp_b_all = np.unique([np.sum(samps[plate_i]['b_all'][bldg_i]) for samps in samps_eval])
            sum_estimated_bldgs_on_plate.append(us.gammapoisson_mean(y_temp_a_all, y_temp_b_all))
        sum_estimated_bldg_cons_on_plate = np.sum(sum_estimated_bldgs_on_plate)
        sum_ns_on_plate = np.sum(obs_plate['n'])
        sums_estimated_bldg_cons_on_plate.append(sum_estimated_bldg_cons_on_plate)
        sums_ns_on_plate.append(sum_ns_on_plate)

        # loop through every building on each plate for the
        # purpose of doing building-level error analysis
        for bldg_i, bldg in enumerate(graph_data_plate['bldgs']):

            bldg_count_for_plotting = bldg_count_for_plotting + 1

            y_temp_hist = np.array([np.sum(samps[plate_i]['y'][bldg_i]) for samps in samps_eval])
            y_temp_a_all = np.unique([np.sum(samps[plate_i]['a_all'][bldg_i]) for samps in samps_eval])
            y_temp_b_all = np.unique([np.sum(samps[plate_i]['b_all'][bldg_i]) for samps in samps_eval])
            if (y_temp_a_all.size > 1) or (y_temp_b_all.size > 1):
                raise Exception(f'a_all size ({y_temp_a_all.size}) or b_all size ({y_temp_b_all.size}) > 0')
            y_temp_a_all = y_temp_a_all[0]
            y_temp_b_all = y_temp_b_all[0]

            mean_temp = np.round(np.mean(y_temp_hist))
            pred = us.gammapoisson_mean(y_temp_a_all, y_temp_b_all)

            post_pred_output_dir = os.path.join(output_dir, 'postpred')
            os.makedirs(post_pred_output_dir, exist_ok=True)

            if bldg_count_for_plotting < n_postpred_to_plot:
                plot_postpred = True
            else:
                plot_postpred = False

            # plot posterior predictive
            in_95_cr = \
                plot_posterior_predictive(post_pred_output_dir,
                                          mean_temp,
                                          y_temp_hist,
                                          y_temp_a_all,
                                          y_temp_b_all,
                                          bldg,
                                          plot=plot_postpred)



        cdfs.append(us.gammapoisson_cdf(mean_temp, y_temp_a_all, y_temp_b_all))
        for y_temps in y_temp_hist:
            cdfs_sampled.append(us.gammapoisson_cdf(y_temps, y_temp_a_all, y_temp_b_all))

        means.append(mean_temp)
        preds.append(pred)
        in_95_crs.append(in_95_cr)

    means = np.array(means)
    preds = np.array(preds)
    errors = means - preds

    sums_estimated_bldg_cons_on_plate = np.array(sums_estimated_bldg_cons_on_plate)
    sums_ns_on_plate = np.array(sums_ns_on_plate)

    # plot cdf histgram
    plot_cdf_histogram(output_dir, cdfs)
    plot_cdf_sampled_histogram(output_dir, cdfs_sampled)

    #####################################

    # plot_cdf_histogram_sampled(cdfs_sampled)
    # save jsons
    error_metrics_dict = {
        f'means_{fold}': means.tolist(),
        f'preds_{fold}': preds.tolist(),
        f'errors_{fold}': errors.tolist(),
        f'nll_by_bldg_{fold}': nll_by_bldg.tolist(),

        f'rmse_{fold}': np.sqrt(np.mean(np.square(((means - preds))), axis=0)),
        f'rmspe_{fold}': np.sqrt(
            np.mean(np.square(((means[means > 0.0] - preds[means > 0.0]) / means[means > 0.0])), axis=0)),
        f'mae_{fold}': np.mean(np.abs(means - preds)),
        f'mape_{fold}': np.mean(np.abs((means[means > 0.0] - preds[means > 0.0]) / means[means > 0.0])),

        f'sums_ns_on_plate_{fold}': sums_ns_on_plate.tolist(),
        f'sums_estimated_bldg_cons_on_plate_{fold}': sums_estimated_bldg_cons_on_plate.tolist(),
        f'errors_for_sums_on_plate_{fold}': (sums_ns_on_plate - sums_estimated_bldg_cons_on_plate).tolist(),
        f'percent_errors_for_sums_on_plate_{fold}': (np.abs((sums_ns_on_plate - sums_estimated_bldg_cons_on_plate) /
                                                            sums_ns_on_plate)).tolist(),

        f'rmse_sums_by_plate_{fold}': np.sqrt(
            np.mean(np.square(((sums_ns_on_plate - sums_estimated_bldg_cons_on_plate))), axis=0)),
        f'rmspe_sums_by_plate_{fold}': np.sqrt(
            np.mean(np.square(((sums_ns_on_plate - sums_estimated_bldg_cons_on_plate) / sums_ns_on_plate)), axis=0)),
        f'mae_sums_by_plate_{fold}': np.mean(np.abs(sums_ns_on_plate - sums_estimated_bldg_cons_on_plate)),
        f'mape_sums_by_plate_{fold}': np.mean(
            np.abs((sums_ns_on_plate - sums_estimated_bldg_cons_on_plate) / sums_ns_on_plate)),
        f'95_cr_{fold}': np.mean(in_95_crs)
    }

    with open(os.path.join(output_dir, f'error_metrics_{fold}.json'), "w") as outfile:
        json.dump(error_metrics_dict, outfile)

    return error_metrics_dict


def gen_samps_and_obs_inner(graph_data_inner, dirmult_args, data_meters, data_n):
    n = []
    m = []
    for meter in graph_data_inner['meters']:
        n_connected = len(graph_data_inner['meters_to_bldgs'][meter])
        m.append(np.repeat(1. / n_connected, repeats=n_connected))

        meter_ind = data_meters.tolist().index(meter)

        n.append(data_n[meter_ind][0])
    n = np.round(n).astype(int)

    # init y's with nans, making sure that the lists are the right shape and size
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
            y_temp.append(np.nan)
        y.append(y_temp)

    # sample ys loop through each meter. We'll be sampling from the multinomial at the meter-level!
    for i, meter_temp in enumerate(graph_data_inner['meters']):
        n_temp = n[i]
        m_temp = m[i]
        y_samps = np.random.multinomial(n_temp, m_temp)

        # now we need to find matching buildings, and for each one,
        # update the lower-level y value
        matching_bldgs = graph_data_inner['meters_to_bldgs'][meter_temp]
        for matching_bldg, y_samp in zip(matching_bldgs, y_samps):
            matching_bldg_index = graph_data_inner['bldgs'].index(matching_bldg)

            meter_list_relative_to_bldg = graph_data_inner['bldgs_to_meters'][matching_bldg]
            meter_index_relative_to_bldg = meter_list_relative_to_bldg.index(meter_temp)

            y[matching_bldg_index][meter_index_relative_to_bldg] = y_samp

    samps = {
        'm': m,
        'y': y,
        'a_all': np.repeat(dirmult_args['a_all_init'], repeats=graph_data_inner['n_bldgs']),
        'b_all': np.repeat(dirmult_args['b_all_init'], repeats=graph_data_inner['n_bldgs'])
    }
    obs = {
        'n': n
    }

    return samps, obs


def generate_data(dirmult_args,
                  data_bldgs_to_meters,
                  data_meters_to_bldgs,
                  data_meters,
                  data_n):
    # restructure training data as graph data structure
    graph_data = []
    for bldgs_to_meters_for_k, \
            meters_to_bldgs_for_k in zip(data_bldgs_to_meters,
                                         data_meters_to_bldgs):
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

    samps = []
    obs = []
    for graph_data_inner in graph_data:
        samps_inner, obs_inner = \
            gen_samps_and_obs_inner(graph_data_inner,
                                    dirmult_args,
                                    data_meters,
                                    data_n)
        samps.append(samps_inner)
        obs.append(obs_inner)

    samps_tuples = \
        wrap_by_coloring_problem(dirmult_args,
                                 samps,
                                 obs,
                                 graph_data)

    return samps_tuples
