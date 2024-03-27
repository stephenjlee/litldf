from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

from scipy.stats import nbinom

from ldf.models.distn_base import DistnBase
import ldf.utils.utils_stat as us

class DistnGapo(DistnBase):

    def __init__(self, params):
        super().__init__(params)
        self.a0 = params['a0']
        self.b0 = params['b0']
        self.nu = params['nu']

    def get_distn_type(self):
        return 'discrete'

    @staticmethod
    def np_ll(k, a, b):
        # define the log likelihood function via numpy
        return us.gammapoisson_logpmf(k, a, b)

    def tf_nll(self, y_true, y_pred):  # 2 dimensional y_true and y_pred

        a_of_x_tf = K.flatten(y_pred[:, 0])
        b_of_x_tf = K.flatten(y_pred[:, 1])

        a_all_tf = a_of_x_tf + self.a0
        b_all_tf = b_of_x_tf + self.b0

        n = a_all_tf
        p = 1.0 / (1.0 + b_all_tf)

        y_true = K.flatten(y_true)

        dist = tfp.distributions.NegativeBinomial(
            total_count=n,
            probs=p)

        log_p_si_given_xi_tf = dist.log_prob(y_true)

        return -tf.reduce_mean(log_p_si_given_xi_tf)

    def tf_rmse(self, y_true, y_pred):  # 2 dimensional y_true and y_pred

        a_of_x_tf = K.flatten(y_pred[:, 0])
        b_of_x_tf = K.flatten(y_pred[:, 1])

        a_all_tf = a_of_x_tf + self.a0
        b_all_tf = b_of_x_tf + self.b0

        s_true = K.flatten(y_true)

        s_pred = a_all_tf / b_all_tf

        return K.sqrt(K.mean(K.square(s_pred - s_true)))

    def get_std(self, a_all, b_all):
        return np.sqrt(us.gammapoisson_var(a_all, b_all))

    def get_std_params(self, params):
        a_all = params['a_all']
        b_all = params['b_all']

        return self.get_std(a_all, b_all)

    def interpret_predict_output(self, yhat):
        a_all = yhat[:, 0] + self.a0
        b_all = yhat[:, 1] + self.b0
        mean_preds = a_all / b_all

        std_pred = self.get_std(a_all, b_all)

        preds_params = {
            'a_all': a_all,
            'b_all': b_all
        }

        preds_params_flat = []
        for a, b in zip(a_all, b_all):
            preds_params_flat.append({
                'a_all': a,
                'b_all': b
            })

        return mean_preds, std_pred, preds_params, preds_params_flat

    def compute_nll(self, preds_params, y):
        train_a_all = preds_params['a_all']
        train_b_all = preds_params['b_all']

        nlls = -1. * self.np_ll(y, train_a_all, train_b_all)
        mean_nll = np.mean(nlls)

        return mean_nll, nlls

    def ppf(self, x, a, b):
        # derivation of GaPo variance
        r_wiki = a
        p_wiki = 1. / (b + 1.)

        # switching the definition of successes and failures
        n_wiki_alt = r_wiki
        p_wiki_alt = 1. - p_wiki

        # (q, n, p, loc=0)
        return nbinom.ppf(x, n_wiki_alt, p_wiki_alt)

    def ppf_params(self, x, params):
        a = params['a_all']
        b = params['b_all']

        return self.ppf(x, a, b)

    def cdf(self, x, a, b):
        # derivation of GaPo variance
        r_wiki = a
        p_wiki = 1. / (b + 1.)

        # switching the definition of successes and failures
        n_wiki_alt = r_wiki
        p_wiki_alt = 1. - p_wiki

        # (q, n, p, loc=0)
        return nbinom.cdf(x, n_wiki_alt, p_wiki_alt)

    def cdf_params(self, y, params):
        a_all = params['a_all']
        b_all = params['b_all']

        a_all_flat = a_all.flatten()
        b_all_flat = b_all.flatten()
        y_flat = y.flatten()

        cdfs = self.cdf(y_flat, a_all_flat, b_all_flat)

        return cdfs

    def sample_posterior_params(self, params, size=None):
        a_all = params['a_all']
        b_all = params['b_all']

        return us.gammapoisson_sample(a_all, b_all, size=size)

    def get_output_dim(self):
        return 2
