from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

from scipy.stats import norm

from ldf.models.distn_base import DistnBase
import ldf.utils.utils_stat as us


class DistnNorm(DistnBase):

    def __init__(self, params):
        super().__init__(params)

        self.epsilon = np.finfo(np.float32).eps

    def get_distn_type(self):

        return 'continuous'

    @staticmethod
    def np_ll(x, mean, sig):

        # define the log likelihood function via numpy
        return norm.logpdf(x, loc=mean, scale=sig)

    def tf_nll(self, y_true, y_pred):  # 2 dimensional y_true and y_pred

        mu = K.flatten(y_pred[:, 0])
        sig = K.flatten(y_pred[:, 1]) + self.epsilon

        y_true = K.flatten(y_true)

        dist = tfp.distributions.Normal(loc=mu,
                                        scale=sig)

        log_p_si_given_xi_tf = dist.log_prob(y_true)

        return -tf.reduce_mean(log_p_si_given_xi_tf)

    def interpret_predict_output(self, yhat):

        mu = yhat[:, 0]
        sig = yhat[:, 1] + self.epsilon
        mean_preds = mu

        std_pred = sig

        preds_params = {
            'mu': mu,
            'sig': sig
        }

        mu_flat = mu.flatten()
        sig_flat = sig.flatten()

        preds_params_flat = []
        for mu_temp, sig_temp in zip(mu_flat, sig_flat):
            preds_params_flat.append({
                'mu': mu_temp,
                'sig': sig_temp
            })

        return mean_preds, std_pred, preds_params, preds_params_flat

    def compute_nll(self, preds_params, y):

        train_mu = preds_params['mu']
        train_sig = preds_params['sig']

        nlls = -1. * self.np_ll(y, train_mu, train_sig)
        mean_nll = np.mean(nlls)

        return mean_nll, nlls

    def ppf(self, x, mu, sig):

        return norm.ppf(x, loc=mu, scale=sig)

    def ppf_params(self, x, params):
        mu = params['mu']
        sig = params['sig']

        return self.ppf(x, mu, sig)

    def cdf(self, x, mean, sig):

        # (q, n, p, loc=0)
        return norm.cdf(x, loc=mean, scale=sig)

    def cdf_params(self, y, params):
        mu = params['mu']
        sig = params['sig']

        mu_flat = mu.flatten()
        sig_flat = sig.flatten()
        y_flat = y.flatten()

        cdfs = self.cdf(y_flat, mu_flat, sig_flat)

        return cdfs

    def sample_posterior_params(self, params, size=None):

        mu = params['mu']
        sig = params['sig']

        return us.normal_sample(mu, sig, size=size)

    def get_output_dim(self):

        return 2
