from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

from scipy.stats import gamma

from ldf.models.distn_base import DistnBase
import ldf.utils.utils_stat as us


class DistnGam(DistnBase):

    def __init__(self, params):
        super().__init__(params)

        self.epsilon = np.finfo(np.float32).eps

    def get_distn_type(self):
        return 'continuous'

    @staticmethod
    def np_ll(x, a, b):
        # define the log likelihood function via numpy
        return us.gamma_ab_logpdf(x, a, b)

    def tf_nll(self, y_true, y_pred):  # 2 dimensional y_true and y_pred

        a_of_x_tf = K.flatten(y_pred[:, :, 0]) + self.epsilon
        b_of_x_tf = K.flatten(y_pred[:, :, 1]) + self.epsilon

        y_true = K.flatten(y_true)

        dist = tfp.distributions.Gamma(concentration=a_of_x_tf,
                                       rate=b_of_x_tf)

        log_p_si_given_xi_tf = dist.log_prob(y_true)

        return -tf.reduce_mean(log_p_si_given_xi_tf)

    def interpret_predict_output_ts(self, yhat):
        a_all = yhat[:, :, 0] + self.epsilon
        b_all = yhat[:, :, 1] + self.epsilon
        mean_preds = a_all / b_all
        mean_preds = mean_preds.reshape(mean_preds.shape[0], mean_preds.shape[1], 1)

        preds_params = {
            'a_all': a_all,
            'b_all': b_all
        }

        a_all_flat = a_all.flatten()
        b_all_flat = b_all.flatten()

        preds_params_flat = []
        for a, b in zip(a_all_flat, b_all_flat):
            preds_params_flat.append({
                'a_all': a,
                'b_all': b
            })

        return mean_preds, preds_params, preds_params_flat

    def compute_nll(self, preds_params, y):
        train_a_all = preds_params['a_all']
        train_b_all = preds_params['b_all']

        nlls = -1. * self.np_ll(y, train_a_all, train_b_all)
        mean_nll = np.mean(nlls)

        return mean_nll, nlls

    def ppf(self, x, a, b):
        return gamma.ppf(x, float(a), scale=1. / float(b))

    def ppf_params(self, x, params):

        a = params['a_all']
        b = params['b_all']

        return self.ppf(x, a, b)

    def cdf(self, x, a, b):
        return gamma.cdf(x, a, scale=1. / b)


    def cdf_params(self, y, params):
        a_all = params['a_all']
        b_all = params['b_all']

        a_all_flat = a_all.flatten()
        b_all_flat = b_all.flatten()
        y_flat = y.flatten()

        cdfs = self.cdf(y_flat, a_all_flat, b_all_flat)

        return cdfs
