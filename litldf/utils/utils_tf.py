import sys, os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import tensorflow as tf
import ldf.utils.utils_stat as us
import matplotlib.pyplot as plt
keras = tf.keras
model_from_json = keras.models.model_from_json

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))


def load_saved_model(json_path, weights_path):
    with open(json_path, 'r') as f:
        model = model_from_json(f.read())  # load the model
    model.load_weights(weights_path)
    return model


def eval_lstm_gapo_calibration(output_dir, a_all, b_all, test_y, fold):
    a_all_flat = a_all.flatten()
    b_all_flat = b_all.flatten()
    test_y_flat = test_y.flatten()

    cdfs = []
    # loop through each sample in the val set
    for i, (a, b, y) in enumerate(zip(a_all_flat, b_all_flat, test_y_flat)):

        cdf = us.gammapoisson_cdf(y, a, b)

        # add to cdfs
        cdfs.append(cdf)

    cdfs = np.array(cdfs)

    fig = plt.figure()
    plt.hist(cdfs, bins=50)
    plt.title(f'Model Calibration Histogram for Fold {fold}')
    plt.xlabel('Cumulative Distribution Function Evaluations \n at Observed Conusmption Values')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'calibration_fold_{fold}.pdf'))
    plt.close(fig)

    return cdfs
