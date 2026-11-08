"""
RandomForest.utils
==================

Mirror of Mondrian_RF/utils.py and kNN_Euclidean/utils.py.

css_utils imports exactly two names from this module:
    two_one_norm
    evaluate_all_lifetimes

Both are provided with IDENTICAL signatures to the Mondrian originals, so the
only change needed in css_utils is the import line.

evaluate_all_lifetimes
----------------------
The Mondrian version sweeps the LIFETIME (its complexity knob) and returns a
curve {'times', 'mse'}; downstream code plots it and takes np.min(mse) as the
"best MSE over lifetimes".

The random-forest analogue sweeps MAX_DEPTH (the RF complexity knob) and returns
the same {'times', 'mse', 'y_hat_test'} structure. 'times' holds the max_depth
values, so np.min(res['mse']) is the best MSE over the depth sweep -- mirroring
the Mondrian convention (both minimise over the complexity sweep on the test
set, so method-to-method comparisons stay apples-to-apples with the existing
Mondrian results).

The forest size M passed by the driver (n_estimators in the experiment) is used
as the number of trees. lifetime_max, delta and rng are accepted for signature
parity and ignored (a random forest has no lifetime, ridge term, or external
RNG -- the depth sweep replaces the lifetime sweep).

ABLATION KNOBS (PI request)
---------------------------
BOOTSTRAP and MAX_FEATURES below are the two switches that separate bagging from
feature subsampling. Edit them (or pass bootstrap=/max_features= through) to run
the ablation:
    standard RF          : BOOTSTRAP=True,  MAX_FEATURES=1.0  (sklearn default)
    no bagging           : BOOTSTRAP=False
    feature subsampling  : MAX_FEATURES="sqrt"   (or a fraction like 0.33)

NOTE ON PLOT LABELS: css_utils labels the x-axis "Lifetime". Under this module
that axis is actually max_depth. The numbers and minima are correct; only the
label text is now a misnomer. If you want it accurate, change the xlabel string
in css_utils to "max_depth".
"""

import numpy as np
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error

from .RandomForest import RFRegressor


# Complexity sweep for the random forest (analogue of the lifetime grid).
# Ascending max_depth means INCREASING model complexity (deep trees overfit).
MAX_DEPTH_GRID = (1, 2, 3, 4, 5, 7, 10, 15, 20, 30)

# Ablation switches (see module docstring).
BOOTSTRAP    = True     # bagging on/off
MAX_FEATURES = 1.0      # per-split feature subsampling; 1.0 = all features

# Forest randomness / parallelism.
RANDOM_STATE = 0
N_JOBS       = -1


def two_one_norm(H):
    """Identical to Mondrian_RF.utils.two_one_norm."""
    return np.sum(np.apply_along_axis(norm, 0, H)) / H.shape[1]


def evaluate_all_lifetimes(
    X, y, X_test, y_test, M, lifetime_max, delta=0, rng=np.random,
    max_depth_grid=MAX_DEPTH_GRID, bootstrap=BOOTSTRAP,
    max_features=MAX_FEATURES, random_state=RANDOM_STATE, n_jobs=N_JOBS,
):
    """
    Drop-in replacement for Mondrian_RF.utils.evaluate_all_lifetimes.

    Parameters
    ----------
    X, y, X_test, y_test
        Train / test features and targets.
    M : int
        Number of trees in the forest (forest size passed by the driver).
    lifetime_max, delta, rng
        Accepted for signature compatibility with the Mondrian version and
        IGNORED (a random forest has no lifetime, ridge term, or RNG). The depth
        sweep replaces the lifetime sweep.
    max_depth_grid : iterable of int, default=MAX_DEPTH_GRID
        The RF complexity sweep.
    bootstrap : bool, default=BOOTSTRAP
        Bagging switch (ablation knob).
    max_features : default=MAX_FEATURES
        Feature subsampling (ablation knob).

    Returns
    -------
    results : dict
        {'times': [max_depth...], 'mse': [test MSE...], 'y_hat_test': ndarray}
        'times' is ascending max_depth. np.min(results['mse']) = best MSE over
        the sweep.
    y_hat_test : ndarray
        Test predictions at the last (largest) max_depth evaluated -- returned
        only to match the Mondrian function's 2-tuple signature; downstream code
        uses results['mse'].
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    X_test = np.ascontiguousarray(np.asarray(X_test, dtype=float))
    y = np.asarray(y)
    y_test = np.asarray(y_test)

    times = []
    mse_list = []
    y_hat_test = None

    for depth in max_depth_grid:
        model = RFRegressor(
            n_estimators=int(M), max_depth=int(depth),
            bootstrap=bootstrap, max_features=max_features,
            random_state=random_state, n_jobs=n_jobs,
        ).fit(X, y)
        y_hat = model.predict(X_test)
        times.append(int(depth))
        mse_list.append(float(mean_squared_error(y_test, y_hat)))
        y_hat_test = y_hat

    results = {"times": times, "y_hat_test": y_hat_test, "mse": mse_list}
    return results, y_hat_test