"""
kNN_Euclidean.utils
===================

Mirror of Mondrian_RF/utils.py (Mondrian_utils.py).

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

The k-NN analogue sweeps the NUMBER OF NEIGHBOURS (k-NN's complexity knob) and
returns the same {'times', 'mse', 'y_hat_test'} structure. 'times' holds the
n_neighbors values, so np.min(res['mse']) is the best MSE over the neighbour
sweep -- mirroring the Mondrian convention (both minimise over the complexity
sweep on the test set, so method-to-method comparisons stay apples-to-apples
with the existing Mondrian results).

Consequence for the rotation-invariance test: at k=d the TrIM and Weighted CSS
feature sets are related by a rotation + a global scalar, to which Euclidean
k-NN is exactly invariant. So for EVERY n_neighbors value the two give
identical predictions -- their entire MSE curves coincide, not merely their
minima. (This holds at weight_power=1.0, where Weighted CSS = P Λ is a pure
rotation of TrIM = P Λ Pᵀ.)

NO feature scaling is applied anywhere (see kNN_Euclidean.kNN).

NOTE ON PLOT LABELS: css_utils labels the x-axis "Lifetime". Under this module
that axis is actually n_neighbors. The numbers and minima are correct; only the
label text is now a misnomer. If you want it accurate, change the xlabel string
in css_utils to "n_neighbors".
"""

import numpy as np
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error

from .kNN import KNNEuclideanRegressor


# Complexity sweep for k-NN (analogue of the lifetime grid). Ascending k means
# decreasing model complexity (k=1 is the most flexible fit).
N_NEIGHBORS_GRID = (1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100)


def two_one_norm(H):
    """Identical to Mondrian_RF.utils.two_one_norm."""
    return np.sum(np.apply_along_axis(norm, 0, H)) / H.shape[1]


def evaluate_all_lifetimes(
    X, y, X_test, y_test, M, lifetime_max, delta=0, rng=np.random,
    n_neighbors_grid=N_NEIGHBORS_GRID, weights="uniform",
):
    """
    Drop-in replacement for Mondrian_RF.utils.evaluate_all_lifetimes.

    Parameters
    ----------
    X, y, X_test, y_test
        Train / test features and targets.
    M, lifetime_max, delta, rng
        Accepted for signature compatibility with the Mondrian version and
        IGNORED (k-NN has no trees, lifetime, ridge term, or RNG). The neighbour
        sweep replaces the lifetime sweep.
    n_neighbors_grid : iterable of int, default=N_NEIGHBORS_GRID
        The k-NN complexity sweep. Values >= n_train are skipped.
    weights : {"uniform", "distance"}, default="uniform"
        Passed to the k-NN regressor.

    Returns
    -------
    results : dict
        {'times': [n_neighbors...], 'mse': [test MSE...], 'y_hat_test': ndarray}
        'times' is ascending n_neighbors. np.min(results['mse']) = best MSE over
        the sweep.
    y_hat_test : ndarray
        Test predictions at the last (largest) n_neighbors evaluated -- returned
        only to match the Mondrian function's 2-tuple signature; downstream code
        uses results['mse'].
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    X_test = np.ascontiguousarray(np.asarray(X_test, dtype=float))
    y = np.asarray(y)
    y_test = np.asarray(y_test)
    n_train = X.shape[0]

    times = []
    mse_list = []
    y_hat_test = None

    for k in n_neighbors_grid:
        if k >= n_train:
            continue
        model = KNNEuclideanRegressor(n_neighbors=int(k), weights=weights).fit(X, y)
        y_hat = model.predict(X_test)
        times.append(int(k))
        mse_list.append(float(mean_squared_error(y_test, y_hat)))
        y_hat_test = y_hat

    # Tiny-data fallback (only triggers if every grid value >= n_train).
    if not times:
        model = KNNEuclideanRegressor(n_neighbors=1, weights=weights).fit(X, y)
        y_hat_test = model.predict(X_test)
        times = [1]
        mse_list = [float(mean_squared_error(y_test, y_hat_test))]

    results = {"times": times, "y_hat_test": y_hat_test, "mse": mse_list}
    return results, y_hat_test