"""
kNN_Euclidean.kNN
=================

Mirror of Mondrian_RF/Mondrian_forest.py.

This module supplies the downstream regressor used by
kNN_Euclidean.utils.evaluate_all_lifetimes -- a plain Euclidean k-NN
(KNNEuclideanRegressor).

EGOP (H) ESTIMATION IS INTENTIONALLY UNCHANGED
----------------------------------------------
The experiment isolates ONE variable: the downstream learner. It feeds the
*same* estimated H to a different regressor, so the EGOP estimation must stay
exactly as before -- it still probes the finite-difference gradients of a
Mondrian forest. (A k-NN-based EGOP would be a different, noisier estimator and
would confound the comparison.)

We therefore RE-EXPORT the original Mondrian TrIM (and friends) here, so that

    from kNN_Euclidean.kNN import TrIM

resolves to the identical Mondrian-based estimator. css_utils uses TrIM only to
build H (forest.mf.fit -> _get_importance -> H) and to apply the linear
transform forest.transform(...) = X @ (H / two_one_norm(H)); none of that
changes. Only evaluate_all_lifetimes (the regression head) becomes k-NN.

If the Mondrian_RF package is not importable, TrIM re-export is skipped so the
k-NN regressor can still be used standalone.
"""

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.neighbors import KNeighborsRegressor

# --- Re-export the Mondrian EGOP estimator UNCHANGED -----------------------
try:
    from Mondrian_RF.Mondrian_forest import (
        TrIM,
        MondrianForestRegressor,
        WeightedMondrianForestRegressor,
    )
except Exception as _e:  # pragma: no cover - only if Mondrian_RF is absent
    TrIM = None
    MondrianForestRegressor = None
    WeightedMondrianForestRegressor = None
    _MONDRIAN_IMPORT_ERROR = _e


class KNNEuclideanRegressor(RegressorMixin):
    """
    Euclidean k-NN regressor (analogue of MondrianForestRegressor).

    Thin wrapper over sklearn.neighbors.KNeighborsRegressor with the metric
    fixed to "euclidean" -- the property that makes the learner invariant to
    rotations (and to a global scalar) of the feature space, which is exactly
    what the experiment tests.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours (the k-NN complexity knob; analogue of lifetime).
    weights : {"uniform", "distance"}, default="uniform"
        Neighbour weighting. Both choices are rotation invariant.

    Notes
    -----
    NO feature scaling is performed. Per-feature standardization is an
    anisotropic, axis-dependent rescaling that would break rotation invariance
    and erase the H magnitude weighting; it must not be applied here.
    """

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._model = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=float))
        y = np.asarray(y)
        self.X = X
        self.y = y
        k = int(min(self.n_neighbors, X.shape[0]))
        k = max(k, 1)
        self._model = KNeighborsRegressor(
            n_neighbors=k, weights=self.weights, metric="euclidean"
        )
        self._model.fit(X, y)
        return self

    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=float))
        return self._model.predict(X)

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, "weights": self.weights}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self