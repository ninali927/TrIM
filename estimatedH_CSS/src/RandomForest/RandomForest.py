"""
RandomForest.RandomForest
=========================

Self-contained random-forest analogue of Mondrian_RF.Mondrian_forest.
Random-forest code lives ONLY here; Mondrian_forest.py has no random-forest
references (and this file has no Mondrian references).

Two classes, mirroring the Mondrian side one-for-one:

    RFRegressor  -- mirrors MondrianForestRegressor. Thin wrapper over
                    sklearn.ensemble.RandomForestRegressor. Used BOTH as the
                    downstream regressor (RandomForest.utils.evaluate_all_lifetimes,
                    sweeping max_depth) AND as the probed model inside RFTrIM
                    (finite-difference EGOP estimation) -- exactly how
                    MondrianForestRegressor serves both roles on the Mondrian
                    side. One class, two roles, same as Mondrian.

    RFTrIM       -- mirrors TrIM. Identical _get_importance /
                    estimate_H_finite_diff / fit / reiterate / transform /
                    predict logic (H is defined the same way), but the probed
                    model is RFRegressor and the constructor takes
                    bootstrap / max_features / random_state directly as real
                    parameters -- no global backend switch, nothing silently
                    dropped.

ABLATION KNOBS (PI request: bagging vs feature subsampling)
-----------------------------------------------------------
    bootstrap     -> bagging (per-tree bootstrap resampling)
    max_features  -> per-split feature subsampling
Both now reach the EGOP-probing model (via RFTrIM's constructor) AND the
downstream regressor (via evaluate_all_lifetimes), so an ablation run is
apples-to-apples across both uses -- unlike the previous wiring, where these
only reached the downstream regressor.

NOTE: this file used to re-export TrIM from Mondrian_RF.Mondrian_forest so
that `from RandomForest.RandomForest import TrIM` picked up whichever backend
EGOP_BACKEND selected. That switch is gone. Anything that imported TrIM from
here (e.g. css_utils_direct.py, used by rotation_main_rf.py) needs to import
RFTrIM instead when running the RF pipeline -- not fixed here since those
files weren't part of this change.
"""

import numpy as np
from copy import deepcopy
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor


class RFRegressor(RegressorMixin):
    """
    Standard random forest regressor (analogue of MondrianForestRegressor).

    Thin wrapper over sklearn.ensemble.RandomForestRegressor. The complexity
    knob swept by evaluate_all_lifetimes is max_depth (analogue of the Mondrian
    lifetime): deeper trees -> more flexible fit.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum tree depth (the complexity knob). None grows trees fully.
    bootstrap : bool, default=True
        Bagging switch. True = per-tree bootstrap resampling (standard RF);
        False = every tree sees the full training set (no bagging).
    max_features : {"sqrt", "log2", None}, int or float, default=1.0
        Per-split feature subsampling. 1.0 / None = consider all features at
        every split (no feature subsampling); "sqrt" or a fraction enables it.
    random_state : int, default=0
    n_jobs : int, default=-1
        Parallelism over trees.
    """

    def __init__(self, n_estimators=100, max_depth=None, bootstrap=True,
                 max_features=1.0, random_state=0, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model = None

    def set_random_state(self):
        # Parity with MondrianForestRegressor / TrIM's fit(), which calls
        # self.mf.set_random_state() before fitting. A random forest consumes
        # its seed directly at fit time (random_state below), so there is
        # nothing to pre-seed here.
        pass

    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=float))
        y = np.asarray(y)
        self._model = RandomForestRegressor(
            n_estimators=int(self.n_estimators),
            max_depth=self.max_depth,
            bootstrap=self.bootstrap,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=float))
        return self._model.predict(X)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "bootstrap": self.bootstrap,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class RFTrIM(RegressorMixin):
    '''
    RFTrIM is the random-forest analogue of Mondrian_forest.TrIM. Same
    algorithm end to end (_get_importance / estimate_H_finite_diff / fit /
    reiterate / transform / predict are unchanged from TrIM -- H is defined
    identically), but the probed model is RFRegressor instead of
    MondrianForestRegressor, and bootstrap / max_features are real constructor
    parameters that reach that probed model directly.

    Parameters
    ----------
    mf : RFRegressor, default=None, the regressor to probe; if None, a new
        RFRegressor is built from the other parameters below
    n_estimators : int, default=100, number of trees in the probed forest
    max_depth : int or None, default=None, tree depth of the probed forest
    bootstrap : bool, default=True, bagging switch (ablation knob)
    max_features : default=1.0, feature subsampling (ablation knob)
    n_jobs : int, default=-1
    iteration : int, default=1, number of iterations
    step_size : float, default=0.1
        Step size for finite-difference gradient estimation. NOTE: a random
        forest is piecewise-constant, so step_size must stay LARGE enough that
        the +/- probe points land in different leaves (a small step can give a
        zero gradient). Match whatever step_size worked in rotation_main_rf.py
        (3.0) rather than TrIM's small Mondrian default.
    random_state : int, default=42
    '''

    def __init__(self, mf: RFRegressor = None,
                 n_estimators=100, max_depth=None, bootstrap=True,
                 max_features=1.0, n_jobs=-1,
                 iteration=1, step_size=0.1, random_state=42) -> None:
        if mf is None:
            self.mf = RFRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                   bootstrap=bootstrap, max_features=max_features,
                                   random_state=random_state, n_jobs=n_jobs)
        else:
            self.mf = mf
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.step_size = step_size
        self.iteration = iteration
        self.X = None
        self.y = None
        self.H = None

    def _get_importance(self):
        importance = []

        for dim in range(self.X.shape[1]):
            x_eval_pos = deepcopy(self.X)
            x_eval_neg = deepcopy(self.X)
            x_eval_pos[:, dim] = x_eval_pos[:, dim] + self.step_size / 2.0
            x_eval_neg[:, dim] = x_eval_neg[:, dim] - self.step_size / 2.0

            y_eval_pos = self.mf.predict(self.transform(x_eval_pos))
            y_eval_neg = self.mf.predict(self.transform(x_eval_neg))

            y_diff = y_eval_pos - y_eval_neg
            importance_temp = y_diff / self.step_size
            importance.append(importance_temp)

        # the shape of importance is (n_features, n_samples)
        importance = np.vstack(importance)
        return importance

    def estimate_H_finite_diff(self):
        importance = self._get_importance()
        H = np.matmul(importance, np.transpose(importance)) / self.X.shape[0]
        return H

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = y
        self.mf.set_random_state()
        self.mf.fit(X, y)
        if self.iteration > 0:
            self.H = self.estimate_H_finite_diff()
            for _ in range(self.iteration - 1):
                self.reiterate()
            self.mf.fit(self.transform(deepcopy(self.X)), self.y)
        return self

    def reiterate(self):
        from .utils import two_one_norm  # local import: RF utils, not Mondrian's
        X = self.transform(deepcopy(self.X))
        self.mf.fit(X, self.y)
        self.H = self.estimate_H_finite_diff()

    def transform(self, X):
        from .utils import two_one_norm
        if self.H is None:
            return X
        return np.matmul(X, self.H / two_one_norm(self.H))

    def predict(self, X):
        if self.iteration > 0:
            return self.mf.predict(self.transform(X))
        else:
            return self.mf.predict(X)

    def set_params(self, **params):
        for key, value in params.items():
            if key in ["n_estimators", "max_depth", "bootstrap", "max_features", "n_jobs"]:
                setattr(self.mf, key, value)
            else:
                setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        return {
            "n_estimators": self.mf.n_estimators,
            "max_depth": self.mf.max_depth,
            "bootstrap": self.bootstrap,
            "max_features": self.max_features,
            "step_size": self.step_size,
        }


class WeightedRFForestRegressor(RFTrIM):
    '''
    Axis-aligned analogue of RFTrIM (mirrors WeightedMondrianForestRegressor):
    H is diagonal instead of full. Included now for structural parity;
    not yet used by any driver script -- needed later for the axis-aligned
    task, at which point it should be checked the same way RFTrIM was.
    '''

    def estimate_H_finite_diff(self):
        importance = self._get_importance()
        H = np.diag(np.sum(importance ** 2, axis=1)) / self.X.shape[0]
        return H