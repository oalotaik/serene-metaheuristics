"""A lightweight LightGBM surrogate for the gating step.

The ablation showed the surrogate-gated slate is what makes SERENE-MH efficient,
and the gating score was a *linear* posterior mean. This surrogate replaces that
score with a gradient-boosted model that predicts the reward (improvement) of an
action in a given context, so the gate can capture nonlinear context x action
effects the linear model cannot.

It is trained online on the (context, action, observed reward) tuples the search
logs, and refit periodically (GBDTs do not update incrementally). Until it has
enough data it reports `ready == False`, and the controller falls back to the
linear gating score.
"""

import warnings

import numpy as np


class LGBMSurrogate:
    """Online LightGBM regressor: predicts reward for (context, action)."""

    def __init__(self, n_actions, refit_every=50, min_data=50, **lgbm_params):
        self.n_actions = n_actions
        self.refit_every = refit_every
        self.min_data = min_data
        self.params = dict(
            n_estimators=60, num_leaves=15, learning_rate=0.1,
            min_child_samples=5, subsample=1.0, verbosity=-1,
        )
        self.params.update(lgbm_params)
        self.reset()

    def reset(self):
        self._ctx = []      # list of context vectors
        self._act = []      # list of action indices
        self._reward = []   # list of observed rewards
        self.model = None
        self._since_fit = 0

    def observe(self, context, action_index, reward):
        self._ctx.append(np.asarray(context, dtype=float))
        self._act.append(int(action_index))
        self._reward.append(float(reward))

    def _design_matrix(self, contexts, actions):
        """Stack context rows with the action index as the last (categorical) column."""
        contexts = np.asarray(contexts, dtype=float).reshape(len(actions), -1)
        actions = np.asarray(actions, dtype=float).reshape(-1, 1)
        return np.hstack([contexts, actions])

    def maybe_refit(self):
        """Refit every `refit_every` steps once enough data has accumulated."""
        self._since_fit += 1
        if len(self._reward) < self.min_data or self._since_fit < self.refit_every:
            return
        self._since_fit = 0
        import lightgbm as lgb

        X = self._design_matrix(self._ctx, self._act)
        y = np.asarray(self._reward, dtype=float)
        cat_index = X.shape[1] - 1  # the action column
        try:
            model = lgb.LGBMRegressor(**self.params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y, categorical_feature=[cat_index])
            self.model = model
        except Exception:
            pass  # keep the previous model if a refit fails (e.g. degenerate data)

    @property
    def ready(self):
        return self.model is not None

    def predict(self, context, action_indices):
        """Predicted reward for each action index in the given context."""
        X = self._design_matrix(
            np.tile(np.asarray(context, dtype=float), (len(action_indices), 1)),
            action_indices,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict(X)
