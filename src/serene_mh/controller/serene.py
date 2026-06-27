"""SERENE-MH controller (v1): a contextual Thompson-sampling operator selector.

Idea
----
At each step the search has a *context*: how much budget is left, whether it is
improving or stuck, how often moves are being accepted, and so on. Different
operators are good in different contexts (big destroy-and-rebuild moves help when
stuck; small tweaks help near the end). SERENE-MH learns this mapping online and
picks operators accordingly, instead of using fixed or non-contextual weights.

How it works
------------
1. CONTEXT. We read a few normalised numbers from the search telemetry (fraction
   of budget used, recent improvement, acceptance rate, stagnation, gap to best).

2. FEATURES. For each action we build a feature vector that places the context in
   that action's own "slot" (a block-structured one-hot-times-context encoding).
   This lets every action have its own context-dependent value - the thing that
   makes the policy genuinely *contextual* rather than a global weighting.

3. POLICY = Bayesian linear model + Thompson sampling. We keep a Gaussian
   posterior over a weight vector that maps features -> reward. To choose, we draw
   one sample from the posterior and score the actions with it; sampling is what
   gives exploration (well-understood actions are scored near their mean, uncertain
   ones can jump). We then update the posterior with the rewards we observe.

4. SURROGATE-GATED SLATE (optional). When `slate_size > n_exec`, we let the policy
   *propose* `slate_size` actions but only *evaluate* the `n_exec` that look best
   under the posterior mean (our cheap surrogate). This is the sample-efficiency
   lever: it matters most when evaluating a solution is expensive (the restoration
   problem), and is a no-op when slate_size == n_exec (plain contextual bandit).

5. COUNTERFACTUAL CREDIT. The reward an action earns is its improvement over the
   incumbent it started from - computed by the engine *before* the accept/reject
   gate - so an operator is credited for the move it produced, not for whether the
   acceptance rule happened to keep it.

6. TRANSFER. `export_mean()` / the `prior_mean` argument let a policy learned on
   one instance warm-start another instance of the same family (same operators).

Honest scope of v1
-------------------
The "surrogate" here is the policy's own posterior mean - a shared-parameter
model, so evaluating some actions already informs the predicted value of the
others (cross-action learning is built in). A *separate* gradient-boosted
surrogate and a formal doubly-robust / inverse-propensity off-policy estimator
(for crediting actions that were proposed but not evaluated) are planned
extensions; they are intentionally left out here so their statistics can be
designed carefully rather than approximated.
"""

import numpy as np


class SereneMH:
    """Contextual Thompson-sampling operator selector (the SERENE-MH controller).

    Parameters
    ----------
    actions : list
        The action list (from `build_actions`). Same object the engine indexes.
    slate_size : int
        How many actions the policy proposes each step (the "K").
    n_exec : int
        How many of the proposed actions are actually evaluated (the "k_exec").
        With slate_size == n_exec there is no gating: it is a plain contextual
        bandit that evaluates the n_exec sampled-best actions.
    exploration : float
        Scale of the Thompson-sampling noise (larger = more exploration).
    prior_precision : float
        Strength of the zero-mean prior on the weights (ridge regularisation).
    prior_mean : np.ndarray or None
        Optional warm-start weight vector (e.g. learned on another instance).
    """

    name = "serene-mh"

    # the context features we read from state.telemetry(), in a fixed order
    CONTEXT_KEYS = ("frac_budget", "recent_improvement", "accept_rate")

    def __init__(
        self,
        actions,
        slate_size: int = 1,
        n_exec: int = 1,
        exploration: float = 0.3,
        prior_precision: float = 1.0,
        prior_mean=None,
    ):
        self.actions = actions
        self.n = len(actions)
        self.slate_size = max(1, min(slate_size, self.n))
        self.n_exec = max(1, min(n_exec, self.slate_size))
        self.nu = exploration
        self.lam = prior_precision

        # feature size: each action gets a block of [bias, context...].
        self.m = len(self.CONTEXT_KEYS) + 2  # + stagnation + incumbent_gap (transformed)
        self.block = self.m + 1              # +1 bias term per action block
        self.d = self.n * self.block

        self.prior_mean = None if prior_mean is None else np.asarray(prior_mean, dtype=float)
        self._init_posterior()

        # filled in by `select`, read by `update`
        self._phi = None  # feature matrix for the current step, shape (n, d)

    # ------------------------------------------------------------------ posterior
    def _init_posterior(self):
        """Start (or restart) the Gaussian posterior over the weight vector.

        We track A_inv (the posterior covariance up to the noise scale) and b,
        with posterior mean mu = A_inv @ b. Starting A_inv = (1/lam) I encodes the
        zero-mean ridge prior; a `prior_mean` shifts the starting mean.
        """
        self.A_inv = np.eye(self.d) / self.lam
        if self.prior_mean is not None:
            self.b = self.lam * self.prior_mean.copy()
        else:
            self.b = np.zeros(self.d)
        self.mu = self.A_inv @ self.b

    def reset(self):
        self._init_posterior()
        self._phi = None

    # ------------------------------------------------------------------ features
    def _context(self, state) -> np.ndarray:
        """Turn the search telemetry into a short, bounded feature vector."""
        t = state.telemetry()
        # the three already-bounded signals
        base = [min(max(t[k], 0.0), 1.0) for k in self.CONTEXT_KEYS]
        # stagnation: a raw count -> soft value in [0, 1) that needs no fixed scale
        stagnation = t["stagnation"]
        stagnation_soft = stagnation / (stagnation + 10.0)
        # gap of the incumbent above the best-so-far, clipped
        gap = min(max(t["incumbent_gap"], 0.0), 1.0)
        return np.array(base + [stagnation_soft, gap], dtype=float)

    def _feature_matrix(self, context: np.ndarray) -> np.ndarray:
        """Build the (n_actions x d) feature matrix for the current context.

        Row i has [bias, context] written into action i's block and zeros
        elsewhere, so the model learns a separate context-to-reward map per action.
        """
        feat = np.concatenate(([1.0], context))  # length self.block
        phi = np.zeros((self.n, self.d))
        for i in range(self.n):
            phi[i, i * self.block:(i + 1) * self.block] = feat
        return phi

    # ------------------------------------------------------------------ selecting
    def _sample_weights(self, rng) -> np.ndarray:
        """Draw one weight vector from the posterior N(mu, nu^2 * A_inv)."""
        cov = self.nu ** 2 * self.A_inv
        cov = 0.5 * (cov + cov.T)  # keep it symmetric despite float drift
        # small jitter on the diagonal so the Cholesky factorisation is stable
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(self.d))
        z = rng.standard_normal(self.d)
        return self.mu + L @ z

    def select(self, state, rng) -> list[int]:
        context = self._context(state)
        phi = self._feature_matrix(context)
        self._phi = phi  # remember for update()

        # 1. propose: score actions with one Thompson sample, take the best `slate_size`
        sampled = phi @ self._sample_weights(rng)
        proposed = np.argsort(-sampled)[: self.slate_size]

        # 2. gate: among the proposed, evaluate the `n_exec` best under the mean
        if self.n_exec >= len(proposed):
            executed = proposed
        else:
            mean_scores = phi[proposed] @ self.mu
            keep = np.argsort(-mean_scores)[: self.n_exec]
            executed = proposed[keep]

        return [int(i) for i in executed]

    # ------------------------------------------------------------------ learning
    def update(self, state, outcomes, rng) -> None:
        """Update the posterior with the reward each evaluated action earned."""
        for o in outcomes:
            phi = self._phi[o.index]
            self._posterior_update(phi, o.reward)
        self.mu = self.A_inv @ self.b

    def _posterior_update(self, phi: np.ndarray, reward: float) -> None:
        """Rank-1 Bayesian-linear update (Sherman-Morrison) for one (phi, reward)."""
        Ainv_phi = self.A_inv @ phi
        denom = 1.0 + phi @ Ainv_phi
        self.A_inv -= np.outer(Ainv_phi, Ainv_phi) / denom
        self.b += reward * phi

    # ------------------------------------------------------------------ transfer
    def export_mean(self) -> np.ndarray:
        """Return the learned weight vector, to warm-start another instance."""
        return self.mu.copy()


def average_priors(means) -> np.ndarray:
    """Average several learned weight vectors into one prior.

    A simple, transparent stand-in for the hierarchical/empirical-Bayes prior:
    train on a few instances of a family, average the learned weights, and pass
    the result as `prior_mean` to warm-start a fresh instance. (A fuller
    empirical-Bayes treatment is a planned extension.)
    """
    stack = np.vstack([np.asarray(m, dtype=float) for m in means])
    return stack.mean(axis=0)
