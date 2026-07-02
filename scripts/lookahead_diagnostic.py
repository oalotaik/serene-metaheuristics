"""Multi-step credit diagnostic: does context predict which operator sets up
better FUTURE gains, beyond its immediate one-step reward?

The one-step diagnostic (context_diagnostic.py) uses immediate Delta only. Here we
attribute a k-step RETURN to each operator: from a sampled state, commit to the
operator, then continue a cheap search for k-1 more steps, and measure cumulative
improvement. If context's per-step operator-selection regret reduction GROWS with
horizon k, context carries multi-step (setup) signal the one-step test misses.

Exploratory only (no PROJECT_LOG / git changes). Parametrized like the one-step
tool.

    python scripts/lookahead_diagnostic.py                    # 6 inst x 2 seeds, ~30 states, k=5, R=2
    python scripts/lookahead_diagnostic.py --k 8 --states 40
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from serene_mh.core import build_actions, SimulatedAnnealing
from serene_mh.problems import load_cflp_benchmarks
from context_diagnostic import (A_PROGRESS, B_STRUCTURE, structure_features, op_descriptor,
                                instance_features, _make_model, _build_X)
from serene_mh.core.engine import SearchState

warnings.filterwarnings("ignore")

DEFAULT_INSTANCES = ["cap41", "cap71", "cap81", "cap101", "cap111", "cap131"]
DEFAULT_SEEDS = [0, 1]
OUT = "results/tables/cflp_context_diagnostic"


def _rollout_checkpoints(prob, first_cand, actions, rng, horizons):
    """One greedy random-operator rollout starting at first_cand (= depth 1).
    Returns {h: best objective reached by depth h} for each horizon in `horizons`,
    so all horizons come from the SAME rollout (fair comparison, cheap)."""
    maxk = max(horizons)
    cur = first_cand
    obj_at = {}
    if 1 in horizons:
        obj_at[1] = cur.objective
    for step in range(2, maxk + 1):
        act = actions[int(rng.integers(len(actions)))]
        c = act.operator.apply(cur, rng, **act.params)
        c.objective = prob.evaluate(c)
        if c.objective < cur.objective:
            cur = c
        if step in horizons:
            obj_at[step] = cur.objective
    return obj_at


def generate_lookahead_log(instances, seeds, n_iters, n_states, horizons, R):
    rows = []
    for name, prob in instances.items():
        actions = build_actions(prob.operators())
        allopen = float((prob.unit_cost.min(axis=0) * prob.demand).sum())
        inst_feats = instance_features(prob)
        sample_every = max(1, n_iters // n_states)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            acc = SimulatedAnnealing(max_evals=n_iters)  # host just to walk the state space
            inc = prob.initial_solution(rng)
            inc.objective = prob.evaluate(inc)
            st = SearchState(problem=prob, incumbent=inc, best=inc, n_evals=1, max_evals=n_iters)
            acc.reset()
            for it in range(n_iters):
                if it % sample_every == 0:                      # snapshot -> rollouts here
                    ctx = st.telemetry()
                    struct = structure_features(prob, st.incumbent.data, allopen)
                    s = st.incumbent
                    base = abs(s.objective) + 1e-9
                    for action in actions:
                        ret = {h: [] for h in horizons}
                        for _ in range(R):
                            cand = action.operator.apply(s, rng, **action.params)
                            cand.objective = prob.evaluate(cand)
                            obj_at = _rollout_checkpoints(prob, cand, actions, rng, horizons)
                            for h in horizons:
                                ret[h].append((s.objective - obj_at[h]) / base)
                        rows.append({
                            "instance": name, "seed": seed, "iter": it, **ctx, **struct, **inst_feats,
                            **op_descriptor(action),
                            **{f"return_{h}": float(np.mean(ret[h])) for h in horizons},
                        })
                # advance the walk one cheap step
                act = actions[int(rng.integers(len(actions)))]
                c = act.operator.apply(st.incumbent, rng, **act.params)
                c.objective = prob.evaluate(c)
                st.n_evals += 1
                if acc.accept(st, c, rng):
                    st.incumbent = c
                if c.objective < st.best.objective:
                    st.best = c
                acc.step(st)
                st.iteration += 1
            print(f"  rolled out {name} seed={seed}")
    return pd.DataFrame(rows)


def within_instance_regret(df, target, use_onehot, cols, kind):
    """Leave-one-seed-out within instance; regret of context-aware vs context-free
    operator pick on live contexts, using `target` as the reward signal."""
    reg_ctx, reg_cf = [], []
    for inst in df.instance.unique():
        di = df[df.instance == inst]
        for ts in di.seed.unique():
            tr, te = di[di.seed != ts], di[di.seed == ts].copy()
            Xtr = _build_X(tr, use_onehot, cols)
            Xte = _build_X(te, use_onehot, cols).reindex(columns=Xtr.columns, fill_value=0)
            m = _make_model(kind); m.fit(Xtr, tr[target].values)
            te["pred"] = m.predict(Xte)
            cf_op = tr.groupby("op_name")[target].mean().idxmax()
            for _, g in te.groupby("iter"):
                if g[target].max() <= 0:
                    continue
                oracle = g[target].max()
                reg_ctx.append(oracle - g.loc[g["pred"].idxmax(), target])
                cf = g[g.op_name == cf_op][target]
                reg_cf.append(oracle - (cf.mean() if len(cf) else np.nan))
    reg_ctx, reg_cf = np.array(reg_ctx), np.array(reg_cf)
    try:
        p = wilcoxon(reg_ctx, reg_cf).pvalue
    except ValueError:
        p = 1.0
    return reg_ctx.mean(), reg_cf.mean(), reg_cf.mean() - reg_ctx.mean(), p


def main(horizons=(1, 3, 5, 8), R=2, n_states=100, n_iters=200, seeds=None, instances=None):
    names = instances or DEFAULT_INSTANCES
    seeds = seeds if seeds is not None else [0, 1, 2]
    horizons = list(horizons)
    insts = load_cflp_benchmarks(names)
    print(f"lookahead: {len(insts)} inst x {len(seeds)} seeds, ~{n_states} states, "
          f"horizons={horizons}, R={R}")
    df = generate_lookahead_log(insts, seeds, n_iters, n_states, horizons, R)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "lookahead_log.csv"), index=False)
    print(f"logged {len(df)} rows -> {OUT}/lookahead_log.csv\n")

    # does context (op+B) help MORE as the horizon grows?
    print("within-instance regret reduction (context helps if reduction>0), GBDT, op+B:\n")
    print(f"{'horizon':8s} {'regret_ctx':>10s} {'regret_cf':>10s} {'reduction':>10s} {'p':>8s}")
    out_rows = []
    for h in horizons:
        rc, rf, red, p = within_instance_regret(df, f"return_{h}", True, B_STRUCTURE, "lgbm")
        print(f"{h:<8d} {rc:10.5f} {rf:10.5f} {red:+10.5f} {p:8.4f}")
        out_rows.append({"horizon": h, "regret_ctx": rc, "regret_cf": rf,
                         "reduction": red, "wilcoxon_p": p})
    pd.DataFrame(out_rows).to_csv(os.path.join(OUT, "lookahead_regret_by_horizon.csv"), index=False)
    print(f"\nwrote {OUT}/lookahead_regret_by_horizon.csv")


if __name__ == "__main__":
    a = sys.argv[1:]
    n_states = int(a[a.index("--states") + 1]) if "--states" in a else 100
    n_seeds = int(a[a.index("--seeds") + 1]) if "--seeds" in a else 3
    main(n_states=n_states, seeds=list(range(n_seeds)))
