"""Offline diagnostic: does context carry *operator-selection* signal on CFLP?

Two hypotheses for why context features didn't help the online controller:
  (H1) the features carry no usable signal ("wrong features"), or
  (H2) the signal exists but is masked online by exploration/gating.

This script separates them, and in the SAME run also tests whether richer CFLP
*structure* features would help. It does NOT touch PROJECT_LOG / git - it is an
exploratory tool.

METHOD
  Data: for each (instance, seed), run an exploratory host; at each iteration probe
  EVERY operator once from the current incumbent and log (context + structure +
  instance + operator descriptors, realized relative-Delta reward). Probing all
  operators at the same state gives perfectly paired data and lets us measure the
  decision-relevant quantity offline: operator-selection REGRET.

  Analysis (leave-instance-out CV): for each feature set, fit a model of reward,
  then per held-out context compare the realized reward of
    - context-free pick   = argmax global mean-operator (what greedy-mean does)
    - context-aware pick   = argmax model prediction given the features
    - oracle               = best operator actually available
  Regret = oracle - pick. If context-aware regret << context-free regret, the
  features carry selection signal (=> H2 masking); if ~equal, they don't (=> H1).

  Feature sets compared: op-identity only; +A (current progress context);
  +B (CFLP structure); +A+B; +A+B+C (instance); and operator-descriptors instead
  of one-hot identity. Run with both a GBDT and a linear model to tell "features"
  apart from "model too weak".

    python scripts/context_diagnostic.py                 # default: 6 inst x 2 seeds x 150 iters, SA
    python scripts/context_diagnostic.py --host ALNS --iters 100 --seeds 3
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from serene_mh.core import build_actions, Greedy, SimulatedAnnealing, Tabu, RecordToRecord
from serene_mh.core.engine import SearchState, relative_improvement_reward
from serene_mh.problems import load_cflp_benchmarks
from serene_mh.problems.cflp import _facility_load

warnings.filterwarnings("ignore")

# ---- defaults (all overridable) -------------------------------------------------
DEFAULT_INSTANCES = ["cap41", "cap71", "cap81", "cap101", "cap111", "cap131"]  # 2 per tier
DEFAULT_HOST = "SA"
DEFAULT_SEEDS = [0, 1]
DEFAULT_ITERS = 150
OUT = "results/tables/cflp_context_diagnostic"

# ---- feature groups (column names) ----------------------------------------------
A_PROGRESS = ["frac_budget", "recent_improvement", "accept_rate", "stagnation", "incumbent_gap"]
B_STRUCTURE = ["frac_open", "capacity_slack", "feasibility_margin", "util_mean", "util_cv",
               "util_min", "frac_util_below_50", "frac_util_below_25", "serving_headroom",
               "fixed_share", "mean_open_fixed"]
C_INSTANCE = ["inst_log_m", "inst_n_over_m", "inst_tightness", "inst_cv_fixed", "inst_cv_cap",
              "inst_cv_demand", "inst_fixed_to_transport", "inst_uc_diff"]
D_DESC = ["op_is_greedy", "op_dir", "op_param"]

_GREEDY = {"greedy_drop", "greedy_open", "greedy_swap", "destroy_repair"}
_DIR = {"close": -1, "greedy_drop": -1, "open": 1, "greedy_open": 1,
        "swap": 0, "greedy_swap": 0, "multiflip": 0, "destroy_repair": 0}


def _host(name, max_evals):
    return {"LS": Greedy(), "SA": SimulatedAnnealing(max_evals=max_evals),
            "Tabu": Tabu(tenure=20), "ALNS": RecordToRecord(deviation=0.02)}[name]


def _cv(x):
    x = np.asarray(x, dtype=float)
    return float(x.std() / (abs(x.mean()) + 1e-12))


def instance_features(prob):
    f, cap, d, uc = prob.fixed_cost, prob.capacity.astype(float), prob.demand.astype(float), prob.unit_cost
    return {
        "inst_log_m": float(np.log(prob.m)),
        "inst_n_over_m": prob.n / prob.m,
        "inst_tightness": float(cap.sum() / d.sum()),
        "inst_cv_fixed": _cv(f), "inst_cv_cap": _cv(cap), "inst_cv_demand": _cv(d),
        "inst_fixed_to_transport": float(f.mean() / (uc.mean() * d.mean() + 1e-12)),
        "inst_uc_diff": float(np.mean(uc.std(axis=0) / (uc.mean(axis=0) + 1e-12))),
    }


def structure_features(prob, mask, allopen_serving):
    m = prob.m
    n_open = int(mask.sum())
    open_cap = float(prob.capacity[mask].sum())
    total_d = float(prob.demand.sum())
    loads = _facility_load(mask, prob.demand, prob.unit_cost)
    caps = prob.capacity[mask].astype(float)
    util = loads[mask] / np.maximum(caps, 1e-9) if n_open else np.array([0.0])
    best = prob.unit_cost[mask].min(axis=0) if mask.any() else prob.unit_cost.max(axis=0)
    serving = float((best * prob.demand).sum())
    open_fixed = float(prob.fixed_cost[mask].sum())
    return {
        "frac_open": n_open / m,
        "capacity_slack": open_cap / total_d,
        "feasibility_margin": (open_cap - total_d) / total_d,
        "util_mean": float(util.mean()), "util_cv": _cv(util), "util_min": float(util.min()),
        "frac_util_below_50": float((util < 0.5).mean()), "frac_util_below_25": float((util < 0.25).mean()),
        "serving_headroom": (serving - allopen_serving) / max(serving, 1e-9),
        "fixed_share": open_fixed / max(open_fixed + serving, 1e-9),
        "mean_open_fixed": (open_fixed / max(n_open, 1)) / (prob.fixed_cost.mean() + 1e-9),
        "raw_n_open": n_open, "raw_open_cap": open_cap, "raw_total_demand": total_d,
        "raw_open_fixed": open_fixed, "raw_serving_proxy": serving,
    }


def op_descriptor(action):
    name = action.operator.name
    param = action.params.get("k", action.params.get("q", 0))
    return {"op_name": name, "op_is_greedy": int(name in _GREEDY), "op_dir": _DIR.get(name, 0),
            "op_param": param}


# ---- data generation (probe all operators each step) ----------------------------
def generate_log(instances, host, seeds, n_iters):
    rows = []
    for name, prob in instances.items():
        actions = build_actions(prob.operators())
        max_evals = len(actions) * n_iters
        allopen_serving = float((prob.unit_cost.min(axis=0) * prob.demand).sum())
        inst_feats = instance_features(prob)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            acc = _host(host, max_evals)
            inc = prob.initial_solution(rng)
            inc.objective = prob.evaluate(inc)
            st = SearchState(problem=prob, incumbent=inc, best=inc, n_evals=1, max_evals=max_evals)
            acc.reset()
            for it in range(n_iters):
                ctx = st.telemetry()
                struct = structure_features(prob, st.incumbent.data, allopen_serving)
                start = st.incumbent
                cands, iter_rows = [], []
                for action in actions:
                    cand = action.operator.apply(start, rng, **action.params)
                    cand.objective = prob.evaluate(cand)
                    st.n_evals += 1
                    r = relative_improvement_reward(start, cand)
                    iter_rows.append({
                        "instance": name, "seed": seed, "iter": it, **ctx, **struct, **inst_feats,
                        **op_descriptor(action),
                        "delta_n_open": int(cand.data.sum() - start.data.sum()),
                        "n_changed": int((cand.data != start.data).sum()),
                        "reward": r, "cand_obj": cand.objective, "incumbent_obj": start.objective,
                    })
                    cands.append(cand)
                rows.extend(iter_rows)
                best_cand = min(cands, key=lambda c: c.objective)
                accepted = acc.accept(st, best_cand, rng)
                if accepted:
                    st.incumbent = best_cand
                improved = best_cand.objective < st.best.objective
                if improved:
                    st.best = best_cand
                st.stagnation = 0 if improved else st.stagnation + 1
                st.recent_rewards.append(max(x["reward"] for x in iter_rows))
                st.recent_accepts.append(1 if accepted else 0)
                acc.step(st)
                st.iteration += 1
            print(f"  logged {name} seed={seed}")
    return pd.DataFrame(rows)


# ---- offline analysis (leave-instance-out selection regret) ---------------------
def _make_model(kind):
    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             min_child_samples=30, verbose=-1)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))


def _build_X(df, use_onehot, cols):
    parts = []
    if use_onehot:
        parts.append(pd.get_dummies(df["op_name"], prefix="op"))
    if cols:
        parts.append(df[cols].reset_index(drop=True))
    X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    return X


def evaluate_spec(df, use_onehot, cols, model_kind):
    from scipy.stats import wilcoxon
    reg_ctx, reg_cf = [], []
    for test in df["instance"].unique():
        tr, te = df[df.instance != test], df[df.instance == test].copy()
        Xtr = _build_X(tr, use_onehot, cols)
        Xte = _build_X(te, use_onehot, cols).reindex(columns=Xtr.columns, fill_value=0)
        model = _make_model(model_kind)
        model.fit(Xtr, tr["reward"].values)
        te["pred"] = model.predict(Xte)
        cf_op = tr.groupby("op_name")["reward"].mean().idxmax()  # context-free = best global mean op
        for _, g in te.groupby(["seed", "iter"]):
            oracle = g["reward"].max()
            ctx_pick = g.loc[g["pred"].idxmax(), "reward"]
            cf_rows = g[g["op_name"] == cf_op]["reward"]
            cf_pick = cf_rows.mean() if len(cf_rows) else np.nan
            reg_ctx.append(oracle - ctx_pick)
            reg_cf.append(oracle - cf_pick)
    reg_ctx, reg_cf = np.array(reg_ctx), np.array(reg_cf)
    try:
        p = wilcoxon(reg_ctx, reg_cf).pvalue
    except ValueError:
        p = 1.0
    return {"mean_regret_ctx": reg_ctx.mean(), "mean_regret_cf": reg_cf.mean(),
            "regret_reduction": reg_cf.mean() - reg_ctx.mean(), "wilcoxon_p": p}


def analyze(df, out):
    specs = {
        "op_only":        (True, []),
        "op+A":           (True, A_PROGRESS),
        "op+B":           (True, B_STRUCTURE),
        "op+A+B":         (True, A_PROGRESS + B_STRUCTURE),
        "op+A+B+C":       (True, A_PROGRESS + B_STRUCTURE + C_INSTANCE),
        "descriptors+A+B": (False, D_DESC + A_PROGRESS + B_STRUCTURE),
    }
    results = []
    for model_kind in ("lgbm", "linear"):
        for sname, (use_onehot, cols) in specs.items():
            r = evaluate_spec(df, use_onehot, cols, model_kind)
            results.append({"model": model_kind, "features": sname, **r})
            print(f"  {model_kind:6s} {sname:16s} "
                  f"regret ctx={r['mean_regret_ctx']:.4f} cf={r['mean_regret_cf']:.4f} "
                  f"reduction={r['regret_reduction']:+.4f} p={r['wilcoxon_p']:.4f}")
    res = pd.DataFrame(results)
    os.makedirs(out, exist_ok=True)
    res.to_csv(os.path.join(out, "regret_by_featureset.csv"), index=False)
    return res


def main(instances=None, host=DEFAULT_HOST, seeds=None, n_iters=DEFAULT_ITERS):
    names = instances or DEFAULT_INSTANCES
    seeds = seeds or DEFAULT_SEEDS
    insts = load_cflp_benchmarks(names)
    print(f"context diagnostic: {len(insts)} instances x {len(seeds)} seeds x {n_iters} iters, host={host}")
    df = generate_log(insts, host, seeds, n_iters)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "probe_log.csv"), index=False)
    print(f"logged {len(df)} (context, operator) tuples -> {OUT}/probe_log.csv\n")
    analyze(df, OUT)
    print(f"\nwrote analysis to {OUT}/regret_by_featureset.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    host = args[args.index("--host") + 1] if "--host" in args else DEFAULT_HOST
    n_iters = int(args[args.index("--iters") + 1]) if "--iters" in args else DEFAULT_ITERS
    seeds = list(range(int(args[args.index("--seeds") + 1]))) if "--seeds" in args else DEFAULT_SEEDS
    main(host=host, seeds=seeds, n_iters=n_iters)
