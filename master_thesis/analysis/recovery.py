"""
Recovery (README § Recovery): synthetic groups of R runs from the surrogate, the models
fitted and compared on them as on the runs, and the tests the runs' analysis shares.
recovery.ipynb sets the ranges and reads the results; this module computes the ranges'
profiles, one dataset per SCC array task, the merge and the readings (schedule.md
§ Recovery). No LLM import.

    python analysis/recovery.py task <id> <settings.json> <audit dir> <out dir>
    python analysis/recovery.py merge <out dir> <settings.json> <datasets.jsonl>
    python analysis/recovery.py selftest
"""

from __future__ import annotations

import csv
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import scipy

import fit
import surrogate
from config import NETWORK_MAX_ROUNDS
from metrics import Z95, audit_declarations, audit_units, signatures, wilson
from surrogate import LIMIT, Pool

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "analysis" / "logs" / "recovery"

LEVEL = 0.0125          # one-sided, H1 and H2 each (README § Scope)
LEVEL_COUPLING = 0.025  # one-sided, H3
# The models: each generates data, each is a candidate in model recovery and has a held-out
# score; the outcome tests replicate M1 and M3.
MODELS = tuple(fit.MODELS)
REPLICATED = ("M1", "M3")

# Step 1's ranges: a rate log-uniform from the one whose half-life is the entries an agent
# holds at H, two per round, to 1; a weight, profiled alone, log-uniform from its onset to its
# saturation, since behaviour answers it over orders of magnitude (``saturation``). "Moves" is
# by more than SATURATED of a summary's scale on one of the SUMMARIES (the audit's sizing rule).
RECORD_ENTRIES = 2 * NETWORK_MAX_ROUNDS
RATE_LOW = 1 - 0.5 ** (1 / RECORD_ENTRIES)
RATES = {"a_K": "gamma_K", "a": "beta"}   # each rate and the weight it scales
WEIGHTS = tuple(RATES.values())
LOG_DRAWN = (*RATES, *WEIGHTS)   # drawn log-uniform, so recovery reads them on the log scale
SUMMARIES = {"mean": 2.0, "changes": 1.0, "divergence": 1.0}  # scales
SATURATED = 0.05
DOUBLINGS = 12      # a guard, not a setting: a weight inert or still moving 2^12 from 1 aborts
# Cells recovery leaves out, by name: one whose saturation aborts (a weight still moving
# behaviour at 2^DOUBLINGS) is named here, with the reason. None is.
SKIPPED: tuple[str, ...] = ()
STEPS = (1, 2, 4)   # the nested R: every run, every 2nd, every 4th (33, 17, 9)
HALVINGS = 4        # n_rep read at n_rep, / 2, / 4, / 8 (40, 20, 10, 5)
Z_LEVEL = NormalDist().inv_cdf(1 - LEVEL)


def grid(R: int) -> np.ndarray:
    """One group's p_inter: R evenly spaced over [0, 0.5], ends included."""
    return np.linspace(0, 0.5, R)


def dataset(rule, R: int, pool: Pool, rounds: int, seed: int) -> dict:
    """One synthetic group: R runs, one per p_inter of the grid, each from its own seed."""
    runs = []
    for j, p_inter in enumerate(grid(R)):
        s = seed * R + j
        graph, starts = surrogate.draw_setup(p_inter, s)
        out = surrogate.run(rule, graph, starts, pool, rounds, s, record=True)
        runs.append({"p_inter": p_inter, "seed": s, "graph": graph, "starts": starts, **out})
    return {"rule": rule, "runs": runs, "decls": [d for r in runs for d in r["decls"]]}


def nested(ds: dict, step: int) -> dict:
    """Every ``step``-th run of ``ds``: the grid at (R - 1) / step + 1."""
    runs = ds["runs"][::step]
    keep = {r["seed"] for r in runs}
    return {"rule": ds["rule"], "runs": runs, "decls": [d for d in ds["decls"] if d["run"] in keep]}


# -- The tests ----------------------------------------------------------------

def declaration_stats(ll: dict, data: dict, mask=None) -> dict:
    """The held-out gains per run from ``ll``, over ``mask``: d(M3:M1) (H1); d(M1:M0) and
    d(M2:M0), descriptive."""
    g = lambda a, b: fit.run_gains(ll[a], ll[b], data, mask)  # noqa: E731
    return {"H1": g("M3", "M1"), "M1:M0": g("M1", "M0"), "M2:M0": g("M2", "M0")}


def declarations(data: dict, seed: int = 0, **fit_kw) -> dict:
    """``declaration_stats`` on the held-out scores, and each model's fold fits by run."""
    ll, fits = {}, {}
    for m in MODELS:
        ll[m], fits[m] = fit.heldout(data, m, seed, **fit_kw)
    return declaration_stats(ll, data) | {"fits": fits, "ll": ll}


def declaration_p(stats: dict, seed: int = 0) -> dict:
    return {"H1": fit.sign_flip(stats["H1"], seed)}


def replicate_values(ds: dict, fits: dict, pool, rounds: int, n_rep: int, seed: int = 0, *,
                     state: str = "final") -> dict:
    """Per outcome a run names (its ``values``, else ``divergence``), model and run, the pair
    (its ``n_rep`` replicates' values, the run's), all from the model's fit on the other runs:
    ``fits``, the declaration tests' folds (``declarations``). ``pool``: one, or one per run by its seed.
    ``state``: the state ``surrogate.run`` reads (schedule.md § Sensitivity)."""
    k: dict = {}
    for j, r in enumerate(ds["runs"]):
        replay = pool[r["seed"]] if isinstance(pool, dict) else pool
        observed = r.get("values", {"divergence": r["divergence"]})
        for m in REPLICATED:
            rule = fits[m][r["seed"]]
            reps = []
            for i in range(n_rep):
                s = (seed + j) * n_rep + i
                out = surrogate.run(rule, r["graph"], r["starts"], replay, rounds, s)
                reps.append(surrogate.run_values(out, r["graph"], state))
            for o, y in observed.items():
                k.setdefault(o, {}).setdefault(m, []).append((np.array([v[o] for v in reps]), y))
    return k


def outcome_stats(k: dict, score=fit.crps_score) -> dict:
    """Per run, the ``score`` of its outcome under M1 and M3 from their (replicates, observed)
    pairs, and the gain s(M3) - s(M1) (H2)."""
    s = {m: np.array([score(x, y) for x, y in v]) for m, v in k.items()}
    return {"scores": s, "H2": s["M3"] - s["M1"]}


def outcome_p(stats: dict, seed: int = 0) -> dict:
    return {"H2": fit.sign_flip(stats["H2"], seed)}


def outcomes(ds: dict, fits: dict, pool: Pool, rounds: int, n_rep: int, seed: int = 0) -> dict:
    """The outcome statistics on D, from the declaration tests' fold ``fits``."""
    return outcome_stats(replicate_values(ds, fits, pool, rounds, n_rep, seed)["divergence"])


def analyse(ds: dict, seed: int = 0, outcome_args: dict | None = None) -> dict:
    """Every per-run statistic and p-value of one group, as the runs' analysis computes
    them; the outcome tests only with ``outcome_args`` (pool, rounds, n_rep)."""
    data = fit.arrays(ds["decls"])
    stats = declarations(data, seed)
    p_inter = np.array([r["p_inter"] for r in ds["runs"]])
    d = np.array([r["divergence"] for r in ds["runs"]])
    p = declaration_p(stats, seed) | {"H3": fit.coupling_test(p_inter, d, seed=seed)[1]}
    if outcome_args:
        out = outcomes(ds, stats["fits"], seed=seed, **outcome_args)
        stats["outcome_scores"] = out.pop("scores")
        stats.update(out)
        p.update(outcome_p(out, seed))
    return {"stats": stats, "p": p, "decisions": decide(p), "p_inter": p_inter, "divergence": d}


def rejects(p: float, level: float = LEVEL) -> bool:
    return p <= level


def decide(p: dict) -> dict[str, bool]:
    """The rejections: H1 and H2, each where present, at LEVEL; H3 at LEVEL_COUPLING."""
    return {h: rejects(p[h], LEVEL_COUPLING if h == "H3" else LEVEL)
            for h in ("H1", "H2", "H3") if h in p}


# -- The checks ---------------------------------------------------------------

def draw_rule(model: str, ranges: dict, rng: np.random.Generator) -> dict:
    """A rule of ``model`` from ``ranges`` ({parameter: value or (lo, hi)}): a value fixed,
    a range (a rate's or a weight's) log-uniform; the name intercepts ``ranges`` holds."""
    def draw(v):
        return float(np.exp(rng.uniform(*np.log(v)))) if isinstance(v, (list, tuple)) else float(v)
    keys = fit.param_names(model) + [k for k in fit.NAME_TERMS if k in ranges]
    return {"model": model, "params": {k: draw(ranges[k]) for k in keys}}


def baseline(cell_dir: Path | str, seed: int = 0) -> dict:
    """b, κ and the name intercepts, which every range fixes: M0 fitted to the audit's
    empty-memory declarations, which both conditions share."""
    decls = audit_declarations(audit_units(cell_dir, "meetings"), "empty")
    return fit.fit(fit.arrays(decls), "M0", seed)["params"]


def behaviour(rule, pool: Pool, R: int, rounds: int, seed: int) -> dict[str, float]:
    """What one dataset at ``rule`` does: the declarations' mean, how often an agent
    changes pole, the mean D."""
    ds = dataset(rule, R, pool, rounds, seed)
    y = np.array([d["y"] for d in ds["decls"]])
    by_agent: dict = {}
    for d in ds["decls"]:
        by_agent.setdefault((d["run"], d["agent"]), []).append(d["y"])
    changes = [np.mean(np.diff(v) != 0) for v in by_agent.values() if len(v) > 1]
    return {"mean": float(y.mean()), "changes": float(np.mean(changes)),
            "divergence": float(np.mean([r["divergence"] for r in ds["runs"]]))}


def profile(base: dict, name: str, values, pool: Pool, R: int, rounds: int, seed: int = 0) -> list[dict]:
    """``behaviour`` with one parameter of ``base`` at each of ``values``, the same seed throughout."""
    return [{"model": base["model"], "parameter": name, "value": float(v),
             **behaviour({**base, "params": {**base["params"], name: float(v)}}, pool, R, rounds, seed)}
            for v in values]


def saturation(base: dict, name: str, pool: Pool, R: int, rounds: int,
               seed: int = 0) -> tuple[float, float, list[dict]]:
    """A weight's onset and saturation, and the profile behind them, by powers of 2 from 1. The
    onset is the largest value that does not move behaviour from the weight at 0; the
    saturation, the first value past it whose doubling moves nothing, so a stretch where the
    weight has not yet acted is no plateau."""
    done: dict[float, dict] = {}

    def at(v: float) -> dict:
        if v not in done:
            done[v] = profile(base, name, (v,), pool, R, rounds, seed)[0]
        return done[v]

    def moves(a: dict, b: dict) -> bool:
        return any(abs(a[k] - b[k]) > SATURATED * s for k, s in SUMMARIES.items())

    zero = at(0.0)
    up = not moves(zero, at(1.0))       # 1 still acts like 0: the onset lies above, else below
    v = 1.0
    while moves(zero, at(v)) != up:
        v = v * 2 if up else v / 2
        if abs(math.log2(v)) > DOUBLINGS:
            raise ValueError(f"{base['model']} {name}: no onset within 2^±{DOUBLINGS}")
    onset = v / 2 if up else v
    acted = at(2 * onset)
    while moves(acted, at(2 * acted["value"])):
        acted = at(2 * acted["value"])
        if acted["value"] > 2 ** DOUBLINGS:
            raise ValueError(f"{base['model']} {name} still moves at {acted['value']}")
    return onset, acted["value"], sorted(done.values(), key=lambda r: r["value"])


def cell_ranges(fixed: dict, pool: Pool, R: int, rounds: int,
                seed: int = 0) -> tuple[dict, list[dict]]:
    """Step 1 on one cell: every generating model's ranges and the profiles behind them. Each
    weight is profiled alone, the others at 0, since a strong habit freezes the agents and
    hides the partner: γ_K in M1, β in M2; rates at their geometric middle. ``fixed``:
    ``baseline``'s b, κ and name intercepts."""
    mid = dict.fromkeys(RATES, math.sqrt(RATE_LOW))
    names = [k for k in fit.NAME_TERMS if k in fixed]

    def base(model: str) -> dict:
        p = {**fixed, **mid, **dict.fromkeys(WEIGHTS, 0.0)}
        return {"model": model, "params": {k: p[k] for k in fit.param_names(model) + names}}

    span, rows = {}, []
    for m, w in (("M1", "gamma_K"), ("M2", "beta")):
        onset, top, r = saturation(base(m), w, pool, R, rounds, seed)
        span[w] = (onset, top)
        rows += r

    def spec(k: str):
        return fixed[k] if k in fixed else (RATE_LOW, 1.0) if k in RATES else span[k]
    return {m: {k: spec(k) for k in fit.param_names(m) + names} for m in MODELS}, rows


def choose_model(data: dict, seed: int = 0) -> dict:
    """Pooled fits of the models and one leave-one-run-out pass of each, every fit from random
    starts as on the runs: per-declaration held-out log-likelihoods ``ll``, each fold's fits by
    run, the summed scores and the model best on each criterion."""
    pooled = {m: fit.fit(data, m, seed) for m in MODELS}
    ll = {m: np.empty(len(data["y"])) for m in MODELS}
    fits: dict = {m: {} for m in MODELS}
    for r in np.unique(data["run"]):
        out = data["run"] == r
        train, test = fit.subset(data, ~out), fit.subset(data, out)
        for m in MODELS:
            x = fit.fit(train, m, seed)
            ll[m][out] = fit.loglik(x["params"], test, m)
            fits[m][r] = x
    scores = {m: float(ll[m].sum()) for m in MODELS}
    bic = {m: f["bic"] for m, f in pooled.items()}
    return {"pooled": pooled, "ll": ll, "fits": fits, "scores": scores, "bic": bic,
            "chosen": max(scores, key=scores.get), "chosen_bic": min(bic, key=bic.get)}


# -- The SCC array ------------------------------------------------------------
# One task per dataset (schedule.md § Recovery; RUNBOOK § Recovery on the CPUs).

def write_settings(path: Path | str, cells: dict, rounds: int, R: int, n_rep: int, n: int) -> dict:
    """The array's one input: per cell its ``cell_ranges``, H, R, n_rep and the
    datasets ``n`` per cell and generating model, beside the commit."""
    if (R - 1) % STEPS[-1] or n_rep % 2 ** (HALVINGS - 1) or n_rep >> (HALVINGS - 1) < 2:
        raise ValueError(f"R - 1 must divide by {STEPS[-1]}, n_rep by {2 ** (HALVINGS - 1)} "
                         "and its smallest half hold two replicates, the fair CRPS's least")
    settings = {"commit": _git_commit(), "rounds": rounds, "R": R, "n_rep": n_rep, "n": n,
                "cells": cells}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings


def n_tasks(settings: dict) -> int:
    return settings["n"] * len(settings["cells"]) * len(MODELS)


def task(i: int, settings: dict, pool_of) -> dict:
    """Dataset ``i``: its cell and generating model from ``i`` modulo cells × models, so the
    first cells × models ids hold each pair once; at each nested R the model choice, the generating
    model's pooled fit, the per-run gains and every decision at each n_rep. ``pool_of``:
    cell -> Pool."""
    cells, k = sorted(settings["cells"]), len(MODELS)
    cell, true = cells[i % (len(cells) * k) // k], MODELS[i % k]
    ranges = settings["cells"][cell]["ranges"][true]
    pool, rounds, R, n_rep = pool_of(cell), settings["rounds"], settings["R"], settings["n_rep"]
    rule = draw_rule(true, ranges, np.random.default_rng(i))
    ds = dataset(rule, R, pool, rounds, i)
    row = {"id": i, "cell": cell, "true": true, "params": rule["params"],
           "divergence": [r["divergence"] for r in ds["runs"]], "node": platform.node(),
           "versions": {"numpy": np.__version__, "scipy": scipy.__version__},  # the SCC's, not the pins
           "commit": _git_commit()}
    for step in STEPS:
        sub = nested(ds, step)
        data = fit.arrays(sub["decls"])
        res = choose_model(data, i)
        stats = declaration_stats(res["ll"], data)
        p_inter, d = (np.array([r[x] for r in sub["runs"]]) for x in ("p_inter", "divergence"))
        p = declaration_p(stats, i) | {"H3": fit.coupling_test(p_inter, d, seed=i)[1]}
        reps = replicate_values(sub, res["fits"], pool, rounds, n_rep, i * R)["divergence"]
        gains = {h: stats[h].tolist() for h in ("H1", "M1:M0", "M2:M0")}
        decisions = {}
        for n in (n_rep >> h for h in range(HALVINGS)):
            out = outcome_stats({m: [(x[:n], y) for x, y in v] for m, v in reps.items()})
            decisions[n] = decide(p | outcome_p(out, i))
            if n == n_rep:
                gains["H2"] = out["H2"].tolist()
        fitted = res["pooled"][true]
        row[f"R{len(sub['runs'])}"] = {
            "fitted": fitted["params"], "at_bound": fitted["at_bound"], "scores": res["scores"],
            "bic": res["bic"], "chosen": res["chosen"], "chosen_bic": res["chosen_bic"],
            "gains": gains, "decisions": decisions}
    return row


def merge(out_dir: Path | str, settings_path: Path | str, path: Path | str) -> list[int]:
    """The array's rows into one JSONL at ``path`` by id, the settings as its sidecar. Returns
    the missing ids and writes nothing while any is; aborts on a duplicate id or a second
    commit."""
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    rows: dict = {}
    for f in Path(out_dir).glob("task_*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["id"] in rows:
            raise ValueError(f"id {r['id']} twice")
        rows[r["id"]] = r
    missing = sorted(set(range(n_tasks(settings))) - set(rows))
    if missing:
        return missing
    commits = {r["commit"] for r in rows.values()}
    if len(commits) != 1:
        raise ValueError(f"rows from several commits: {sorted(commits)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(rows[i]) + "\n" for i in sorted(rows)), encoding="utf-8")
    path.with_suffix(path.suffix + ".params.json").write_text(
        json.dumps({**settings, "commit": commits.pop()}, indent=2), encoding="utf-8")
    return []


def read_rows(path: Path | str) -> list[dict]:
    """``merge``'s rows; none before it has run."""
    path = Path(path)
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


# -- The readings -------------------------------------------------------------

def sizes(row: dict) -> list[int]:
    """The nested R a row holds."""
    return sorted(int(k[1:]) for k in row if re.fullmatch(r"R\d+", k))


def parameter_rows(rows: list[dict], R: int) -> list[dict]:
    """Step 3's plot at ``R``: one row per dataset and parameter, its true and recovered value."""
    return [{"true_model": r["true"], "parameter": k, "cell": r["cell"], "simulated": v,
             "recovered": r[f"R{R}"]["fitted"][k]} for r in rows for k, v in r["params"].items()]


def _corr(a, b) -> float:
    """Pearson's r, NaN where either side does not vary."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else math.nan


def _logs(*cols) -> tuple[np.ndarray, ...]:
    """The columns' logs over the rows where every value is positive."""
    cols = [np.asarray(c, float) for c in cols]
    ok = np.logical_and.reduce([c > 0 for c in cols])
    return tuple(np.log(c[ok]) for c in cols)


def parameter_recovery(rows: list[dict], R: int) -> list[dict]:
    """Per cell, model and parameter at ``R``, simulated against recovered: r and mean bias,
    for a ``LOG_DRAWN`` parameter on the log scale (bias a mean log ratio) over the fits
    recovered above 0, the others counted in ``nonpositive``. b, κ and the name intercepts,
    fixed per cell, have bias alone, on their own scale."""
    long = parameter_rows(rows, R)
    out = []
    for c, m, k in sorted({(x["cell"], x["true_model"], x["parameter"]) for x in long}):
        pairs = [(x["simulated"], x["recovered"]) for x in long
                 if (x["cell"], x["true_model"], x["parameter"]) == (c, m, k)]
        sim, rec = np.array(pairs).T
        n, log = len(sim), k in LOG_DRAWN
        if log:
            sim, rec = _logs(sim, rec)
        out.append({"cell": c, "true_model": m, "parameter": k, "scale": "log" if log else "raw",
                    "n": n, "nonpositive": n - len(sim), "r": _corr(sim, rec),
                    "bias": float((rec - sim).mean()) if len(sim) else math.nan})
    return out


def induced_correlations(rows: list[dict], R: int) -> list[dict]:
    """Per cell, model and pair of drawn parameters at ``R``, r among the recovered values
    beside r among the simulated ones, on the log scale they are drawn on, the recovered
    over the fits with both above 0: what the fit adds to the draws."""
    out = []
    for c, m in sorted({(r["cell"], r["true"]) for r in rows}):
        of_m = [r for r in rows if (r["cell"], r["true"]) == (c, m)]
        names = [k for k in of_m[0]["params"] if len({r["params"][k] for r in of_m}) > 1]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                fitted = _logs(*zip(*[(r[f"R{R}"]["fitted"][a], r[f"R{R}"]["fitted"][b]) for r in of_m]))
                out.append({"cell": c, "true_model": m, "a": a, "b": b,
                            "simulated": _corr(*_logs(*zip(*[(r["params"][a], r["params"][b]) for r in of_m]))),
                            "recovered": _corr(*fitted), "n": len(fitted[0])})
    return out


def truth(model: str, test: str) -> bool:
    """Whether ``test``'s hypothesis holds under data from ``model`` (README § Recovery, What it
    speaks to): a rejection is power where True, a false rejection where False. H1, H2
    and H3 all hold where feedback acts, under M2 and M3."""
    return model in ("M2", "M3")


def rejection_rows(rows: list[dict]) -> list[dict]:
    """One row per dataset, R, n_rep and test: whether it rejects and whether
    its hypothesis holds (``truth``)."""
    return [{"true": r["true"], "cell": r["cell"], "R": R, "n_rep": int(n), "test": h, "rejects": v,
             "holds": truth(r["true"], h)}
            for r in rows for R in sizes(r) for n, dec in r[f"R{R}"]["decisions"].items()
            for h, v in dec.items()]


def normal_power(gains: list, R: int) -> float:
    """Φ(√R · d̄/s − z_LEVEL) averaged over datasets, each a list of per-run gains: a test's
    power at ``R`` in the normal approximation; checked by predicting
    the largest R from the smaller ones."""
    z = [math.sqrt(R) * np.mean(g) / np.std(g, ddof=1) - Z_LEVEL for g in gains if np.std(g) > 0]
    return float(np.mean([NormalDist().cdf(x) for x in z])) if z else math.nan


def confusion(pairs: list[tuple[str, str]]) -> tuple[dict, dict]:
    """P(chosen | true) and its inversion P(true | chosen) from (true, chosen) pairs."""
    def table(rows):
        out = {}
        for a in sorted({r[0] for r in rows}):
            n = sum(r[0] == a for r in rows)
            out[a] = {b: sum(r == (a, b) for r in rows) / n for b in sorted({r[1] for r in rows})}
        return out
    return table(pairs), table([(c, t) for t, c in pairs])


def read_curve(path: Path | str) -> tuple[list[float], np.ndarray, np.ndarray, int]:
    """A benchmark ``_divergence.csv``: (p_inter values, D's mean and sd per round × value,
    runs per value)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    p_inters = sorted({float(r["p_inter"]) for r in rows})
    T = max(int(r["round"]) for r in rows)
    mean, sd = np.zeros((2, T, len(p_inters)))
    for r in rows:
        at = int(r["round"]) - 1, p_inters.index(float(r["p_inter"]))
        mean[at], sd[at] = float(r["mean"]), float(r["sd"])
    return p_inters, mean, sd, int(rows[0]["n_runs"])


def check_limit(bench: np.ndarray, sur: np.ndarray) -> dict:
    """The selftest's guard on the surrogate: per p_inter (rows), the Limit's and the
    symmetric benchmark's D at one round over their runs (columns), Welch's z at 5%; it
    fails if more p_inter fail than the upper 95% bound of Binomial(k, 0.05)."""
    diff = np.abs(bench.mean(1) - sur.mean(1))
    se = np.sqrt((bench.var(1, ddof=1) + sur.var(1, ddof=1)) / bench.shape[1])
    z = np.where(se > 0, diff / np.where(se > 0, se, 1), np.where(diff > 0, np.inf, 0))
    fails = int((z > Z95).sum())
    return {"fails": fails, "k": len(bench), "bound": binom_upper(len(bench)),
            "pass": fails <= binom_upper(len(bench))}


def binom_upper(k: int, p: float = 0.05, level: float = 0.95) -> int:
    """The smallest c with P(Binomial(k, p) <= c) >= level."""
    cdf, c = 0.0, 0
    while True:
        cdf += math.comb(k, c) * p ** c * (1 - p) ** (k - c)
        if cdf >= level:
            return c
        c += 1


def divergence_curve(rule, pool: Pool, seeds, rounds: int = surrogate.LONG_RUN_ROUNDS,
                     p_inters=None, jobs: int = 1, **kw) -> dict:
    """``rule``'s D̄(p_inter, t), with D's sd, to ``rounds`` on the benchmarks' grid ``p_inters``."""
    p_inters = np.round(np.arange(51) / 100, 2) if p_inters is None else np.asarray(p_inters)
    mean, sd = surrogate.divergence_moments(rule, pool, p_inters, seeds, rounds, jobs, **kw)
    return {"mean": mean, "sd": sd, "p_inters": p_inters, "n": len(seeds)}


# -- Output -------------------------------------------------------------------
# Each file named, with a sidecar (schedule.md § Recovery, Files).

def _git_commit() -> str:
    """HEAD, suffixed "-dirty" where tracked files outside analysis/logs/ differ from
    it: outputs, untracked or rewritten by a rerun, say nothing about the code."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    try:
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".",
                    ":(exclude)analysis/logs")
        return git("rev-parse", "HEAD") + ("-dirty" if dirty else "")
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def save(path: Path, rows: list[dict], params: dict) -> Path:
    """``rows`` as a CSV at ``path`` and ``<path>.params.json``: ``params`` and the commit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    path.with_suffix(path.suffix + ".params.json").write_text(
        json.dumps({"commit": _git_commit(), **params}, indent=2, default=str), encoding="utf-8")
    return path


def save_curve(path: Path, curve: dict, params: dict) -> None:
    """``divergence_curve``'s D̄(p_inter, t) at ``path`` in the benchmark's format, so
    ``read_curve`` reads it."""
    mean, sd = curve["mean"], curve["sd"]
    save(path, [{"p_inter": float(p_inter), "round": t + 1, "mean": round(float(mean[t, i]), 6),
                 "sd": round(float(sd[t, i]), 6), "n_runs": curve["n"]}
                for i, p_inter in enumerate(curve["p_inters"]) for t in range(len(mean))], params)


# -- Selftest -----------------------------------------------------------------

def selftest() -> None:
    from benchmark_sft import run_one

    # On a small grid: the Limit equals the symmetric benchmark in distribution.
    p_inters, seeds, T = (0.0, 0.25, 0.5), range(1, 121), 60
    bench = np.array([[run_one(p_inter, s, epsilon=0.1, both_learn=True, max_rounds=T)
                       ["divergence"] for s in seeds] for p_inter in p_inters])
    sur = np.array([[surrogate.run(LIMIT, *surrogate.draw_setup(p_inter, s), Pool.exact(), T, s)
                     ["divergence"] for s in seeds] for p_inter in p_inters])
    res = check_limit(bench, sur)
    assert res["pass"], (bench.mean(1), sur.mean(1), res)
    print(f"ok   Limit vs symmetric benchmark at round {T}: mean D {np.round(bench.mean(1), 3)} vs "
          f"{np.round(sur.mean(1), 3)} over {len(seeds)} runs, {res['fails']} of {res['k']} p_inter differ")

    rule = {"model": "M3",
            "params": {"b": 0.0, "kappa": 0.3, "gamma_K": 2.0, "a_K": 0.3, "beta": 3.0, "a": 0.3}}
    ds = dataset(rule, 5, Pool.exact(), 20, seed=1)
    out = analyse(ds, outcome_args={"pool": Pool.exact(), "rounds": 20, "n_rep": 3})
    assert out["stats"]["H1"].mean() > 0 and len(out["stats"]["H2"]) == 5
    print(f"ok   one M3 group of 5 runs: H1 d_r {np.round(out['stats']['H1'], 1)}, "
          f"p {({k: v for k, v in out['p'].items()})}, decisions {out['decisions']}")

    # 0.02 fails H1's level and passes H3's; the declarations do not gate the outcomes.
    assert decide({"H1": 0.02, "H2": 0.01, "H3": 0.02}) == {"H1": False, "H2": True, "H3": True}
    assert decide({"H1": 0.0, "H3": 0.5}) == {"H1": True, "H3": False}
    x = np.random.default_rng(0).uniform(size=(3, 60))
    assert check_limit(x, x[:, ::-1])["fails"] == 0 and check_limit(x, x + 1)["fails"] == 3
    assert normal_power([[1.0, 1.0, 1.0, 1.1]] * 2, 4) > 0.99 and normal_power([[1.0, -1.0, 0.5, -0.5]], 9) < 0.05
    assert binom_upper(1) == 0 and binom_upper(20) == 3
    assert wilson(0, 300)[0] == 0 and wilson(300, 300)[1] == 1
    # The latest entry sets the row; an empty window none.
    table = signatures([{"y": y, "window": w} for y, w in
                        [(1, [(-1, 1), (1, 1)]), (-1, [(1, 1)]), (-1, [(-1, 1)]), (1, [])]])
    table = table.set_index(["s", "o"])
    assert table.loc[(1, 1), ["n", "mean"]].tolist() == [2, 0] and table.loc[(-1, 1), "mean"] == -1
    assert len(table) == 2 and signatures([]).empty
    t, inv = confusion([("M1", "M1"), ("M1", "M2"), ("M2", "M2")])
    assert t["M1"]["M2"] == 0.5 and inv["M2"]["M1"] == 0.5
    print("ok   levels, Binomial bound, the signature table, confusion and inversion on hand cases")

    fixed = {"b": 0.1, "kappa": 0.3}
    kept = {**fixed, fit.NAME_TERMS[0]: 0.5}   # with a name intercept
    spec = {**kept, **dict.fromkeys(RATES, (RATE_LOW, 1.0)), **dict.fromkeys(WEIGHTS, (0.25, 64.0))}
    drawn = [draw_rule("M3", spec, np.random.default_rng(i)) for i in range(50)]
    assert all(d["params"][k] == v for d in drawn for k, v in kept.items())
    assert all(spec[k][0] <= v <= spec[k][1] for d in drawn for k, v in d["params"].items() if k not in kept)
    assert np.median([d["params"]["a"] for d in drawn]) < 0.5, "a rate is log-uniform"
    assert np.median([d["params"]["beta"] for d in drawn]) < 16, "a weight is log-uniform"
    assert set(draw_rule("M1", spec, np.random.default_rng(0))["params"]) == \
        {"b", "kappa", "gamma_K", "a_K", fit.NAME_TERMS[0]}
    assert set(draw_rule("M2", spec, np.random.default_rng(0))["params"]) == {"b", "kappa", "beta", "a", fit.NAME_TERMS[0]}
    assert math.isclose((1 - RATE_LOW) ** RECORD_ENTRIES, 0.5)
    tops = []
    for rounds in (5, 10):      # both searches: the onset at or above 1, then below it
        onset, top, rows = saturation({"model": "M1", "params": {
            **fixed, "gamma_K": 0.0, "a_K": 0.15}}, "gamma_K", Pool.exact(), 3, rounds)
        by = {r["value"]: r for r in rows}
        moved = lambda a, b: any(abs(by[a][k] - by[b][k]) > SATURATED * s for k, s in SUMMARIES.items())  # noqa: E731
        assert [r["value"] for r in rows] == sorted(by) and rows[0]["value"] == 0 and rows[0]["model"] == "M1"
        assert not moved(0.0, onset) and moved(0.0, 2 * onset), "the onset: the last value acting like 0"
        assert not moved(top, 2 * top) and all(moved(v, 2 * v) for v in by if 2 * onset <= v < top)
        tops.append((onset, top))
    assert tops[0][0] >= 1 > tops[1][0]
    print(f"ok   a draw keeps the baseline, stays in its range with rates and weights log-uniform; "
          f"gamma_K's onset and saturation {tops} on two toy groups")

    # Weights and rates on the log scale: γ_K recovered at twice its value, a_K at its inverse;
    # b, fixed, on its own.
    hand = [{"true": "M1", "cell": "c", "params": {"b": 0.1, "gamma_K": t, "a_K": t / 10},
             "R3": {"fitted": {"b": 0.3, "gamma_K": 2 * t, "a_K": 0.01 / t}}} for t in (1.0, 2.0, 4.0, 8.0)]
    table = {x["parameter"]: x for x in parameter_recovery(hand, 3)}
    assert math.isclose(table["gamma_K"]["r"], 1) and math.isclose(table["gamma_K"]["bias"], math.log(2))
    assert table["a_K"]["n"] == 4 and math.isclose(table["a_K"]["r"], -1)
    assert table["b"]["scale"] == "raw" and math.isnan(table["b"]["r"]) and math.isclose(table["b"]["bias"], 0.2)
    induced = induced_correlations(hand, 3)
    assert len(induced) == 1 and math.isclose(induced[0]["simulated"], 1) and math.isclose(induced[0]["recovered"], -1)
    assert math.isnan(_corr([1, 1], [0, 1]))
    # A second cell, recovered at a quarter and once at 0, is read apart from the first: the 0
    # counted, off the log scale.
    other = [{**h, "cell": "d", "R3": {"fitted": {"b": 0.1, "gamma_K": 0.0 if t == 1 else t / 4,
                                                   "a_K": 0.01 / t}}}
             for h, t in zip(hand, (1.0, 2.0, 4.0, 8.0))]
    table = {(x["cell"], x["parameter"]): x for x in parameter_recovery(hand + other, 3)}
    d = table[("d", "gamma_K")]
    assert math.isclose(d["bias"], -math.log(4)) and math.isclose(d["r"], 1)
    assert d["n"] == 4 and d["nonpositive"] == 1 and table[("c", "gamma_K")]["nonpositive"] == 0
    assert [(x["cell"], x["n"]) for x in induced_correlations(hand + other, 3)] == [("c", 4), ("d", 3)]
    print("ok   parameter recovery's r and bias on the log scale, induced correlations on hand "
          "cases, per cell")

    # One group; the tests' held-out scores are the choice's.
    ds2 = dataset(rule, 3, Pool.exact(), 30, seed=2)
    data = fit.arrays(ds2["decls"])
    choice = choose_model(data, 2)
    assert set(choice["scores"]) == set(choice["bic"]) == set(MODELS)
    assert choice["chosen"] in MODELS and choice["chosen_bic"] in MODELS
    assert all(np.isclose(choice["ll"][m].sum(), choice["scores"][m]) for m in MODELS)
    print(f"ok   model choice on one group: {choice['chosen']}")

    import tempfile
    settings = {"commit": "x", "rounds": 10, "R": 5, "n_rep": 16, "n": 1,
                "cells": {"c": {"ranges": dict.fromkeys(MODELS, spec)}}}
    row = task(2, settings, lambda cell: Pool.exact())
    assert row["true"] == "M2" and sizes(row) == [2, 3, 5] and len(row["divergence"]) == 5
    assert row["versions"] == {"numpy": np.__version__, "scipy": scipy.__version__}
    assert set(row["R5"]["decisions"]) == {16, 8, 4, 2} and len(row["R3"]["gains"]["H2"]) == 3
    assert set(row["R5"]["decisions"][2]) == {"H1", "H2", "H3"}
    long = rejection_rows([json.loads(json.dumps(row))])
    assert len(long) == 3 * 4 * 3 and {x["R"] for x in long} == {2, 3, 5}
    assert [truth(m, "H1") for m in MODELS] == [truth(m, "H3") for m in MODELS] == [False, False, True, True]
    assert set(row["R3"]["gains"]) == {"H1", "M1:M0", "M2:M0", "H2"}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        for i in range(2):
            (d / f"task_{i}.json").write_text(json.dumps({**row, "id": i}), encoding="utf-8")
        assert merge(d, d / "settings.json", d / "rows.jsonl") == [2, 3] and not (d / "rows.jsonl").exists()
        (d / "task_2.json").write_text(json.dumps(row), encoding="utf-8")
        (d / "task_3.json").write_text(json.dumps({**row, "id": 3}), encoding="utf-8")
        assert merge(d, d / "settings.json", d / "rows.jsonl") == []
        assert [r["id"] for r in read_rows(d / "rows.jsonl")] == list(range(4))
        (d / "task_4.json").write_text(json.dumps({**row, "commit": "y"}), encoding="utf-8")
        try:
            merge(d, d / "settings.json", d / "rows.jsonl")
        except ValueError:
            pass
        else:
            raise AssertionError("a duplicate id must abort")
    print("ok   one array task at three R and four n_rep; the merge lists missing ids and refuses duplicates")

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test_divergence.csv"
        mean, sd = np.array([[0.5, 0.2], [0.5, 0.1]]), np.array([[0.1, 0.2], [0.3, 0.4]])
        save_curve(p, {"mean": mean, "sd": sd, "p_inters": np.array([0.0, 0.5]), "n": 10}, {"check": "test"})
        p_inters, m, s, n = read_curve(p)
        assert p_inters == [0.0, 0.5] and (m == mean).all() and (s == sd).all() and n == 10
        assert json.loads(p.with_suffix(".csv.params.json").read_text())["check"] == "test"
    assert divergence_curve(LIMIT, Pool.exact(), range(1, 21), 20, p_inters=(0.0, 0.5))["mean"].shape == (20, 2)
    print("ok   a D curve written and read back in the benchmark's format")


def _cli(argv: list[str]) -> None:
    if argv[:1] == ["task"] and len(argv) == 5:
        i, settings_path, audit, out_dir = int(argv[1]), *argv[2:]
        out = Path(out_dir) / f"task_{i}.json"
        if out.is_file():
            print(f"{out} exists; skipped")
            return
        settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        row = task(i, settings, lambda cell: Pool.from_audit(Path(audit) / cell))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(row), encoding="utf-8")
        tmp.replace(out)   # a task cut off leaves no row, so a rerun redoes it
    elif argv[:1] == ["merge"] and len(argv) == 4:
        missing = merge(*argv[1:])
        if missing:
            raise SystemExit(f"{len(missing)} ids missing: {missing[:50]}{' …' if len(missing) > 50 else ''}")
    elif argv == ["selftest"]:
        selftest()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    _cli(sys.argv[1:])
