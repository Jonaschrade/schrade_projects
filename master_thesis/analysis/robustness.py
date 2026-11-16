"""
The specification tables (schedule.md § Sensitivity): each test rerun under what its
primary specification fixes. Descriptive, outside the error control, gating nothing. One
row per variant and test: its statistic, p and whether its reading at its level
``agrees`` with the primary's. Outcomes, tests and rules stay in metrics, fit, recovery and
surrogate; this module only varies what they are given. No LLM import.

    python analysis/robustness.py selftest
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import fit
import metrics
import recovery
import surrogate
from recovery import LEVEL_COUPLING, declaration_p, declaration_stats, outcome_p, outcome_stats
from surrogate import Pool


def _rows(variant: str, stats: dict, p: dict) -> list[dict]:
    """One row per test of ``p``."""
    return [{"variant": variant, "test": t, "mean": float(np.mean(stats[t])), "p": q,
             "rejects": recovery.rejects(q)} for t, q in p.items()]


def _agrees(rows: list[dict]) -> pd.DataFrame:
    """The rows, each test's ``agrees`` against its first row, the primary."""
    df = pd.DataFrame(rows)
    return df.assign(agrees=df.rejects == df.groupby("test").rejects.transform("first"))


def declarations_table(decls: list[dict], seed: int = 0) -> pd.DataFrame:
    """H1 under each specification, on one condition's group."""
    data = fit.arrays(decls)
    primary = recovery.declarations(data, seed)
    ll = primary["ll"]
    rows = _rows("primary", primary, declaration_p(primary, seed))

    explored = declaration_stats(ll, data, mask=data["explored_latest"])
    rows += _rows("explored entries", explored, declaration_p(explored, seed))
    for variant, dt, kw in (("free signs", data, {"free_signs": True}),
                            ("no name intercepts", fit.without_names(data), {})):
        stats = recovery.declarations(dt, seed, **kw)
        rows += _rows(variant, stats, declaration_p(stats, seed))
    return _agrees(rows)


def outcomes_table(ds: dict, fits: dict, pool: Pool, rounds: int, n_rep: int,
                   other_runs: dict[str, Pool], seed: int = 0) -> pd.DataFrame:
    """H2 under each outcome, score and surrogate assumption, from the declaration tests'
    fold ``fits`` (``recovery.declarations``). Each of ``ds``' runs names its ``values`` and
    ``divergence_half``, D at round H/2; ``other_runs`` its pool from the other runs
    (``Pool.from_other_runs``)."""
    k = recovery.replicate_values(ds, fits, pool, rounds, n_rep, seed)
    primary = outcome_stats(k["divergence"])
    rows = _rows("primary", primary, outcome_p(primary, seed))
    for name, by_model in k.items():
        if name != "divergence":
            stats = outcome_stats(by_model)
            rows += _rows(f"outcome {name}", stats, outcome_p(stats, seed))
    squared = outcome_stats(k["divergence"], fit.squared_score)
    rows += _rows("squared error", squared, outcome_p(squared, seed))
    half = {"runs": [{**r, "values": {"divergence": r["divergence_half"]}} for r in ds["runs"]]}
    for variant, call in (
            ("the runs' exchanges",
             lambda: recovery.replicate_values(ds, fits, other_runs, rounds, n_rep, seed)),
            ("at H/2",
             lambda: recovery.replicate_values(half, fits, pool, rounds // 2, n_rep, seed,
                                               state="held"))):
        stats = outcome_stats(call()["divergence"])
        rows += _rows(variant, stats, outcome_p(stats, seed))
    return _agrees(rows)


def coupling_table(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """H3's test under each outcome and against the realised graph, on ``metrics.run_outcome``'s
    frame: ``mean`` the outcome's, ``slope`` its on p_inter, ``transition`` its isotonic
    midpoint."""
    p_inter = df.p_inter.to_numpy(float)
    d = df.divergence.to_numpy(float)
    values = pd.DataFrame(list(df["values"]))

    def row(variant: str, x, y, transition: bool = True) -> dict:
        y = np.asarray(y, float)
        slope, p = fit.coupling_test(x, y, seed=seed)
        return {"variant": variant, "test": "H3", "mean": y.mean(), "slope": slope, "p": p,
                "rejects": p <= LEVEL_COUPLING,
                "transition": fit.transition(p_inter, y) if transition else np.nan}

    rows = [row("primary", p_inter, d)]
    rows += [row(f"outcome {k}", p_inter, values[k]) for k in values if k != "divergence"]
    rows.append(row("cross-edge share", df.cross_share.to_numpy(float), d, False))
    return _agrees(rows)


def selftest() -> None:
    rng = np.random.default_rng(1)
    pos = (1, -1)
    decls = [{"y": int(rng.choice(pos)), "f": int(rng.choice(pos)), "run": f"r{i % 4}", "agent": fit.NAME_ORDER[i % 3],
              "round": int(i % 40), "explored": [bool(i % 2)] * 6,
              "window": [(int(rng.choice(pos)), int(rng.choice(pos))) for _ in range(6)]} for i in range(160)]
    table = declarations_table(decls)
    assert {"primary", "explored entries", "free signs", "no name intercepts"} == set(table.variant)
    assert len(table) == table.variant.nunique() and table[table.variant == "primary"].agrees.all()
    print(f"ok   declarations: {table.variant.nunique()} variants of H1 on noise")

    rule = {"model": "M3",
            "params": {"b": 0.0, "kappa": 0.3, "gamma_K": 2.0, "a_K": 0.3, "beta": 3.0, "a": 0.3}}
    ds = recovery.dataset(rule, 4, Pool.exact(), 16, seed=1)
    for r in ds["runs"]:
        r["values"] = surrogate.run_values(r, r["graph"])
        r["divergence_half"] = 0.0
    fits = recovery.declarations(fit.arrays(ds["decls"]))["fits"]
    table = outcomes_table(ds, fits, Pool.exact(), 16, 2, {r["seed"]: Pool.exact() for r in ds["runs"]})
    assert {"outcome canon", "outcome persistent", "squared error", "the runs' exchanges", "at H/2"} <= set(table.variant)
    assert len(table) == table.variant.nunique()
    print(f"ok   outcomes: {table.variant.nunique()} variants of H2, primary as recovery.outcomes reads it")
    direct = recovery.outcomes(ds, fits, Pool.exact(), 16, 2)
    assert np.isclose(table[(table.variant == "primary") & (table.test == "H2")]["mean"].item(), direct["H2"].mean())

    p_inter = np.linspace(0, 0.5, 21)
    d = np.where(p_inter < 0.2, 0.6, 0.1)
    df = pd.DataFrame({"p_inter": p_inter, "divergence": d,
                       "values": [{"divergence": v, "canon": 1.0, "persistent": v} for v in d],
                       "cross_share": p_inter})
    table = coupling_table(df).set_index("variant")
    assert table.loc["primary", "rejects"] and table.loc["outcome persistent", "agrees"]
    assert 0.15 < table.loc["primary", "transition"] < 0.2
    assert not table.loc["outcome canon", "agrees"], "an outcome constant over runs cannot fall"
    assert table.loc["cross-edge share", "rejects"]
    print("ok   H3: outcomes, the realised graph")

    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run_x"
        (run / "network_rounds").mkdir(parents=True)
        (run / "params.json").write_text(json.dumps(
            {"SBM_P_INTER": 0.1, "NETWORK_MAX_ROUNDS": 2}), encoding="utf-8")
        graph = {"communities": dict(A=0, B=0, C=1, D=1), "edges": [["A", "B"], ["C", "D"], ["B", "C"]]}
        for t, ys in enumerate(((1, 1, -1, -1), (1, 1, 1, -1), (1, 1, -1, -1), (1, 1, -1, -1))):
            (run / "network_rounds" / f"round_{t:04d}.json").write_text(json.dumps(
                {"round": t, "positions": dict(zip("ABCD", ys)), **(graph if t == 0 else {})}), encoding="utf-8")
        end, start = metrics.run_outcome(run), metrics.run_outcome(run, at=0)
    assert end["values"] == {"divergence": 1.0, "canon": 1.0, "persistent": 0.5}, end
    assert end["cross_share"] == 1 / 3, end
    assert start["values"]["persistent"] == start["divergence"] == 1, start
    assert start["blocks"] == [[1, 1], [-1, -1]], start
    print("ok   a logged run's outcome: D, its alternatives, persistence over the run, an earlier round, the realised graph")


if __name__ == "__main__":
    if sys.argv[1:] != ["selftest"]:
        raise SystemExit("usage: python analysis/robustness.py selftest")
    selftest()
