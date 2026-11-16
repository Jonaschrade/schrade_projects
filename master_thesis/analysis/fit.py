"""
M0–M3 fitted to declarations by maximum likelihood, the held-out scores and the
tests (README § The models, § Hypotheses and tests, § The fit). No LLM import: it
runs locally on ``metrics``' declaration dicts (y, f, window, run, agent).

Every term is a trace T <- T + a(v - T) from 0 over the entries shown, so each is
linear in its values given its rate: ``arrays`` stores the values once, aligned by
lag, and a likelihood evaluation is a few matrix-vector products. The selftest
checks every trace against ``network.opinion``'s scalar definitions, which the
runs' benchmark imports.

    python analysis/fit.py selftest
"""

from __future__ import annotations

import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import isotonic_regression, minimize
from scipy.special import expit, log_expit

_REPO = next(d for d in Path(__file__).resolve().parents if (d / "simulation" / "config.py").is_file())
sys.path.insert(0, str(_REPO / "simulation"))
sys.path.insert(0, str(_REPO / "analysis"))

from agents.names import NAMES  # noqa: E402
from network.opinion import own_pole_lags, perceived_reward  # noqa: E402

# Model -> (habit K, feedback ΔQ): bias, habit, feedback alone (the canon's rule), both.
MODELS = {"M0": (False, False), "M1": (True, False), "M2": (False, True), "M3": (True, True)}
# Name intercepts, reference-coded in alphabetical order: b is the first name's lean,
# b_<name> each other name's departure from it.
NAME_ORDER = tuple(sorted(NAMES))
NAME_TERMS = tuple(f"b_{n}" for n in NAME_ORDER[1:])

# Multi-start (Wilson & Collins 2019, rule 5): stop once STALL consecutive starts
# fail to raise the best log-likelihood by TOL. Heuristic: W&C give no count; TOL
# is far below the ½ log n per parameter any comparison here could turn on.
STALL, TOL = 5, 1e-4
# Sign-flip and permutation draws: the Monte Carlo SE of a p-value near the tests'
# lower level, 0.0125, stays at a tenth of it, √(0.0125 · 0.9875 / B) <= 0.00125.
PERM_DRAWS = 7900


# -- Data ---------------------------------------------------------------------

def arrays(decls: list[dict]) -> dict:
    """The declarations as lag-aligned arrays (column j = lag j, the newest entry at 0):
    ``K`` ŝ; ``G`` ŝ r̂ by own-pole lag; ``X`` the name dummies, one column per
    ``NAME_TERMS``; plus y, f, run, agent, round and whether the latest entry is explored.
    A declaration's name is its ``name``, else its ``agent``."""
    y = np.array([d["y"] for d in decls])
    if not np.isin(y, (1, -1)).all():
        raise ValueError("declarations must be poles, ±1")
    n, L = len(decls), max([len(d["window"]) for d in decls] + [1])
    K, G = np.zeros((n, L)), np.zeros((n, L))
    for i, d in enumerate(decls):
        w = d["window"][::-1]   # newest first: index = lag
        lags = own_pole_lags([s for s, _ in w[::-1]])[::-1]
        for j, ((s, o), m) in enumerate(zip(w, lags)):
            K[i, j] = s
            G[i, m] += s * perceived_reward(s, o)
    names = np.array([d.get("name", d.get("agent")) for d in decls], dtype=object)
    if not np.isin(names, NAMES).all():
        raise ValueError(f"names outside agents.names: {sorted(set(names) - set(NAMES), key=str)[:5]}")
    return {"y": y, "f": np.array([d["f"] for d in decls], float),
            "run": np.array([d.get("run", "") for d in decls]), "K": K, "G": G,
            "X": (names[:, None] == np.array(NAME_ORDER[1:])).astype(float),
            "agent": np.array([d.get("agent", "") for d in decls]),
            "round": np.array([d.get("round", 0) for d in decls]),
            "explored_latest": np.array([bool((d.get("explored") or [False])[-1]) for d in decls])}


def subset(data: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in data.items()}


def without_names(data: dict) -> dict:
    """``data`` fitted with b alone, no name intercepts."""
    return {k: v for k, v in data.items() if k != "X"}


def name_term(p: dict, name: str) -> float:
    """``name``'s intercept beside b: 0 for the reference and for a fit without names."""
    if name not in NAMES:
        raise ValueError(f"{name!r} is not in agents.names")
    return p.get(f"b_{name}", 0.0)


def _decay(a: float, L: int) -> tuple[np.ndarray, np.ndarray]:
    """a(1 - a)^lag for lags 0 … L-1, and its derivative in a."""
    j = np.arange(L)
    q = (1 - a) ** j
    dq = np.where(j > 0, j * (1 - a) ** np.maximum(j - 1, 0), 0.0)
    return a * q, q - a * dq


# -- Models -------------------------------------------------------------------

def param_names(model: str) -> list[str]:
    habit, feedback = MODELS[model]
    names = ["b", "kappa"]
    if habit:
        names += ["gamma_K", "a_K"]
    if feedback:
        names += ["beta", "a"]
    return names


def _bounds(name: str, free_signs: bool = False) -> tuple:
    if name == "beta":
        return (None, None) if free_signs else (0.0, None)
    if name.startswith("a"):
        return (0.0, 1.0)
    return (None, None)


def _eta_jac(p: dict, data: dict, model: str) -> tuple[np.ndarray, dict]:
    """η and its derivative in each parameter; the derivative in a weight is its trace."""
    habit, feedback = MODELS[model]
    L = data["K"].shape[1]
    e = p["b"] + p["kappa"] * data["f"]
    J = {"b": np.ones(len(e)), "kappa": data["f"]}
    if "X" in data:
        for t, x in zip(NAME_TERMS, data["X"].T):
            if t in p:   # a name the fit never saw adds nothing
                J[t] = x
                e = e + p[t] * x
    if habit:
        w, dw = _decay(p["a_K"], L)
        J["gamma_K"], J["a_K"] = data["K"] @ w, p["gamma_K"] * (data["K"] @ dw)
        e = e + p["gamma_K"] * J["gamma_K"]
    if feedback:
        w, dw = _decay(p["a"], L)
        J["beta"], J["a"] = data["G"] @ w, p["beta"] * (data["G"] @ dw)
        e = e + p["beta"] * J["beta"]
    return e, J


def terms(p: dict, data: dict, model: str) -> dict[str, np.ndarray]:
    """The traces K and ΔQ at the rates in ``p``."""
    _, J = _eta_jac({"b": 0.0, "kappa": 0.0, "gamma_K": 1.0, "beta": 1.0, **p},
                    without_names(data), model)
    return {t: J[w] for t, w in (("K", "gamma_K"), ("dQ", "beta")) if w in J}


def eta(p: dict, data: dict, model: str) -> np.ndarray:
    return _eta_jac(p, data, model)[0]


def loglik(p: dict, data: dict, model: str) -> np.ndarray:
    """Per-declaration log P(y), the binary logit P(y = +1) = σ(η)."""
    return log_expit(data["y"] * eta(p, data, model))


def predict(p: dict, data: dict, model: str) -> np.ndarray:
    """P(y = +1) per declaration."""
    return expit(eta(p, data, model))


def sample(p: dict, data: dict, model: str, rng: np.random.Generator) -> np.ndarray:
    """Declarations drawn from the model, as poles."""
    return np.where(rng.random(len(data["y"])) < predict(p, data, model), 1, -1)


# -- Fitting ------------------------------------------------------------------

def _objective(v: np.ndarray, names: list[str], data: dict, model: str) -> tuple[float, np.ndarray]:
    """Negative log-likelihood and its gradient."""
    e, J = _eta_jac(dict(zip(names, v)), data, model)
    ye = data["y"] * e
    d_eta = data["y"] * expit(-ye)   # d log σ(yη) / dη
    return -log_expit(ye).sum(), -np.array([d_eta @ J[n] for n in names])


def _start(names: list[str], rng: np.random.Generator) -> np.ndarray:
    """A random start in each parameter's natural units."""
    lo = {n: _bounds(n)[0] for n in names}
    return np.array([rng.random() if _bounds(n)[1] == 1.0 else
                     abs(rng.normal()) if lo[n] == 0.0 else rng.normal() for n in names])


def fit(data: dict, model: str, seed: int = 0, free_signs: bool = False) -> dict:
    """Maximum likelihood from random starts until the best stops improving;
    ``free_signs`` drops β ≥ 0. A name intercept is fitted for every name in ``data``
    but the reference, which must be there too."""
    seen = data["X"].any(0) if "X" in data else np.zeros(len(NAME_TERMS), bool)
    if seen.any() and data["X"].sum(1).min() > 0:
        raise ValueError(f"no declaration by the reference name {NAME_ORDER[0]}")
    names = param_names(model) + [t for t, s in zip(NAME_TERMS, seen) if s]
    rng = np.random.default_rng(seed)
    best, stall = None, 0
    while stall < STALL:
        res = minimize(_objective, _start(names, rng), (names, data, model), method="L-BFGS-B",
                       jac=True, bounds=[_bounds(n, free_signs) for n in names])
        if best is None or res.fun < best.fun - TOL:
            best, stall = res, 0
        else:
            stall += 1
    p = {k: float(v) for k, v in zip(names, best.x)}
    n, k = len(data["y"]), len(names)
    return {"model": model, "params": p, "ll": -best.fun, "n": n, "k": k,
            "bic": k * math.log(n) - 2 * -best.fun,
            "at_bound": ["beta"] if p.get("beta") == 0.0 else []}


def heldout(data: dict, model: str, seed: int = 0, **kw) -> tuple[np.ndarray, dict]:
    """Leave one run out: per-declaration log-likelihood of each run under ``model``
    fitted on the others, and each fold's fit by its run; a fit is a surrogate rule."""
    ll, fits = np.empty(len(data["y"])), {}
    for r in np.unique(data["run"]):
        out = data["run"] == r
        f = fit(subset(data, ~out), model, seed, **kw)
        ll[out] = loglik(f["params"], subset(data, out), f["model"])
        fits[r] = f
    return ll, fits


def run_gains(ll_a: np.ndarray, ll_b: np.ndarray, data: dict,
              mask: np.ndarray | None = None) -> np.ndarray:
    """d_r: the held-out log-likelihood gain of a over b, summed per run (over ``mask``)."""
    d = ll_a - ll_b if mask is None else np.where(mask, ll_a - ll_b, 0.0)
    return np.array([d[data["run"] == r].sum() for r in np.unique(data["run"])])


# -- Tests (README § Hypotheses and tests) ------------------------------------

def sign_flip(d: np.ndarray, seed: int = 0) -> float:
    """One-sided p of a mean above 0: exact over all sign patterns when they number at
    most PERM_DRAWS, else PERM_DRAWS random ones."""
    d = np.asarray(d, float)
    obs = d.mean()
    if 2 ** len(d) <= PERM_DRAWS:
        signs = np.array(list(product((1, -1), repeat=len(d))))
        return float((signs @ d / len(d) >= obs - 1e-12).mean())
    signs = np.random.default_rng(seed).choice((1, -1), (PERM_DRAWS, len(d)))
    return float((1 + (signs @ d / len(d) >= obs - 1e-12).sum()) / (1 + PERM_DRAWS))


def validity_test(d_own: np.ndarray, d_full: np.ndarray, share: float = 0.5, reps: int = PERM_DRAWS,
                  seed: int = 0) -> float:
    """One-sided bootstrap p that the mean gain in `own` is not below ``share`` of `full`'s,
    resampling runs in each condition."""
    rng = np.random.default_rng(seed)
    d_own, d_full = np.asarray(d_own), np.asarray(d_full)
    diff = [d_own[rng.integers(0, len(d_own), len(d_own))].mean()
            - share * d_full[rng.integers(0, len(d_full), len(d_full))].mean() for _ in range(reps)]
    return float((1 + np.sum(np.array(diff) >= 0)) / (1 + reps))


def crps_score(reps, y: float) -> float:
    """-CRPS of the replicates' outcomes at the observed ``y``, Ferro's (2014) fair estimator:
    unbiased for the score of the replicates' distribution at any n_rep >= 2."""
    x = np.sort(np.asarray(reps, float))
    n = len(x)
    pairs = 2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x)   # sum over i != j of |x_i - x_j|
    return float(-(np.abs(x - y).mean() - pairs / (2 * n * (n - 1))))


def crps_variance(reps, y: float) -> float:
    """n_rep times ``crps_score``'s Monte Carlo variance, to first order: the variance over
    replicates of |x_i - y| - mean_j |x_i - x_j|."""
    x = np.asarray(reps, float)
    return float(np.var(np.abs(x - y) - np.abs(x[:, None] - x).mean(1), ddof=1))


def squared_score(reps, y: float) -> float:
    """``crps_score``'s alternative: -(mean - y)², less the mean's sampling variance."""
    x = np.asarray(reps, float)
    return float(-((x.mean() - y) ** 2 - x.var(ddof=1) / len(x)))


def coupling_test(p_inter: np.ndarray, y: np.ndarray, strata: np.ndarray | None = None,
                  seed: int = 0) -> tuple[float, float]:
    """(least-squares slope of the outcome y on p_inter, p_inter centred within ``strata``
    (topics) when pooled; one-sided p that y falls with p_inter), the null from permuting
    y within strata."""
    x, y = np.asarray(p_inter, float), np.asarray(y, float)
    strata = np.zeros(len(x)) if strata is None else np.asarray(strata)
    groups = [strata == s for s in np.unique(strata)]
    xc = x.copy()
    for m in groups:
        xc[m] -= x[m].mean()
    sxx = xc @ xc
    if sxx == 0:
        return math.nan, 1.0
    obs = xc @ y / sxx
    rng = np.random.default_rng(seed)
    perm = y.copy()
    hits = 0
    for _ in range(PERM_DRAWS):
        for m in groups:
            perm[m] = rng.permutation(y[m])
        hits += xc @ perm / sxx <= obs + 1e-12
    return float(obs), float((1 + hits) / (1 + PERM_DRAWS))


def transition(p_inter: np.ndarray, y: np.ndarray) -> float:
    """The p_inter at which the non-increasing isotonic fit of y on p_inter crosses the
    midpoint of its values at the grid's ends, interpolated linearly; NaN if it does not
    fall."""
    grid, at = np.unique(np.asarray(p_inter, float), return_inverse=True)
    w = np.bincount(at).astype(float)
    f = isotonic_regression(np.bincount(at, np.asarray(y, float)) / w, weights=w,
                            increasing=False).x
    if f[0] <= f[-1]:
        return math.nan
    mid = (f[0] + f[-1]) / 2
    i = int(np.argmax(f <= mid))
    return float(grid[i - 1] + (f[i - 1] - mid) / (f[i - 1] - f[i]) * (grid[i] - grid[i - 1]))


# -- Selftest -----------------------------------------------------------------

def _random_decls(rng: np.random.Generator, n: int, runs: int) -> list[dict]:
    return [{"y": 1, "f": int(rng.choice((1, -1))), "run": f"r{i % runs}", "agent": NAME_ORDER[i % 3],
             "window": [(int(rng.choice((1, -1))), int(rng.choice((1, -1))))
                        for _ in range(rng.integers(0, 12))]} for i in range(n)]


def selftest() -> None:
    from network.opinion import delta_q, trace

    rng = np.random.default_rng(1)
    decls = _random_decls(rng, 600, 6)
    data = arrays(decls)
    t = terms({"a_K": 0.3, "a": 0.2}, data, "M3")
    for i, d in enumerate(decls):
        w = d["window"]
        assert math.isclose(t["K"][i], trace([s for s, _ in w], 0.3), abs_tol=1e-12)
        assert math.isclose(t["dQ"][i], delta_q([s for s, _ in w],
                                                [perceived_reward(s, o) for s, o in w], 0.2),
                            abs_tol=1e-12)
    print("ok   traces equal network.opinion's trace and delta_q (K, dQ)")

    # The analytic gradient against finite differences, every model.
    from scipy.optimize import approx_fprime
    data["y"] = rng.choice((1, -1), len(decls))
    for model in MODELS:
        names = param_names(model) + list(NAME_TERMS)
        v = _start(names, rng).clip(0.05, 0.95) * np.where(
            [_bounds(n)[1] == 1.0 for n in names], 1, 2)
        num = approx_fprime(v, lambda u: _objective(u, names, data, model)[0], 1e-6)
        assert np.allclose(_objective(v, names, data, model)[1], num, rtol=1e-4, atol=1e-3), model
    print("ok   analytic gradient equals finite differences, every model")

    # The fit reaches at least the true likelihood; M1 and M2, each nested in M3, stay below it
    # and above M0.
    true = {"b": 0.2, "kappa": 0.3, "gamma_K": 2.0, "a_K": 0.3, "beta": 3.0, "a": 0.3,
            f"b_{NAME_ORDER[1]}": 0.5, f"b_{NAME_ORDER[2]}": -0.5}
    data["y"] = sample(true, data, "M3", rng)
    f = {m: fit(data, m) for m in MODELS}
    assert f["M3"]["ll"] >= loglik(true, data, "M3").sum() - TOL
    assert all(f["M3"]["ll"] >= f[m]["ll"] - TOL >= f["M0"]["ll"] - 2 * TOL for m in ("M1", "M2"))
    print(f"ok   fit: M3 {f['M3']['ll']:.1f} >= truth {loglik(true, data, 'M3').sum():.1f}, "
          f"M1 {f['M1']['ll']:.1f} and M2 {f['M2']['ll']:.1f} >= M0 {f['M0']['ll']:.1f}")

    ll3, _ = heldout(data, "M3")
    ll1, _ = heldout(data, "M1")
    d = run_gains(ll3, ll1, data)
    assert len(d) == 6 and d.mean() > 0, d
    print(f"ok   held-out d_r(M3:M1) {np.round(d, 1)}, sign-flip p {sign_flip(d):.3f}")

    assert sign_flip(np.ones(20)) < 0.001 and sign_flip(-np.ones(6)) == 1.0
    assert math.isclose(sign_flip(np.array([1.0, -1.0])), 0.75)
    p_inter = np.repeat(np.linspace(0, 0.5, 11), 4)
    y = np.where(p_inter < 0.2, 0.5, 0.05) + np.tile([-0.02, 0.0, 0.01, 0.01], 11)
    stat, pval = coupling_test(p_inter, y)
    assert pval < 0.01 and stat < 0 and 0.15 < transition(p_inter, y) < 0.2
    pooled = coupling_test(np.r_[p_inter, p_inter], np.r_[y, y + 0.3], np.repeat([0, 1], len(y)))
    assert math.isclose(pooled[0], stat), "topic shifts leave the within-topic slope"
    print(f"ok   tests: sign-flip exact and drawn, H3 slope {stat:.2f} p {pval:.4f}, "
          f"isotonic midpoint {transition(p_inter, y):.3f}")

    assert coupling_test(-p_inter, y)[1] > 0.99 and math.isnan(transition(p_inter, -y))
    reps = np.random.default_rng(1).uniform(0, 1, 7)
    brute = np.abs(reps - 0.3).mean() - np.abs(reps[:, None] - reps).sum() / (2 * 7 * 6)
    assert math.isclose(crps_score(reps, 0.3), -brute) and abs(crps_score([0.3] * 5, 0.3)) < 1e-12
    assert crps_score(reps, reps.mean()) > crps_score(reps, 2.0)
    assert squared_score(reps, reps.mean()) > squared_score(reps, 2.0)
    draws = [crps_score(np.random.default_rng(i).uniform(0, 1, 40), 0.3) for i in range(400)]
    big = np.random.default_rng(0).uniform(0, 1, 4000)
    assert 0.7 < np.var(draws) / (crps_variance(big, 0.3) / 40) < 1.3, "first-order variance"
    one = arrays([{"y": 1, "f": 1, "agent": NAME_ORDER[0], "window": [(1, 1), (1, -1)]}])
    assert np.allclose(terms({"a": 1.0, "a_K": 1.0}, one, "M3")["dQ"], -1), "the latest entry's r̂ at rate 1"
    # Feedback's weight reversed: the bound holds at 0, free signs find it.
    data["y"] = sample({**true, "beta": -3.0}, data, "M3", rng)
    bound, free = fit(data, "M3"), fit(data, "M3", free_signs=True)
    assert bound["params"]["beta"] >= 0 and free["params"]["beta"] < -1 and free["ll"] > bound["ll"]
    print("ok   specifications: opposite tail, CRPS, squared error, free signs")

    # Names: reference-coded alphabetically, a term per name seen, none for the reference.
    names = fit(data, "M1")["params"]
    assert {t for t in names if t.startswith("b_")} == {f"b_{n}" for n in NAME_ORDER[1:3]}
    assert fit(data, "M1")["ll"] >= fit(without_names(data), "M1")["ll"] - TOL
    assert name_term(names, NAME_ORDER[0]) == 0 and name_term(names, NAME_ORDER[1]) == names[f"b_{NAME_ORDER[1]}"]
    assert np.isfinite(heldout(data, "M1")[0]).all()
    try:
        fit(subset(data, data["X"].any(1)), "M0")
        raise AssertionError("fitted without the reference name")
    except ValueError:
        pass
    print(f"ok   names: b is {NAME_ORDER[0]}'s lean, a term per other name seen, "
          "a fit without names nested, the reference required")


if __name__ == "__main__":
    if sys.argv[1:] != ["selftest"]:
        raise SystemExit("usage: python analysis/fit.py selftest")
    selftest()
