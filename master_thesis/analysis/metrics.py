"""
Every outcome, label, table and interval the notebooks, fit.py and the surrogate share
(README § Signatures, § Hypotheses and tests; schedule.md § Exchange audit). No LLM
import: it reads logs and runs locally.

A *declaration* is one dict: ``y`` the declared pole (±1), ``f`` +1 when the
list started from TOPIC_TEXT's pole (``topic_first``), else -1, ``window`` the
entries shown, oldest first, each ``(s, o)`` (the turn's self-read ŝ, the probe ô of
the partner's: logged, never shown),
``explored`` one flag per entry, whether it
is an explored entry (``explored_entry``; None where unknown), plus ``run`` and
``agent``. ``run_declarations`` and ``audit_declarations`` build them; the signature
table and fit.py read nothing else.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

POSITIONS = (-1, 1)
Z95 = 1.959963984540054
THIN_N = 30  # below this a table row's normal interval is not read


# -- Outcomes -----------------------------------------------------------------

def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def divergence(blocks: Iterable[Iterable[int]]) -> float:
    """The runs' outcome D = |m_0 - m_1| / 2 (README § Hypotheses and tests), m_c community c's mean
    pole (0 for an empty block): 1 when the two communities hold opposite poles to an
    agent, 0 when they lean alike."""
    m = [float(np.mean([_sign(x) for x in b])) if b else 0.0 for b in map(list, blocks)]
    return abs(m[0] - m[1]) / 2


def _blocks(positions: dict, communities: dict) -> list[list[int]]:
    return [[positions[a] for a, c in communities.items() if c == k] for k in (0, 1)]


def outcome_values(positions: dict, communities: dict) -> dict[str, float]:
    """D and its alternative on one state (schedule.md § Sensitivity): ``canon`` 1
    without consensus, the canon's count, else 0."""
    poles = {_sign(x) for x in positions.values()}
    return {"divergence": divergence(_blocks(positions, communities)),
            "canon": float(poles not in ({1}, {-1}))}


def benchmark_values(rows: pd.DataFrame) -> pd.DataFrame:
    """The values a benchmark's rows give, from the communities' mean opinions ``mag_c``:
    |mag| = 1 iff every agent of the community holds that sign."""
    m0, m1 = rows["mag_c0"], rows["mag_c1"]
    return pd.DataFrame({"divergence": (m0 - m1).abs() / 2,
                         "canon": (~(m0.abs().eq(1) & m0.eq(m1))).astype(float)})


# -- Intervals ----------------------------------------------------------------

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval of a proportion k/n, exactly 0 or 1 at k = 0 or n; (nan,
    nan) at n = 0."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (0.0 if k == 0 else centre - half), (1.0 if k == n else centre + half)


def summarize(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    """n, mean and 95% interval of ``col`` per group: Wilson for a 0/1 column, else
    the normal interval of the mean."""
    rows = []
    for key, g in df.groupby(by, dropna=False):
        v = g[col].astype(float).to_numpy()
        n, m = len(v), v.mean()
        if set(np.unique(v)) <= {0.0, 1.0}:
            lo, hi = wilson(int(v.sum()), n)
        else:
            half = Z95 * v.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
            lo, hi = m - half, m + half
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                     "n": n, "mean": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# -- Reading logs -------------------------------------------------------------

def read_events(directory: Path | str) -> list[dict]:
    with open(Path(directory) / "events.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def read_params(directory: Path | str) -> dict:
    return json.loads((Path(directory) / "params.json").read_text(encoding="utf-8"))


def is_complete(run_dir: Path | str) -> bool:
    """Whether a run wrote its final declarations: the round H + 1 snapshot, written last,
    exists. False for a run still going or aborted (and for one mid-copy)."""
    run_dir = Path(run_dir)
    h = read_params(run_dir)["NETWORK_MAX_ROUNDS"]
    return (run_dir / "network_rounds" / f"round_{h + 1:04d}.json").is_file()


def run_outcome(run_dir: Path | str, at: int | None = None) -> dict:
    """A run's p_inter, its state by community at the final declarations (round H + 1) or
    round ``at``, D and its alternatives (``values``: ``persistent`` the least D over every
    state up to it, the start included) and the realised graph's cross-edge share."""
    run_dir = Path(run_dir)
    params = read_params(run_dir)
    rounds = sorted((run_dir / "network_rounds").glob("round_*.json"))
    read = lambda f: json.loads(f.read_text(encoding="utf-8"))  # noqa: E731
    graph = read(rounds[0])
    if read(rounds[-1])["round"] != params["NETWORK_MAX_ROUNDS"] + 1:
        raise ValueError(f"{run_dir.name} ended at round {read(rounds[-1])['round']}: "
                         "aborted, to be replaced")
    end = len(rounds) - 1 if at is None else at
    communities, edges = graph["communities"], graph["edges"]
    state = read(rounds[end])["positions"]
    blocks = _blocks(state, communities)
    values = outcome_values(state, communities) | {
        "persistent": min(divergence(_blocks(read(f)["positions"], communities))
                          for f in rounds[:end + 1])}
    return {"run": run_dir.name, "p_inter": params["SBM_P_INTER"], "blocks": blocks,
            "divergence": values["divergence"], "values": values,
            "cross_share": sum(communities[a] != communities[b] for a, b in edges) / len(edges)}


def _snapshots(run_dir: Path | str) -> tuple[dict, list[dict]]:
    """A run's communities and its round snapshots in order (rounds 0 … H + 1; an unfinished
    run gives the rounds so far)."""
    files = sorted((Path(run_dir) / "network_rounds").glob("round_*.json"))
    snaps = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return snaps[0]["communities"], snaps


def run_divergence(run_dir: Path | str) -> pd.Series:
    """D at every snapshot a run has written, indexed by round."""
    communities, snaps = _snapshots(run_dir)
    return pd.Series([divergence(_blocks(s["positions"], communities)) for s in snaps],
                     index=[s["round"] for s in snaps], name="divergence")


def run_community_means(run_dir: Path | str) -> pd.DataFrame:
    """Each community's mean declared pole at every snapshot, indexed by round, one column
    per community (0 and 1)."""
    communities, snaps = _snapshots(run_dir)
    return pd.DataFrame([[np.mean(b) for b in _blocks(s["positions"], communities)] for s in snaps],
                        index=[s["round"] for s in snaps], columns=[0, 1])


def dispersion(positions: Iterable[int]) -> float:
    """The declared poles' variance over the whole network, the benchmark's measure: 0 at
    consensus, 1 at an even division, blind to where the communities stand."""
    return float(np.var(list(positions)))


def run_dispersion(run_dir: Path | str) -> pd.Series:
    """``dispersion`` at every snapshot a run has written, indexed by round."""
    _, snaps = _snapshots(run_dir)
    return pd.Series([dispersion(s["positions"].values()) for s in snaps],
                     index=[s["round"] for s in snaps], name="dispersion")


def audit_units(cell_dir: Path | str, sub: str, phases=("audit",)) -> list[dict]:
    """A cell's units of one sub-phase, pooled over ``phases`` (calibration is not pooled)."""
    units = []
    for phase in phases:
        d = Path(cell_dir) / f"{phase}_{sub}"
        if (d / "events.jsonl").is_file():
            units += [{**u, "phase": phase} for u in read_events(d) if u["type"] == "unit"]
    return units


def position_calls(record, path: str = "") -> Iterable[tuple[str, dict]]:
    """Every position call (a dict with x, logprobs, topic_first) nested in ``record``,
    with its path; a start has no logprobs and is skipped."""
    if isinstance(record, dict):
        if {"x", "logprobs", "topic_first"} <= record.keys():
            yield path, record
            return
        for k, v in record.items():
            yield from position_calls(v, f"{path}/{k}")
    elif isinstance(record, list):
        for i, v in enumerate(record):
            yield from position_calls(v, f"{path}/{i}")


# -- Declarations -------------------------------------------------------------

def _self_read(side: dict) -> int | None:
    return ((side.get("turn") or {}).get("self_read") or {}).get("x")


def explored_entry(explored: bool | None, s: int, s_hat: int | None) -> bool | None:
    """Whether a turn stores an explored entry: explored, and its self-read ŝ on the pole
    it argued, so the entry's own pole is drawn. None where unknown."""
    if explored is None or s_hat is None:
        return None
    return bool(explored) and _sign(s_hat) == _sign(s)


def _decl(y: int, topic_first: bool, window, explored, **extra) -> dict:
    return {"y": y, "f": 1 if topic_first else -1, "window": [tuple(e) for e in window],
            "explored": list(explored), **extra}


def run_declarations(run_dir: Path | str) -> list[dict]:
    """Every modelled declaration of one run: in its exchanges and at round H + 1,
    each with the window its shown ids name. Drawn starts are not modelled."""
    events = read_events(run_dir)
    run = Path(run_dir).name
    entries: dict[tuple[str, int], tuple] = {}
    out = []

    def add(agent: str, decl: dict, round_: int) -> None:
        shown = [entries[(agent, i)] for i in decl["shown"]]
        out.append(_decl(decl["x"], decl["topic_first"], [e[:2] for e in shown],
                         [e[2] for e in shown], run=run, agent=agent, round=round_))

    for ev in events:
        if ev["type"] == "exchange":
            for side in ev["sides"].values():
                d = side.get("declaration", {})
                if not d.get("drawn") and "shown" in d:
                    add(side["agent"], d, ev["round"])
            for side in ev["sides"].values():
                if side.get("stored") is not None:
                    s_hat = _self_read(side)
                    entries[(side["agent"], side["stored"])] = (
                        s_hat, side["probe"]["x"],
                        explored_entry(side["explored"], side["s"], s_hat))
        elif ev["type"] == "declaration" and not ev.get("drawn"):
            add(ev["agent"], ev, ev["round"])
    return out


def audit_declarations(units: list[dict], condition: str) -> list[dict]:
    """The meetings' declarations under ``condition``: ``empty`` with no window, ``full``
    or ``own`` with one entry, the side's self-read ŝ and probe ô."""
    out = []
    for u in units:
        for role, decls in u["declarations"].items():
            side = u["exchange"]["sides"][role]
            d = decls[condition]
            window = [] if condition == "empty" else [(_self_read(side), side["probe"]["x"])]
            out.append(_decl(d["x"], d["topic_first"], window, [False] * len(window),
                             agent=side["agent"], kind="meeting", key=u["key"]))
    return out


# -- Signatures (README § Signatures) -----------------------------------------

def signatures(decls: list[dict]) -> pd.DataFrame:
    """The signature table: per latest entry (ŝ, ô), ``summarize``'s n, mean y and
    interval; the share of y on ŝ's pole is (1 + ŝ · mean y) / 2."""
    df = pd.DataFrame([(d["y"], *d["window"][-1]) for d in decls if d["window"]],
                      columns=["y", "s", "o"], dtype=int)
    return summarize(df, ["s", "o"], "y")


# -- Audit readers (schedule.md § Exchange audit) -----------------------------

def perception(units: list[dict]) -> pd.DataFrame:
    """One row per probe: the perceiver's and the partner's argued positions, the
    probe ô, the perceiver's and the partner's self-reads of their turns."""
    rows = []
    for u in units:
        x = u["exchange"]
        for role, other in (("i", "j"), ("j", "i")):
            me, them = x["sides"][role], x["sides"][other]
            rows.append({"key": u["key"], "role": role, "s": me["s"], "partner_s": them["s"],
                         "o_hat": me["probe"]["x"], "self_read": _self_read(me),
                         "partner_self_read": _self_read(them)})
    return pd.DataFrame(rows)


def _turn_row(side: dict, role: str, **extra) -> dict:
    read = _self_read(side)
    return {**extra, "role": role, "s": side["s"], "self_read": read,
            "inconsistent": None if read is None else read != side["s"]}


def turns(units: list[dict]) -> pd.DataFrame:
    """One row per audit turn: its meeting (``unit``, the index in ``units``), role,
    argued position, blind self-read, ``inconsistent`` (ŝ ≠ s), and the side's own probe ``o_hat``."""
    return pd.DataFrame([_turn_row(side, role, unit=i, key=u["key"], o_hat=side["probe"]["x"])
                         for i, u in enumerate(units)
                         for role, side in u["exchange"]["sides"].items()])


def _off_pole(side: dict) -> bool | None:
    read = _self_read(side)
    return None if read is None else _sign(read) != _sign(side["declaration"]["x"])


def run_turns(run_dir: Path | str) -> pd.DataFrame:
    """One row per turn of one run, as ``turns`` (its ``o_hat`` included), with round, agent, the
    logged reward ``r_hat``, ``explored``,
    ``explored_entry`` (its mean the effective exploration rate) and ``off_pole``: ŝ's
    pole is not the declaration's (its mean the realised rate, vs ε)."""
    return pd.DataFrame([_turn_row(side, role, round=ev["round"], o_hat=side["probe"]["x"],
                                   agent=side["agent"], r_hat=side["r_hat"], explored=side["explored"],
                                   explored_entry=explored_entry(side["explored"], side["s"],
                                                                 _self_read(side)),
                                   off_pole=_off_pole(side))
                         for ev in read_events(run_dir) if ev["type"] == "exchange"
                         for role, side in ev["sides"].items()])


def expression_bias(turns: pd.DataFrame) -> dict:
    """The expression channel's lean toward a pole: the mean of ŝ − s over the two argued
    poles, weighted equally, so the mix argued does not move it; 0 without a lean,
    positive toward +1. ``turns``: a ``turns`` or ``run_turns`` frame."""
    t = turns.dropna(subset=["self_read"])
    g = (t.self_read.astype(float) - t.s).groupby(t.s)
    m, v, n = g.mean(), g.var(ddof=1), g.size()
    if not {1, -1} <= set(m.index):
        return {"bias": math.nan, "lo": math.nan, "hi": math.nan, "n": len(t)}
    bias = float((m[1] + m[-1]) / 2)
    half = Z95 * math.sqrt(v[1] / n[1] + v[-1] / n[-1]) / 2
    return {"bias": bias, "lo": bias - half, "hi": bias + half, "n": len(t)}


# Label mass below which a call's logprobs give no answer distribution: Ollama returns
# only the top 20 tokens and the drawn answer, so such a call lists little but the draw.
# Label masses split at a gap around it (audit_analysis § 5 prints both sides).
LABEL_MASS_MIN = 1e-3


def label_probs(logprobs: dict[str, float]) -> dict[int, float] | None:
    """A call's answer distribution: its logprobs renormalised over the labels listed,
    an unlisted one counting as 0; None under LABEL_MASS_MIN."""
    p = {int(k): math.exp(v) for k, v in logprobs.items()}
    mass = sum(p.values())
    return None if mass < LABEL_MASS_MIN else {k: v / mass for k, v in p.items()}


def _read_moments(kind: str, call: dict | None) -> dict:
    """One read's draw and label mass, and over its ``label_probs`` the expected position,
    a re-ask's variance of it, and repeat agreement Σp²; NaN if none."""
    logprobs = call["logprobs"] if call else {}
    p = label_probs(logprobs)
    out = dict.fromkeys(("E", "V", "agree"), math.nan)
    if p:
        x, w = np.array(list(p), dtype=float), np.array(list(p.values()))
        e = w @ x
        out = {"E": e, "V": w @ x**2 - e**2, "agree": w @ w}
    out |= {"x": call["x"] if call else math.nan,
            "mass": sum(math.exp(v) for v in logprobs.values())}
    return {f"{kind}_{k}": v for k, v in out.items()}


def read_noise(units: list[dict]) -> pd.DataFrame:
    """One row per meeting turn: its argued position s, whether its self-read and its
    partner's probe drew one answer order (``same_order``), and ``_read_moments`` of each."""
    rows = []
    for u in units:
        sides = u["exchange"]["sides"]
        for speaker, reader in (("i", "j"), ("j", "i")):
            own = (sides[speaker].get("turn") or {}).get("self_read")
            probe = sides[reader]["probe"]
            rows.append({"s": sides[speaker]["s"],
                         "same_order": own["topic_first"] == probe["topic_first"] if own else None,
                         **_read_moments("self", own), **_read_moments("probe", probe)})
    return pd.DataFrame(rows)


def reliability(noise: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Test-retest per read, from ``read_noise``: ``between``, the variance of
    the expected read over texts; ``sampling``, a re-ask's mean variance (a lower bound,
    unlisted labels counting as 0); ``reliability``, between / (between + sampling);
    ``agree``, the chance a re-ask repeats the answer; ``unread``, the share under
    LABEL_MASS_MIN, left out."""
    rows = []
    for key, g in ([(None, noise)] if by is None else noise.groupby(by)):
        for kind in ("self", "probe"):
            read = g.dropna(subset=[f"{kind}_E"])
            between, sampling = read[f"{kind}_E"].var(ddof=1), read[f"{kind}_V"].mean()
            rows.append({**({by: key} if by else {}), "read": kind, "n": len(read),
                         "between": between, "sampling": sampling,
                         "reliability": between / (between + sampling),
                         "agree": read[f"{kind}_agree"].mean(),
                         "unread": 1 - len(read) / len(g)})
    return pd.DataFrame(rows)


def read_gap(noise: pd.DataFrame) -> pd.DataFrame:
    """The self-read against the probe on one text, by ``same_order``: the mean of (ŝ − ô)²
    (``mismatch``), its ``sampling`` part, both reads' variances, and its ``gap``, the mean
    of (E[ŝ] − E[ô])². The parts should sum to ``mismatch`` if the logprobs describe the draws."""
    d = noise.dropna(subset=["self_E", "probe_E"])
    return (d.assign(mismatch=(d.self_x - d.probe_x) ** 2, sampling=d.self_V + d.probe_V,
                     gap=(d.self_E - d.probe_E) ** 2)
            .groupby("same_order")[["mismatch", "sampling", "gap"]].mean()
            .assign(n=d.groupby("same_order").size()))


def order_bias(calls: Iterable[dict]) -> dict:
    """The lean toward the first-listed answer: half the mean x when the list
    starts from TOPIC_TEXT's pole minus when it starts from the other; 0 without
    a lean, positive toward the first answer. ``calls``: position calls (x, topic_first)."""
    df = pd.DataFrame([{"x": c["x"], "tf": c["topic_first"]} for c in calls])
    if df.empty or df.tf.nunique() < 2:
        return {"bias": math.nan, "lo": math.nan, "hi": math.nan, "n": len(df)}
    a, b = df[df.tf].x.to_numpy(float), df[~df.tf].x.to_numpy(float)
    bias = (a.mean() - b.mean()) / 2
    half = Z95 * math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) / 2
    return {"bias": bias, "lo": bias - half, "hi": bias + half, "n": len(df)}

