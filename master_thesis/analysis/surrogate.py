"""
The surrogate (README § Recovery): runs as main_network.py runs them,
with each declaration drawn from a fitted model instead of the LLM and each
exchange replayed from the exchange audit. It serves recovery, the outcome tests and the
sensitivity analysis. No LLM import; runs locally.

A replayed exchange is drawn uniformly from the audit's meetings at the pair's two
argued positions, each meeting in both orientations; it fixes each side's self-read ŝ
and the ô it perceives. Windows follow exactly from the stored history; a synthetic
agent responds to ŝ and ô themselves, a run's to the turns (README § The models, Inputs).

``LIMIT`` in place of a fitted rule, with ``Pool.exact()``, is the symmetric
benchmark: argmax ΔQ at the canon's α, both sides learning and exploring, exact
expression and perception (``recovery.py selftest``).

    python analysis/surrogate.py selftest
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.special import expit

_REPO = next(d for d in Path(__file__).resolve().parents if (d / "simulation" / "config.py").is_file())
for _p in ("simulation", "analysis", "analysis/benchmark"):
    sys.path.insert(0, str(_REPO / _p))

import fit  # noqa: E402
from agents.names import NAMES  # noqa: E402
from config import (  # noqa: E402
    EXPLORATION_RATE,
    LEARNING_RATE,
    NUM_AGENTS,
    P_START_PLUS,
    SBM_NUM_COMMUNITIES,
)
from metrics import (  # noqa: E402
    audit_units,
    divergence,
    explored_entry,
    outcome_values,
    read_events,
)
from network.matching import draw_pair  # noqa: E402
from network.opinion import (  # noqa: E402
    AgentOpinionState,
    argmax_pole,
    explore,
    perceived_reward,
    update_q_value,
)
from network.topology import build_graph  # noqa: E402

LIMIT = "limit"
LONG_RUN_ROUNDS = 20_000  # T, the canon's run length: mean D at T stands for the long-run mean


# -- Replayed exchanges -------------------------------------------------------

class Pool:
    """Replayed (ŝ_i, ô_i, ŝ_j, ô_j) by (s_i, s_j), each meeting in both orientations
    since the roles differ only in call order; ``table`` None is exact expression and
    perception."""

    def __init__(self, table: dict | None) -> None:
        self.table = table

    @staticmethod
    def _table(exchanges) -> dict:
        table: dict = {}
        for sides in exchanges:
            i, j = sides["i"], sides["j"]
            read_i = (i["turn"]["self_read"]["x"], i["probe"]["x"])
            read_j = (j["turn"]["self_read"]["x"], j["probe"]["x"])
            table.setdefault((i["s"], j["s"]), []).append(read_i + read_j)
            table.setdefault((j["s"], i["s"]), []).append(read_j + read_i)
        return table

    @classmethod
    def from_audit(cls, cell_dir: Path | str) -> "Pool":
        return cls(cls._table(u["exchange"]["sides"] for u in audit_units(cell_dir, "meetings")))

    @classmethod
    def from_other_runs(cls, run_dirs, fallback: "Pool") -> dict[str, "Pool"]:
        """Per run, by directory name: the other runs' exchanges, ``fallback``'s at pairs
        they never held."""
        tables = {Path(d).name: cls._table(e["sides"] for e in read_events(d) if e["type"] == "exchange")
                  for d in run_dirs}
        pools = {}
        for name in tables:
            others: dict = {}
            for other, table in tables.items():
                if other != name:
                    for pair, reads in table.items():
                        others.setdefault(pair, []).extend(reads)
            pools[name] = cls({**fallback.table, **others})
        return pools

    @classmethod
    def exact(cls) -> "Pool":
        return cls(None)

    def draw(self, s_i: int, s_j: int, rng: random.Random) -> tuple:
        if self.table is None:
            return s_i, s_j, s_j, s_i
        options = self.table.get((s_i, s_j))
        if not options:
            raise KeyError(f"no audit exchange at s_i={s_i}, s_j={s_j}")
        return rng.choice(options)


# -- Agents -------------------------------------------------------------------

class _Agent:
    """An agent's stored entries and its traces over all of them, kept incrementally:
    K, the trace of ŝ; Q, the canon's state under update_q_value. ``name`` sets a fitted
    rule's intercept, b plus the name's term."""

    def __init__(self, rule, name: str | None = None) -> None:
        self.rule, self.name = rule, name
        self.entries: list = []
        if rule == LIMIT:
            self.a_K, self.a = None, LEARNING_RATE
        else:
            habit, feedback = fit.MODELS[rule["model"]]
            p = rule["params"]
            self.a_K, self.a = (p["a_K"] if habit else None), (p["a"] if feedback else None)
            self.b = p["b"] + fit.name_term(p, name)
        self.k, self.q = 0.0, AgentOpinionState()

    def add(self, s: int, o: int, explored: bool) -> None:
        self.entries.append((s, o, explored))
        if self.a_K is not None:
            self.k += self.a_K * (s - self.k)
        if self.a is not None:
            update_q_value(self.q, s, perceived_reward(s, o), self.a)

    def declare(self, held: int, rng: random.Random) -> tuple[int, int]:
        """(pole, f): the rule's draw at a random answer order."""
        f = 1 if rng.random() < 0.5 else -1
        if self.rule == LIMIT:
            return argmax_pole(self.q.q_gap, held), f
        p = self.rule["params"]
        eta = self.b + p["kappa"] * f
        if self.a_K is not None:
            eta += p["gamma_K"] * self.k
        if self.a is not None:
            eta += p["beta"] * self.q.q_gap
        return (1 if rng.random() < expit(eta) else -1), f

    def shown(self) -> tuple[list, list]:
        return [(s, o) for s, o, _ in self.entries], [e for *_, e in self.entries]


# -- One run ------------------------------------------------------------------

def draw_setup(p_inter: float, seed: int, n: int = NUM_AGENTS, start: str = "polarized") -> tuple[nx.Graph, dict]:
    """A run's graph (the benchmarks' per seed), its names and starts: ``polarized``, each
    community on its own pole, as the runs; ``coin``, each pole drawn (schedule.md
    § Sensitivity). The names, node attribute ``name``, are placed from their own stream,
    so graph and starts stay the benchmarks'; past 20 agents each name recurs."""
    graph = build_graph(range(n), SBM_NUM_COMMUNITIES, p_inter, random.Random(seed))
    names = list(NAMES) * -(-n // len(NAMES))
    random.Random(f"names:{seed}").shuffle(names)
    nx.set_node_attributes(graph, dict(zip(graph, names)), "name")
    return graph, draw_starts(graph, seed, start)


def draw_starts(graph: nx.Graph, seed: int, start: str = "polarized") -> dict:
    g = random.Random(f"start:{seed}")
    if start == "polarized":
        return {i: 1 if graph.nodes[i]["community"] == 0 else -1 for i in graph}
    if start == "coin":
        return {i: 1 if g.random() < P_START_PLUS else -1 for i in graph}
    raise ValueError(f"unknown start {start!r}")


def run_setup(run_dir: Path | str) -> tuple[nx.Graph, dict]:
    """A logged run's graph and starts (round 0), for the outcome tests' replicates."""
    snap = json.loads((Path(run_dir) / "network_rounds" / "round_0000.json").read_text(encoding="utf-8"))
    graph = nx.Graph()
    for node, c in snap["communities"].items():
        graph.add_node(node, community=c, name=node)
    graph.add_edges_from(snap["edges"])
    starts = {e["agent"]: e["x"] for e in read_events(run_dir)
              if e["type"] == "declaration" and e.get("drawn")}
    return graph, starts


def run(rule, graph: nx.Graph, starts: dict, pool: Pool, rounds: int, seed: int, *,
        epsilon: float = EXPLORATION_RATE, record: bool = False, trace: bool = False) -> dict:
    """One run in main_network.py's order: per exchange the pair, both declarations
    (the start for an agent's first exchange), exploration i then j, the
    replayed exchange, the entries. The state at round t is every agent's
    declaration in round t + 1; the outcome is round H's, the final declarations.
    ``record`` keeps every modelled declaration as a ``metrics`` declaration dict,
    ``trace`` D per round. ``held`` is every agent's latest declaration, what a run's
    round snapshot holds; ``persistent`` the least D over the final declarations and
    every snapshot before them, the start included, as ``metrics.run_outcome`` reads it.
    A round is N exchanges (config.EXCHANGES_PER_ROUND = NUM_AGENTS), so N follows the
    graph."""
    nodes = list(graph)
    members = [[i for i in nodes if graph.nodes[i]["community"] == c]
               for c in range(SBM_NUM_COMMUNITIES)]
    eligible = [i for i in nodes if graph.degree(i) > 0]
    neighbours = {i: list(graph.neighbors(i)) for i in eligible}
    rng, draws = random.Random(seed), random.Random(f"draw:{seed}")
    # The states draw apart from the run, so tracing them changes neither it nor the outcome.
    state_rng, final_rng = random.Random(f"state:{seed}"), random.Random(f"final:{seed}")
    random.seed(seed)   # agent j and exploration draw from the global stream, as in a run
    agents = {i: _Agent(rule, graph.nodes[i].get("name")) for i in nodes}
    held, pending = dict(starts), set(nodes)
    d_of = lambda ys: divergence([[ys[i] for i in m] for m in members])  # noqa: E731
    decls, d_trace, held_d = [], [], [d_of(held)]

    def declare(i, round_, g, log=True):
        y, f = agents[i].declare(held[i], g)
        if log and record:
            w, ex = agents[i].shown()
            decls.append({"y": y, "f": f, "window": w, "explored": ex, "run": seed,
                          "agent": i, "name": agents[i].name, "round": round_})
        return y

    def state(round_, g, log):
        ys = {i: declare(i, round_ + 1, g, log) for i in nodes}
        return ys, d_of(ys)

    for round_ in range(1, rounds + 1):
        for _ in range(len(nodes) if eligible else 0):
            a, b = draw_pair(eligible, neighbours, rng)
            ys = [held[i] if i in pending else declare(i, round_, draws) for i in (a, b)]
            pending -= {a, b}
            held[a], held[b] = ys
            (s_a, ex_a), (s_b, ex_b) = explore(ys[0], epsilon), explore(ys[1], epsilon)
            sh_a, o_a, sh_b, o_b = pool.draw(s_a, s_b, draws)
            agents[a].add(sh_a, o_a, explored_entry(ex_a, s_a, sh_a))
            agents[b].add(sh_b, o_b, explored_entry(ex_b, s_b, sh_b))
        held_d.append(d_of(held))
        if trace and round_ < rounds:
            d_trace.append(state(round_, state_rng, False)[1])
    final, d = state(rounds, final_rng, True)
    d_trace.append(d)
    return {"divergence": d, "final": final, "held": held, "persistent": min([d, *held_d]),
            "decls": decls, "divergence_trace": d_trace if trace else None}


def run_values(out: dict, graph: nx.Graph, state: str = "final") -> dict[str, float]:
    """``metrics.outcome_values`` on one of ``run``'s states, and ``persistent``."""
    return outcome_values(out[state], dict(graph.nodes(data="community"))) | {
        "persistent": out["persistent"]}


# -- Over seeds ---------------------------------------------------------------

def _task_run(task, rule, pool, rounds, kw, **run_kw) -> tuple[nx.Graph, dict]:
    p_inter, seed = task
    graph, starts = draw_setup(p_inter, seed, kw.get("n", NUM_AGENTS), kw.get("start", "polarized"))
    return graph, run(rule, graph, starts, pool, rounds, seed, **run_kw,
                      **{k: v for k, v in kw.items() if k not in ("n", "start")})


def _trace_task(task, rule, pool, rounds, kw):
    return _task_run(task, rule, pool, rounds, kw, trace=True)[1]["divergence_trace"]


def divergence_moments(rule, pool: Pool, p_inters, seeds, rounds: int, jobs: int = 1,
                       **kw) -> tuple[np.ndarray, np.ndarray]:
    """D's mean and sd over ``seeds`` per round (rows) and p_inter (columns)."""
    tasks = [(p_inter, s) for p_inter in p_inters for s in seeds]
    one = partial(_trace_task, rule=rule, pool=pool, rounds=rounds, kw=kw)

    def summed(traces) -> tuple[np.ndarray, np.ndarray]:
        # Trace by trace: all of them at once would not fit in memory at T rounds.
        total, square = np.zeros((2, rounds, len(p_inters)))
        for i, trace in zip((i for i in range(len(p_inters)) for _ in seeds), traces):
            total[:, i] += trace
            square[:, i] += np.square(trace)
        n = len(seeds)
        mean = total / n
        return mean, np.sqrt(np.maximum(square - n * mean ** 2, 0) / max(n - 1, 1))

    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            return summed(ex.map(one, tasks, chunksize=max(1, len(tasks) // (4 * jobs))))
    return summed(map(one, tasks))


# -- Selftest -----------------------------------------------------------------

def _hand_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from((i, {"community": int(i > 1)}) for i in range(4))
    graph.add_edges_from(((0, 1), (2, 3), (1, 2)))
    return graph


def selftest() -> None:
    rule = {"model": "M3",
            "params": {"b": 0.1, "kappa": 0.2, "gamma_K": 2.0, "a_K": 0.2, "beta": 3.0, "a": 0.3,
                       f"b_{fit.NAME_ORDER[1]}": 1.0, f"b_{fit.NAME_ORDER[2]}": -1.0}}
    graph, starts = draw_setup(0.2, 3)
    assert sorted(dict(graph.nodes(data="name")).values()) == sorted(NAMES)
    assert graph.edges == build_graph(range(NUM_AGENTS), SBM_NUM_COMMUNITIES, 0.2, random.Random(3)).edges
    out = run(rule, graph, starts, Pool.exact(), 30, 3, record=True, trace=True)
    again = run(rule, graph, starts, Pool.exact(), 30, 3, record=True)
    assert out["final"] == again["final"] and len(out["divergence_trace"]) == 30
    assert out["divergence_trace"][-1] == out["divergence"] >= out["persistent"]
    data = fit.arrays(out["decls"])
    # Each logged window, recomputed by fit, gives the probability the surrogate drew from.
    eta = fit.eta(rule["params"], data, rule["model"])
    for d, e in zip(out["decls"][:200], eta[:200]):
        a = _Agent(rule, d["name"])
        for s, o in d["window"]:
            a.add(s, o, False)
        p = rule["params"]
        mine = p["b"] + p.get(f"b_{d['name']}", 0.0) + p["kappa"] * d["f"] + p["gamma_K"] * a.k + p["beta"] * a.q.q_gap
        assert math.isclose(mine, e, abs_tol=1e-9), (mine, e)
    print(f"ok   surrogate: {len(out['decls'])} declarations reproduce per seed; "
          "its traces and name intercepts equal fit.eta on every logged window")

    mean, sd = divergence_moments(LIMIT, Pool.exact(), (0.0, 0.5), range(1, 5), 3)
    by_hand = np.array([[_trace_task((p_inter, s), LIMIT, Pool.exact(), 3, {}) for s in range(1, 5)]
                        for p_inter in (0.0, 0.5)])   # p_inter x seed x round
    assert np.allclose(mean, by_hand.mean(1).T) and np.allclose(sd, by_hand.std(1, ddof=1).T)
    print("ok   D's mean and sd over the runs' traces")

    # An agent that never flips: a polarized start held by the limit's argmax at ε = 0.
    graph, starts = draw_setup(0.0, 1, start="polarized")
    out = run(LIMIT, graph, starts, Pool.exact(), 20, 1, epsilon=0.0)
    values = run_values(out, graph)
    assert all(v == 1 for v in values.values()) and out["held"] == starts, values
    # One community united, the other split: D 1/2, no consensus.
    values = run_values({"final": {0: 1, 1: 1, 2: -1, 3: 1}, "persistent": 0.0}, _hand_graph())
    assert values == {"divergence": 0.5, "canon": 1.0, "persistent": 0.0}, values
    assert run_values({"final": dict.fromkeys(range(4), 1), "persistent": 0.0}, _hand_graph())["canon"] == 0
    print("ok   a held polarized start gives D = 1 on every value; hand cases of D and canon")

    def meeting(s_i: int, read_i: tuple, s_j: int, read_j: tuple) -> dict:
        side = lambda s, r: {"s": s, "turn": {"self_read": {"x": r[0]}}, "probe": {"x": r[1]}}
        return {"type": "unit", "exchange": {"sides": {"i": side(s_i, read_i), "j": side(s_j, read_j)}}}

    with tempfile.TemporaryDirectory() as cell:
        (Path(cell) / "audit_meetings").mkdir()
        (Path(cell) / "audit_meetings" / "events.jsonl").write_text("".join(
            json.dumps(m) + "\n" for m in (meeting(1, (1, -1), -1, (-1, 1)),
                                           meeting(-1, (1, 1), 1, (1, 1)),
                                           meeting(1, (1, 1), 1, (-1, 1)))), encoding="utf-8")
        table = Pool.from_audit(cell).table
    assert table == {(1, -1): [(1, -1, -1, 1), (1, 1, 1, 1)],
                     (-1, 1): [(-1, 1, 1, -1), (1, 1, 1, 1)],
                     (1, 1): [(1, 1, -1, 1), (-1, 1, 1, 1)]}, table
    print("ok   each meeting is filed in both orientations, its sides together")

    with tempfile.TemporaryDirectory() as root:
        dirs = [Path(root) / name for name in ("a", "b")]
        for d, m in zip(dirs, (meeting(1, (1, -1), -1, (-1, 1)), meeting(1, (-1, -1), -1, (1, 1)))):
            d.mkdir()
            (d / "events.jsonl").write_text(json.dumps({"type": "exchange", **m["exchange"]}) + "\n",
                                            encoding="utf-8")
        pools = Pool.from_other_runs(dirs, Pool({(1, 1): [(1, 1, 1, 1)], (1, -1): [(-1, 1, -1, 1)]}))
    assert pools["a"].table[(1, -1)] == [(-1, -1, 1, 1)] and pools["b"].table[(1, -1)] == [(1, -1, -1, 1)]
    assert pools["a"].table[(1, 1)] == [(1, 1, 1, 1)]
    print("ok   a run's pool holds the other runs' exchanges, the audit's at pairs they never held")


if __name__ == "__main__":
    if sys.argv[1:] != ["selftest"]:
        raise SystemExit("usage: python analysis/surrogate.py selftest")
    selftest()
