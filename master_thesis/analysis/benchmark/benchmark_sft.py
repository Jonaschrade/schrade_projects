"""Scalar-agent SFT benchmark: the canon at this project's graph, pairing and horizon.

An agent *is* its declared pole in {-1, +1} and the reward is the canon's binary
r = s_a * s_b. The graph, the pairing, the update, ``explore()`` and config are
imported, never reimplemented, so the benchmark cannot drift from the runs or from
the ΔQ that analysis/fit.py fits.

``--rule assimilation`` swaps the update for binary assimilation (``assimilate()``,
analysis/README.md § The references), everything else held.

Switches (defaults: the canon scan's settings)
----------------------------------------------
--rule           sft (the canon's update) or assimilation
--epsilon        exploration rate, by default config's; ``explore()`` draws nothing at 0
--both-learn     agent j explores and learns too (Meylahn & Meylahn 2024)
--start          polarized (each community on its own pole, the canon's § 5.4 and the
                 runs') or coin (Q = (0, 0), each pole drawn)
--agents         N, with N exchanges per round

Ties: a tie keeps the agent's current pole, the start's until its first entry;
only from a coin start is it a coin (analysis/README.md § The references).

This module runs and writes rows; ``benchmark_analysis.ipynb`` reads them. The
rule kept here is ``classify()``, because ``settle_round`` needs the label every
round; the outcome D is ``analysis/metrics.divergence``.

Usage, from the repo root
-------------------------
    python analysis/benchmark/benchmark_sft.py selftest
    python analysis/benchmark/benchmark_sft.py sweep --epsilon 0.1 --reps 300 --max-rounds 20000 \\
        --jobs 6 --out analysis/logs/benchmark/canon_binary_20k.csv
    python analysis/benchmark/benchmark_sft.py sweep --both-learn --epsilon 0.1 \\
        --reps 300 --max-rounds 20000 --jobs 6 --out analysis/logs/benchmark/symmetric_binary_20k.csv
    python analysis/benchmark/benchmark_sft.py sweep --rule assimilation --epsilon 0.1 \\
        --reps 300 --max-rounds 20000 --jobs 6 --out analysis/logs/benchmark/assimilation_binary_20k.csv

Every run seeds itself, so ``--jobs`` changes nothing but the wall time. Every
sweep writes its rows, ``<stem>_divergence.csv`` (D's mean and sd per p_inter and round)
and a ``.params.json`` sidecar beside each.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np

# The repo root is the ancestor holding simulation/config.py.
_REPO = next(
    (d for d in Path(__file__).resolve().parents if (d / "simulation" / "config.py").is_file()),
    None,
)
if _REPO is None:
    raise SystemExit("cannot locate the repo root (no ancestor contains simulation/config.py)")
sys.path.insert(0, str(_REPO / "simulation"))
sys.path.insert(0, str(_REPO / "analysis"))

from config import (  # noqa: E402
    EXCHANGES_PER_ROUND,
    EXPLORATION_RATE,
    LEARNING_RATE,
    NETWORK_MAX_ROUNDS,
    NUM_AGENTS,
    SBM_NUM_COMMUNITIES,
    SBM_PAIRS_INTER,
    SBM_PAIRS_INTRA,
    SBM_P_INTER,
    SBM_P_INTRA,
)
from metrics import divergence  # noqa: E402
from network.matching import draw_pair  # noqa: E402
from network.opinion import (  # noqa: E402
    argmax_pole,
    compute_polarization_metrics,
    delta_q,
    explore,
    init_opinion_states,
    trace,
    update_q_value,
)
from network.topology import build_graph, count_edge_types  # noqa: E402

# The default grid: p_inter = 0-0.50 in steps of 0.01, each step one expected cross edge
# (SBM_PAIRS_INTER = 100), up to ER at p_inter = 0.5.
DEFAULT_LADDER = tuple(i / 100 for i in range(51))

# |mean community Q-gap| below this = no settled community stance: one full-reward TD
# step from Q = 0, and one assimilation step from A = 0. On a mean it weighs side by conviction, so a 5:5 division
# never clears it, and it holds the label off where a mean's sign flips on noise.
CONVICTION = LEARNING_RATE


# -- Graph ---------------------------------------------------------------------

def structure_summary(G: nx.Graph, opinion: dict) -> dict:
    """The graph's structure and, from ``opinion`` (node -> pole, 0 undecided), the
    opinion side: ``assort_stance`` and ``best_response_share``'s ``eta``."""
    degrees = [d for _, d in G.degree()]
    blocks: dict[int, set] = {}
    for node, data in G.nodes(data=True):
        blocks.setdefault(data["community"], set()).add(node)
    n_intra, n_inter = count_edge_types(G)

    def _assort(attr: str) -> float | None:
        try:
            v = nx.attribute_assortativity_coefficient(G, attr)
        except (ValueError, ZeroDivisionError, nx.NetworkXError):
            return None
        return None if v != v else round(float(v), 4)

    try:
        deg_assort = nx.degree_assortativity_coefficient(G)
        deg_assort = None if deg_assort != deg_assort else round(float(deg_assort), 4)
    except (ValueError, ZeroDivisionError, nx.NetworkXError):
        deg_assort = None

    nx.set_node_attributes(G, opinion, "stance")
    eta, n_isolated, _ = best_response_share(G, opinion)
    return {
        "clustering": round(nx.average_clustering(G), 4) if G.number_of_nodes() else None,
        "modularity": (round(nx.community.modularity(G, list(blocks.values())), 4)
                       if G.number_of_edges() else None),
        "assort_community": _assort("community"),
        "assort_degree": deg_assort,
        "std_degree": round(float(np.std(degrees)), 4) if degrees else None,
        "tie_ratio": round(n_intra / n_inter, 4) if n_inter else None,
        "assort_stance": _assort("stance") if len(set(opinion.values())) > 1 else None,
        "eta": round(eta, 4) if eta == eta else None,
        "n_isolated": n_isolated,
    }


# -- One run -------------------------------------------------------------------

def classify(mus: list[float], conviction: float = CONVICTION) -> str:
    """Regime label from the community mean signed Q-gaps (analysis/README.md § Regime definitions):
    opposite signs = bi-polarization, one sign = consensus (direction recorded), any
    community without a settled mean = unsettled."""
    if min(abs(m) for m in mus) < conviction:
        return "unsettled"
    if len({m > 0 for m in mus}) > 1:
        return "bi-polarization"
    return "consensus (+)" if mus[0] > 0 else "consensus (-)"


def block_order(signs: Sequence[int], blocks: Sequence[int]) -> tuple[list[float], int]:
    """Each block's mean opinion (pole, 0 undecided), in block-id order, and the
    holdouts: agents opposing their block's strict majority."""
    mags, holdouts = [], 0
    for b in sorted(set(blocks)):
        s = [x for x, c in zip(signs, blocks) if c == b]
        m = sum(s) / len(s)
        mags.append(m)
        if m:
            holdouts += sum(1 for x in s if x == (-1 if m > 0 else 1))
    return mags, holdouts


def best_response_share(G: nx.Graph, opinion: dict) -> tuple[float, int, int]:
    """(eta, n_isolated, n_undecided): the share of decided, non-isolated agents
    playing a best response in Banisch & Olbrich's coordination game (§ 5.3): at
    least as many decided neighbours with them as against."""
    satisfied = decided = isolated = undecided = 0
    for v in G:
        if G.degree(v) == 0:
            isolated += 1
            continue
        o = opinion.get(v, 0)
        if o == 0:
            undecided += 1
            continue
        nb = [opinion.get(u, 0) for u in G.neighbors(v)]
        decided += 1
        satisfied += nb.count(o) >= nb.count(-o)
    return (satisfied / decided if decided else float("nan")), isolated, undecided


def assimilate(state, partner_pole: int, alpha: float) -> None:
    """A <- A + α(s_partner − A): the trace of the partner's argued poles, declared as
    sign(A) (Martins' 2008 CODA, DeGroot through a binary channel). A is held as
    Q(+1) with Q(−1) at 0, so ΔQ = A and every readout reads it unchanged."""
    state.q_pos += alpha * (partner_pole - state.q_pos)


def run_one(
    p_inter: float,
    seed: int,
    *,
    n: int = NUM_AGENTS,
    alpha: float = LEARNING_RATE,
    epsilon: float = 0.0,
    both_learn: bool = False,
    start: str = "polarized",
    max_rounds: int = NETWORK_MAX_ROUNDS,
    rule: str = "sft",
) -> dict:
    """One run, in main_network's order: per exchange the pair, both declarations
    (the canon's argmax), agent i's exploration, then agent j's under
    ``both_learn``, the reward on the argued poles, the learners' updates.
    Under ``rule="assimilation"`` each learner instead assimilates the other's
    argued pole; the draws are the same, so one seed's runs pair across rules.

    No stopping rule: the run ends at ``max_rounds``; the label is recorded every
    round, which gives ``settle_round``, and D every round, which the sweep averages.
    The row's ``divergence_trace`` is that list.
    """
    k = SBM_NUM_COMMUNITIES
    G = build_graph(range(n), k, p_inter, random.Random(seed))
    block_of = [G.nodes[i]["community"] for i in range(n)]
    members = [[i for i in range(n) if block_of[i] == c] for c in range(k)]

    rng = random.Random(seed)
    random.seed(seed)   # agent j, ties and exploration draw from the global stream
    eligible = [i for i in range(n) if G.degree(i) > 0]
    neighbours = {i: list(G.neighbors(i)) for i in eligible}

    keep = start == "polarized"
    held = [1 if b == 0 else -1 for b in block_of] if keep else [None] * n
    states = init_opinion_states(list(range(n)))
    by_index = [states[i] for i in range(n)]

    def declare(i: int) -> int:
        pole = argmax_pole(by_index[i].q_gap, held[i])
        if keep:
            held[i] = pole
        return pole

    def signs() -> list[int]:
        if keep:
            return [argmax_pole(s.q_gap, held[i]) for i, s in enumerate(by_index)]
        return [(s.q_gap > 0) - (s.q_gap < 0) for s in by_index]

    def community_means() -> list[float]:
        return [sum(by_index[i].q_gap for i in idx) / len(idx) for idx in members]

    regime_trace: list[str] = []
    divergence_trace: list[float] = []
    first_uniform = None   # first round every block holds one opinion

    for round_n in range(1, max_rounds + 1):
        for _ in range(n if eligible else 0):
            a, b = draw_pair(eligible, neighbours, rng)
            s_a, s_b = declare(a), declare(b)
            s_a, _ = explore(s_a, epsilon)
            if both_learn:
                s_b, _ = explore(s_b, epsilon)
            if rule == "assimilation":
                assimilate(by_index[a], s_b, alpha)
                if both_learn:
                    assimilate(by_index[b], s_a, alpha)
                continue
            r = float(s_a * s_b)
            update_q_value(by_index[a], s_a, r, alpha)
            if both_learn:
                update_q_value(by_index[b], s_b, r, alpha)

        regime_trace.append(classify(community_means()))
        now = signs()
        divergence_trace.append(divergence([[now[i] for i in idx] for idx in members]))
        if first_uniform is None and all(abs(m) == 1 for m in block_order(now, block_of)[0]):
            first_uniform = round_n

    mus = community_means()
    final_signs = signs()
    mags, n_holdouts = block_order(final_signs, block_of)
    final_label = regime_trace[-1]
    # Last round whose label still differed from the final one; 0 = never.
    settle_round = next((r for r in range(len(regime_trace), 0, -1)
                         if regime_trace[r - 1] != final_label), 0)
    pol = compute_polarization_metrics(states)
    n_intra, n_inter = count_edge_types(G)

    return {
        "p_inter": p_inter,
        # The sweep condition travels with the row, so mixed CSVs stay distinguishable.
        "n_agents": n,
        "alpha": alpha,
        "epsilon": epsilon,
        "both_learn": int(both_learn),
        "rule": rule,
        "start": start,
        "seed": seed,
        "regime": final_label,
        "settle_round": settle_round,
        "rounds": len(regime_trace),
        "n_pos": pol["n_pos"],
        "n_neg": pol["n_neg"],
        "dispersion": pol["dispersion"],
        "mean_q_gap": pol["mean_q_gap"],
        "mu_c0": round(mus[0], 4),
        "mu_c1": round(mus[1], 4),
        "mag_c0": round(mags[0], 4),
        "mag_c1": round(mags[1], 4),
        "n_holdouts": n_holdouts,
        "n_undecided": final_signs.count(0),
        "first_uniform": first_uniform,
        "divergence": round(divergence_trace[-1], 4),
        "n_intra": n_intra,
        "n_inter": n_inter,
        **structure_summary(G, dict(enumerate(final_signs))),
        "divergence_trace": divergence_trace,
    }


# -- Output --------------------------------------------------------------------

def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def write_sidecar(path: Path, args: argparse.Namespace) -> None:
    """``<path>.params.json``: what produced the file."""
    sidecar = path.with_suffix(path.suffix + ".params.json")
    sidecar.write_text(json.dumps({
        "commit": git_sha(),
        "argv": sys.argv,
        "resolved": {k: v for k, v in vars(args).items() if k != "func"},
        "config": {
            "NUM_AGENTS": NUM_AGENTS,
            "EXCHANGES_PER_ROUND": EXCHANGES_PER_ROUND,
            "NETWORK_MAX_ROUNDS": NETWORK_MAX_ROUNDS,
            "SBM_NUM_COMMUNITIES": SBM_NUM_COMMUNITIES,
            "SBM_PAIRS_INTRA": SBM_PAIRS_INTRA,
            "SBM_PAIRS_INTER": SBM_PAIRS_INTER,
            "SBM_P_INTRA": SBM_P_INTRA,
            "SBM_P_INTER": SBM_P_INTER,
            "LEARNING_RATE": LEARNING_RATE,
            "EXPLORATION_RATE": EXPLORATION_RATE,
        },
        "conviction": CONVICTION,
    }, indent=2, default=str), encoding="utf-8")


def write_csv(rows: Iterable[dict], out: str, args: argparse.Namespace) -> None:
    """Stream the per-run rows, then write D's mean and sd per p_inter and round."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    total: dict[float, np.ndarray] = {}
    square: dict[float, np.ndarray] = {}
    runs: dict[float, int] = {}
    n = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = None
        for row in rows:
            trace = np.asarray(row.pop("divergence_trace"), dtype=np.float64)
            p = row["p_inter"]
            total[p] = total.get(p, 0) + trace
            square[p] = square.get(p, 0) + trace ** 2
            runs[p] = runs.get(p, 0) + 1
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            n += 1
    write_sidecar(path, args)
    curve_path = path.with_name(f"{path.stem}_divergence.csv")
    with curve_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["p_inter", "round", "mean", "sd", "n_runs"])
        for p in sorted(total):
            k = runs[p]
            mean = total[p] / k
            sd = np.sqrt(np.maximum(square[p] - k * mean ** 2, 0) / max(k - 1, 1))
            for t, (m, v) in enumerate(zip(mean, sd), 1):
                w.writerow([p, t, round(m, 6), round(v, 6), k])
    write_sidecar(curve_path, args)
    print(f"\nwrote {n} rows to {path}\n      D per round to {curve_path.name}, sidecars beside")


# -- Commands ------------------------------------------------------------------

def _run_task(task: tuple[float, int], **kwargs) -> dict:
    """``run_one`` on one (p_inter, seed), picklable for the sweep's process pool. The
    trace travels as float64, a quarter of a float list's memory: ordered results
    wait in the parent, and 20,000-round lists exhausted 7 GB of RAM."""
    row = run_one(*task, **kwargs)
    row["divergence_trace"] = np.asarray(row["divergence_trace"], dtype=np.float64)
    return row


def cmd_sweep(args: argparse.Namespace) -> None:
    # One pool for the sweep: respawning it per level exhausted Windows' system
    # resources (WinError 1450) a few levels into a 19-level sweep.
    with ProcessPoolExecutor(args.jobs) as pool:
        write_csv(_sweep_rows(args, pool), args.out, args)


def _sweep_rows(args: argparse.Namespace, pool: ProcessPoolExecutor) -> Iterator[dict]:
    """Every (p_inter, seed) row in task order, as the pool returns them."""
    one = partial(_run_task, n=args.agents, alpha=args.alpha, epsilon=args.epsilon,
                  both_learn=args.both_learn, start=args.start, max_rounds=args.max_rounds,
                  rule=args.rule)
    tasks = [(p, seed) for p in args.ladder for seed in range(1, args.reps + 1)]
    results = pool.map(one, tasks, chunksize=max(1, args.reps // 4))
    for i, row in enumerate(results, 1):
        yield row
        if i % args.reps == 0:
            print(f"  p_inter={row['p_inter']:<5}  {args.reps} runs done", flush=True)


def cmd_selftest(_args: argparse.Namespace) -> None:
    """Check the invariants the benchmark rests on. Needs no logs and no GPU."""
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"{'ok  ' if cond else 'FAIL'} {msg}")

    def edge_sets(G: nx.Graph) -> tuple[set, set]:
        """(within, cross) edges."""
        within = {frozenset(e) for e in G.edges
                  if G.nodes[e[0]]["community"] == G.nodes[e[1]]["community"]}
        return within, {frozenset(e) for e in G.edges} - within

    # 1. p_inter = 0 is two complete, disconnected blocks.
    ps = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
    graphs = {p: build_graph(range(NUM_AGENTS), SBM_NUM_COMMUNITIES, p, random.Random(7))
              for p in ps}
    check(count_edge_types(graphs[0.0]) == (2 * SBM_PAIRS_INTRA, 0)
          and not nx.is_connected(graphs[0.0]),
          "p_inter=0: complete blocks, no cross edge, disconnected")

    # 2. One seed's graphs nest: raising p_inter only adds cross edges and removes
    #    within edges, so a coupling contrast is paired.
    check(all(edge_sets(graphs[p])[0] >= edge_sets(graphs[q])[0]
              and edge_sets(graphs[p])[1] <= edge_sets(graphs[q])[1]
              for p, q in zip(ps, ps[1:])),
          f"seed 7: graphs nest across p_inter {ps}")

    # 3. Realized densities are the canon's (200 seeds, tolerance ~3 SE).
    counts = np.array([count_edge_types(build_graph(range(NUM_AGENTS), SBM_NUM_COMMUNITIES,
                                                    0.3, random.Random(s))) for s in range(200)])
    d_within = counts[:, 0].mean() / (2 * SBM_PAIRS_INTRA)
    d_cross = counts[:, 1].mean() / SBM_PAIRS_INTER
    check(abs(d_cross - 0.3) < 0.01 and abs(d_within - 0.7) < 0.01,
          f"p_inter=0.3: cross density {d_cross:.3f}, within {d_within:.3f}")

    # 4. A round is N exchanges, the per-agent clock; the benchmark uses its own N.
    check(EXCHANGES_PER_ROUND == NUM_AGENTS,
          f"exchanges/round {EXCHANGES_PER_ROUND} == N {NUM_AGENTS} (config.py)")

    # 5. fit.py's closed-form ΔQ against a hand trace: at a = 1/2, Q(+1) over r = 1, -1, 1
    #    is 0.5, -0.25, 0.375 and Q(-1) over r = 1 is 0.5, so ΔQ = -0.125.
    check(delta_q([1, 1, -1, 1], [1, -1, 1, 1], 0.5) == -0.125,
          "delta_q matches the hand trace (-0.125)")

    # 6. ... and equals the benchmark's update_q_value over the same entries.
    g = random.Random(11)
    poles = [g.choice((1, -1)) for _ in range(400)]
    rewards = [g.choice((1.0, -1.0)) for _ in poles]
    state = init_opinion_states([0])[0]
    for p, r in zip(poles, rewards):
        update_q_value(state, p, r, LEARNING_RATE)
    worst = abs(state.q_gap - delta_q(poles, rewards, LEARNING_RATE))
    check(worst < 1e-12, f"delta_q == update_q_value over 400 entries (diff {worst:.1e})")

    # 7. explore() draws nothing at ε = 0, so ε = 0 leaves every stream as it was.
    random.seed(5)
    before = random.getstate()
    check(all(explore(x, 0.0) == (x, False) for x in (-1, 1))
          and random.getstate() == before, "explore at epsilon = 0: no draw, no flip")

    # 8. D = |m_0 - m_1| / 2 on the blocks' mean poles.
    check(divergence([[1, 1], [-1, -1]]) == 1 and divergence([[1, -1], [-1, 1]]) == 0
          and abs(divergence([[1, 1, -1], [-1, -1, 1]]) - 1 / 3) < 1e-12, "divergence on hand cases")

    # 9. A run is a function of its seed in every mode, so --jobs cannot change a row.
    check(all(run_one(0.1, 3, **kw) == run_one(0.1, 3, **kw)
              for kw in ({}, {"epsilon": 0.1, "both_learn": True},
                         {"start": "coin", "epsilon": 0.1}, {"n": 10},
                         {"rule": "assimilation", "epsilon": 0.1, "both_learn": True})),
          "run_one reproduces per seed in every mode")

    # 10. assimilate() is the shared trace form over the partner's poles, and A = ΔQ.
    state = init_opinion_states([0])[0]
    for p in poles:
        assimilate(state, p, LEARNING_RATE)
    worst = abs(state.q_gap - trace(poles, LEARNING_RATE))
    check(worst < 1e-12, f"assimilate == trace over 400 poles (diff {worst:.1e})")

    # 11. Without a cross edge or exploration, the polarized start is a fixed point.
    row = run_one(0.0, 3, both_learn=True, rule="assimilation", max_rounds=50)
    check(min(row["divergence_trace"]) == 1 and row["n_holdouts"] == 0,
          "assimilation at p_inter=0, epsilon=0: D = 1 every round")

    print(f"ok   alpha={LEARNING_RATE} p_inter={SBM_P_INTER} p_intra={SBM_P_INTRA} "
          f"horizon={NETWORK_MAX_ROUNDS} (config.py, no stopping rule)")
    print("ok   imported, not reimplemented: network.topology.build_graph, "
          "network.matching.draw_pair, network.opinion (update_q_value, argmax_pole, "
          "explore, delta_q, trace)")

    raise SystemExit(0 if ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sweep", help="D vs coupling -- the transition curve")
    p.add_argument("--rule", choices=("sft", "assimilation"), default="sft",
                   help="sft: the canon's update (default); assimilation: each learner "
                        "traces its partner's argued pole")
    p.add_argument("--epsilon", type=float, default=EXPLORATION_RATE,
                   help=f"exploration rate (default {EXPLORATION_RATE}, "
                        "config.EXPLORATION_RATE; 0 draws nothing)")
    p.add_argument("--alpha", type=float, default=LEARNING_RATE,
                   help=f"learning rate (default {LEARNING_RATE}, the canon's; § 5.4 uses 0.01)")
    p.add_argument("--both-learn", action="store_true",
                   help="agent j explores and learns too")
    p.add_argument("--start", choices=("polarized", "coin"), default="polarized",
                   help="polarized: each community on its own pole (default); "
                        "coin: Q = (0, 0), each pole drawn")
    p.add_argument("--agents", type=int, default=NUM_AGENTS,
                   help=f"N, with N exchanges per round (default {NUM_AGENTS})")
    p.add_argument("--ladder", type=float, nargs="+", default=list(DEFAULT_LADDER),
                   help="cross-pair probabilities p_inter, within pairs at 1 - p_inter "
                        "(default 0-0.50 in steps of 0.01)")
    p.add_argument("--reps", type=int, default=50,
                   help="runs per p_inter, seeds 1 .. reps (default 50)")
    p.add_argument("--max-rounds", type=int, default=NETWORK_MAX_ROUNDS,
                   help=f"rounds per run (default {NETWORK_MAX_ROUNDS}, the simulation's)")
    p.add_argument("--jobs", type=int, default=1,
                   help="worker processes (default 1); rows are identical at any count")
    p.add_argument("--out", required=True,
                   help="per-run rows go here as CSV; D per round and sidecars beside")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("selftest", help="check the invariants the benchmark rests on")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
