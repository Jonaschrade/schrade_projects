"""The exchange audit: one cell's sub-phase (analysis/schedule.md § Exchange audit).

A cell is the model and topic config.py resolves (``SIM_LLM_MODEL``, ``SIM_TOPIC``).
Meetings run ``network.exchange.run_exchange``, the code a run's exchange runs, with
both argued positions assigned; constructed windows draw their entries from the
phase's meetings, rendered through ``memory.store.render`` and declared through
``Agent.declare``. The conditions differ only in what a declaration shows of an
entry, so one audit serves both: every declaration with memory is made under each.

Sub-phases (``--sub``), each one job per cell under its own logged ``SIM_SEED``:
  meetings  a meeting per stratum (i's, j's argued position), then each side's
            declarations, from empty memory and from its one entry under full / own
  windows   the Order (8), Settled (2) and Serial-position (18) windows, each declared
            under full / own on the same entries, once the phase's meetings are
            written; a unit's ``entries`` name their meetings
            ([partner, meeting's index among the units, own role])

Every meeting's turns are self-read blind, as in a run.

Phases (``--phase``): calibration (one unit per key), audit (TARGETS per key).
Units run replicate-major, so a job cut short stays balanced, and a rerun at the same
commit resumes the units written, under its own seed, logged with its node in its
``start`` event; at another commit it aborts, and so do windows on its meetings.
A ParseAbort drops its unit, logged as ``dropped``; a rerun fills it.

Output: ``<out>/<topic>_<model>/<phase>_<sub>/`` (``:`` in the tag written ``-``),
``events.jsonl`` and ``params.json``; ``--out`` defaults to ``logs/audit`` under the
working directory, as a run's goes to ``logs/``, and to ``logs/audit/calibration`` for
the calibration.

Usage
-----
    SIM_LLM_MODEL=gemma4-nodraft:e4b SIM_TOPIC=coffee_tea \\
        python analysis/audit/build_audit.py --phase audit --sub meetings  # via run_audit.sh
    python analysis/audit/build_audit.py --stub --phase calibration --sub meetings --out $S
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import socket
import sys
import time
from collections import Counter
from pathlib import Path

_REPO = next(
    (d for d in Path(__file__).resolve().parents if (d / "simulation" / "config.py").is_file()),
    None,
)
if _REPO is None:
    raise SystemExit("cannot locate the repo root (no ancestor contains simulation/config.py)")
sys.path.insert(0, str(_REPO / "simulation"))

from agents.agent import Agent  # noqa: E402
from agents.llm import LLM, ParseAbort  # noqa: E402
from agents.names import NAMES  # noqa: E402
from config import CONDITIONS, LLM_MODEL, NETWORK_MAX_ROUNDS, SEED, TOPIC_KEY  # noqa: E402
from memory.store import Entry  # noqa: E402
from network.exchange import run_exchange  # noqa: E402
from network.logger import SimulationLogger  # noqa: E402

POSITIONS = (-1, 1)
STRATA = tuple(f"{se},{sr}" for se in POSITIONS for sr in POSITIONS)
# What a declaration at H shows in expectation: N exchanges a round, two sides each.
SETTLED_LEN = 2 * NETWORK_MAX_ROUNDS
# Serial positions: lags 2^j - 1 from the latest entry, even on the log scale the
# geometric weight decays on, and the first entry.
SERIAL_K = tuple(sorted({1} | {SETTLED_LEN + 1 - 2 ** j for j in range(SETTLED_LEN.bit_length())}))
# Units per key in the audit phase, each number to 5% of its scale at 95% in the worst
# case (schedule.md § Exchange audit): 385 calls hold a probability to ±0.05, a mean y
# to ±0.1. Meetings: 2N probes per pair of argued positions, and a settled window's
# SETTLED_LEN distinct meetings in one orientation; serial: its mean pooled over two
# poles; order: half of a contrast of two means.
N = max(math.ceil(385 / 2), SETTLED_LEN)
TARGETS = {"meeting": N, "order": 770, "settled": 385, "serial": math.ceil(385 / 2)}
NEIGHBOURHOOD = 10       # the expected degree, 9 + p_inter, rounded up
SUBS = {"meetings": ("meeting",), "windows": ("order", "settled", "serial")}
CELL = f"{TOPIC_KEY}_{LLM_MODEL.replace(':', '-')}"


def window_keys(kind: str) -> list[str]:
    """Every window of ``kind``, as its key."""
    if kind == "order":
        return [f"order:{a1}{a2}:{first}" for a1 in "ad" for a2 in "ad" for first in (1, -1)]
    if kind == "settled":
        return [f"settled:{p}" for p in POSITIONS]
    return [f"serial:{p}:{k}" for p in POSITIONS for k in SERIAL_K]


def read_meetings(path: Path) -> tuple[dict, dict]:
    """The meetings written, each in both orientations: names and texts by (meeting's
    index among the units, own role), and each [index, role] by (own, partner) argued
    position."""
    turns, pool = {}, {}
    for m, unit in enumerate(read_units(path)):
        sides = unit["exchange"]["sides"]
        for role, other in (("i", "j"), ("j", "i")):
            own, partner = sides[role], sides[other]
            turns[(m, role)] = (own["agent"], own["turn"]["text"],
                                partner["agent"], partner["turn"]["text"])
            pool.setdefault((own["s"], partner["s"]), []).append([m, role])
    return turns, pool


def draw_window(key: str, settled: dict[int, list], rep: int,
                pool: dict) -> tuple[str, list[list]] | None:
    """(agent, entries) for one call of window ``key``, each entry a meeting drawn at
    its two argued positions. Serial call ``rep`` reuses settled call ``rep`` on its
    position, entry k redrawn against the opposite pole. None while a settled call or
    a pair's meetings are missing."""
    kind, *spec = key.split(":")
    if kind == "serial":
        p, k = int(spec[0]), int(spec[1])
        if rep >= len(settled.get(p, [])) or (p, -p) not in pool:
            return None
        agent, entries = settled[p][rep]
        entries = list(entries)
        entries[k - 1] = [entries[k - 1][0], *random.choice(pool[(p, -p)])]
        return agent, entries
    agent = random.choice(NAMES)
    others = [n for n in NAMES if n != agent]
    if kind == "order":
        agree, first = spec[0], int(spec[1])
        pairs = [(pole, pole * (1 if a == "a" else -1)) for pole, a in zip((first, -first), agree)]
        if not all(pair in pool for pair in pairs):
            return None
        return agent, [[random.choice(others), *random.choice(pool[pair])] for pair in pairs]
    p = int(spec[0])
    role = random.choice("ij")   # one orientation, so no turn recurs as the partner's
    exchanges = [e for e in pool.get((p, p), []) if e[1] == role]
    if not exchanges:
        return None
    drawn = (random.sample(exchanges, SETTLED_LEN) if len(exchanges) >= SETTLED_LEN
             else random.choices(exchanges, k=SETTLED_LEN))   # calibration's single meeting
    hood = random.sample(others, NEIGHBOURHOOD)
    return agent, [[random.choice(hood), *e] for e in drawn]


def entries_of(agent: str, spec: list[list], turns: dict) -> list[Entry]:
    """A window's entries: each meeting's turns under the window's names, since a turn
    may name its speaker (partner-blind, it cannot name the other)."""
    entries = []
    for i, (partner, m, role) in enumerate(spec):
        own_name, own_text, partner_name, partner_text = turns[(m, role)]
        entries.append(Entry(i, 1, partner, re.sub(rf"\b{own_name}\b", agent, own_text),
                             re.sub(rf"\b{partner_name}\b", partner, partner_text)))
    return entries


def meeting(llm: LLM, key: str) -> dict:
    si, sj = map(int, key.split(","))
    i_name, j_name = random.sample(NAMES, 2)
    i, j = Agent(i_name, llm), Agent(j_name, llm)
    record = run_exchange(i, j, 1, argued=(si, sj), store=True)
    declarations = {}
    for role, agent in (("i", i), ("j", j)):
        # With no entry the conditions render alike, so one empty declaration serves both.
        declarations[role] = {"empty": agent.declare([], CONDITIONS[0]),
                              **{c: agent.declare(agent.memory.entries, c) for c in CONDITIONS}}
    return {"exchange": record, "declarations": declarations}


def window_call(llm: LLM, agent: str, spec: list[list], turns: dict) -> dict:
    entries, declarer = entries_of(agent, spec, turns), Agent(agent, llm)
    return {"agent": agent, "entries": spec,
            "declaration": {c: declarer.declare(entries, c) for c in CONDITIONS}}


def read_units(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [r for r in map(json.loads, fh) if r["type"] == "unit"]


def units_per_key(phase: str, kind: str, reps: int | None) -> int:
    return 1 if phase == "calibration" else (reps or TARGETS[kind])


def plan(phase: str, kinds: tuple[str, ...], counts: Counter, reps: int | None) -> list[str]:
    """The keys still to run, replicate-major; within a
    replicate, settled before serial, which reuses settled calls."""
    order = []
    for j, kind in enumerate(kinds):
        keys = STRATA if kind == "meeting" else window_keys(kind)
        target = units_per_key(phase, kind, reps)
        order += [(i, j, k) for k in keys for i in range(counts[k], target)]
    return [k for _, _, k in sorted(order, key=lambda t: t[:2])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", required=True, choices=("calibration", "audit"))
    ap.add_argument("--sub", required=True, choices=tuple(SUBS))
    ap.add_argument("--out", help="parent of the cell directory (default: logs/audit, "
                                  "logs/audit/calibration for the calibration)")
    ap.add_argument("--stub", action="store_true", help="canned replies, no Ollama")
    ap.add_argument("--reps", type=int, help="units per key in place of the audit's targets")
    args = ap.parse_args()

    out = Path(args.out or ("logs/audit/calibration" if args.phase == "calibration" else "logs/audit")) / CELL
    turns, pool = {}, {}
    if args.sub == "windows":
        path = out / f"{args.phase}_meetings" / "events.jsonl"
        turns, pool = read_meetings(path)
        owed = len(STRATA) * units_per_key(args.phase, "meeting", args.reps) - len(turns) // 2
        # The stub's replies drop a few meetings on purpose.
        if not turns or (owed > 0 and not args.stub):
            raise SystemExit(f"{path}: {owed} meetings missing; finish that sub-phase first")
    logger = SimulationLogger(run_dir=out / f"{args.phase}_{args.sub}")
    # Every job declares under both conditions, so config's single CONDITION would mislabel it.
    extra = {"phase": args.phase, "sub": args.sub, "stub": args.stub, "reps": args.reps,
             "CONDITION": list(CONDITIONS)}
    new = json.loads(json.dumps({**logger.params(), **extra}, default=str))
    if args.sub == "windows":
        path = out / f"{args.phase}_meetings" / "params.json"
        commit = json.loads(path.read_text(encoding="utf-8"))["git_commit"]
        if commit != new["git_commit"]:
            raise SystemExit(f"{path}: meetings at {commit}, this checkout at {new['git_commit']}")
    params_path = logger.run_dir / "params.json"
    if params_path.is_file():
        old = json.loads(params_path.read_text(encoding="utf-8"))
        diff = sorted(k for k in set(old) | set(new)
                      if k not in ("SEED", "node") and old.get(k) != new.get(k))
        if diff:
            raise SystemExit(f"{params_path} differs in {diff}: a resume must not mix settings")
    else:
        logger.log_params(extra)
    random.seed(SEED)   # names, assigned strata, constructed windows; LLM seeds have their own
    logger.write({"type": "start", "seed": SEED, "node": socket.gethostname(), "time": time.time()})

    kinds = SUBS[args.sub]
    done = read_units(logger.events_path)
    counts = Counter(u["key"] for u in done)
    settled: dict[int, list] = {}   # serial calls reuse the settled calls written
    for u in done:
        if u["kind"] == "settled":
            settled.setdefault(int(u["key"].split(":")[1]), []).append((u["agent"], u["entries"]))
    llm = LLM(client=StubClient() if args.stub else None, on_parse=logger.log_parse)

    def run(key: str) -> dict | None:
        """One unit, written; None if dropped or not yet runnable."""
        kind = key.split(":")[0] if ":" in key else kinds[0]
        t0 = time.time()
        try:
            if kind == "meeting":
                unit = meeting(llm, key)
            else:
                drawn = draw_window(key, settled, counts[key], pool)
                if drawn is None:
                    return None   # its settled call or meeting was dropped; a rerun fills it
                unit = window_call(llm, *drawn, turns)
        except ParseAbort as exc:
            logger.write({"type": "dropped", "kind": kind, "key": key, "seed": SEED,
                          "error": str(exc)})
            return None
        if kind == "settled":
            settled.setdefault(int(key.split(":")[1]), []).append((unit["agent"], unit["entries"]))
        counts[key] += 1
        logger.write({"type": "unit", "kind": kind, "key": key, "seed": SEED,
                      "seconds": round(time.time() - t0, 2), **unit})
        return unit

    print(f"{CELL} {args.phase}_{args.sub}: {len(done)} units written, seed {SEED}", flush=True)
    todo = plan(args.phase, kinds, counts, args.reps)
    written = 0
    for i, key in enumerate(todo):
        written += run(key) is not None
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)}, {logger.elapsed():.0f}s", flush=True)
    print(f"{written}/{len(todo)} units written in {logger.elapsed():.0f}s", flush=True)


class StubClient:
    """Canned Ollama replies for a local run: free text for turns, and under a schema
    a random answer with logprobs over every label; about 2% of schema replies are
    malformed, so the retry and the drop path run too."""

    TEXT = "I see it differently, and here is why: the evidence points my way."

    def generate(self, *, model, prompt, system, think, stream, options, format=None,
                 logprobs=False, top_logprobs=0) -> dict:
        rng = random.Random(options["seed"])
        n = len(prompt) // 4
        if format is None:
            return {"response": self.TEXT, "prompt_eval_count": n}
        if rng.random() < 0.02:
            return {"response": "{", "prompt_eval_count": n, "logprobs": []}
        field = next(iter(format["properties"]))
        labels = format["properties"][field]["enum"]
        weights = [rng.random() for _ in labels]
        lp = {label: math.log(w / sum(weights)) for label, w in zip(labels, weights)}
        answer = rng.choices(labels, weights)[0]
        tokens = [{"token": '{"', "logprob": 0.0, "top_logprobs": []},
                  {"token": field, "logprob": 0.0, "top_logprobs": []},
                  {"token": '": "', "logprob": 0.0, "top_logprobs": []},
                  {"token": answer, "logprob": lp[answer],
                   "top_logprobs": [{"token": k, "logprob": v} for k, v in lp.items()]},
                  {"token": '"}', "logprob": 0.0, "top_logprobs": []}]
        return {"response": json.dumps({field: answer}), "prompt_eval_count": n,
                "logprobs": tokens}


if __name__ == "__main__":
    main()
