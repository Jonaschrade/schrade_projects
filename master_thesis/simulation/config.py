"""
Global configuration: every tunable constant and what it means (README "Design").

Terminology: a turn is one speaker's message; an exchange is agent i and agent j,
from their declarations to their stored entries; a round is
EXCHANGES_PER_ROUND exchanges, the snapshot unit.

Every name in run_sweep.SUPPORTED_OVERRIDES is read here as SIM_<NAME>. A job reads
this file when it starts, so run_stage.sh sets them per sub-stage; the defaults
document interactive runs. What a run used is its params.json.
NUM_AGENTS and NETWORK_MAX_ROUNDS have no override, deliberately (CLAUDE.md).
"""

import os
import secrets

# -- Run ----------------------------------------------------------------------
NUM_AGENTS         = 20   # set, not derived (analysis/schedule.md § Sensitivity)
# H, fixed by design for the pilot and the runs (analysis/schedule.md § Runs, Horizon).
NETWORK_MAX_ROUNDS = 100
# One expected exchange per agent per round as agent i, the random sequential
# update convention; the benchmark sets it to its own N.
EXCHANGES_PER_ROUND = NUM_AGENTS
# ε: the probability that a side argues the other pole (network.opinion.explore),
# the canon's ε-greedy rate (Banisch & Olbrich 2019).
EXPLORATION_RATE = 0.1

# -- Initialization -----------------------------------------------------------
# The start, every agent on a pole (README "Initialization"): "polarized", the
# canon's § 5.4, community 0 on +1 and community 1 on -1 (the runs and pilot phase 2);
# "coin", each pole drawn, +1 with P_START_PLUS (pilot phase 1).
START = os.getenv("SIM_START", "polarized")
if START not in ("coin", "polarized"):
    raise ValueError(f"SIM_START={START!r}; expected coin or polarized")
P_START_PLUS = 0.5

# -- Topology -----------------------------------------------------------------
# The canon's two communities (Banisch & Olbrich 2019, § 5.4): a cross pair linked
# with SBM_P_INTER, a within pair with 1 - SBM_P_INTER (README "Topology").
SBM_NUM_COMMUNITIES = 2


def _pair_counts(n: int, k: int) -> tuple[int, int]:
    """(within pairs per block, cross pairs); needs two equal blocks."""
    if k != 2 or n % 2:
        raise ValueError(f"the canon's topology needs exactly 2 equal blocks, got n={n} k={k}")
    s = n // 2
    return s * (s - 1) // 2, s * s


SBM_PAIRS_INTRA, SBM_PAIRS_INTER = _pair_counts(NUM_AGENTS, SBM_NUM_COMMUNITIES)

# A silently ignored override would run a whole job at the wrong physics.
for _legacy in ("SIM_SBM_P_INTRA", "SIM_SBM_M_INTER", "SIM_SBM_M_INTRA"):
    if os.getenv(_legacy) is not None:
        raise RuntimeError(
            f"{_legacy} is set but no longer read: the topology takes only "
            f"SIM_SBM_P_INTER and links within pairs at 1 - SIM_SBM_P_INTER."
        )

# 0 is legal: two complete, disconnected blocks; 0.5 is ER.
SBM_P_INTER = float(os.getenv("SIM_SBM_P_INTER", "0.15"))
if not 0.0 <= SBM_P_INTER <= 1.0:
    raise ValueError(f"SBM_P_INTER={SBM_P_INTER} outside [0, 1]")
SBM_P_INTRA = 1.0 - SBM_P_INTER

# -- Topic --------------------------------------------------------------------
# One topic per run. `text` (+1) and `opposite` (-1) state one comparison both ways,
# so each pole is a claim of its own, shown as it stands; no trailing period. `opener`
# names the +1 pole first, `opener_swapped` the -1 pole; agents.prompts.draw_opener
# draws one per exchange, which both turns answer. `issue` names the subject for the
# setting (agents.prompts.SETTING) and neither pole. Rationale: topic_candidates.md.
TOPICS: dict[str, dict] = {
    "nuclear": {
        "label": "Nuclear power",
        "issue": "energy policy",
        "text": "Nuclear power does more good than harm",
        "opposite": "Nuclear power does more harm than good",
        "opener": "Does nuclear power do more good than harm, or more harm than good?",
        "opener_swapped": "Does nuclear power do more harm than good, or more good than harm?",
    },
    "renewables": {
        "label": "Renewables vs. fossil fuels",
        "issue": "energy policy",
        "text": "Renewables should be prioritized over fossil fuels",
        "opposite": "Fossil fuels should be prioritized over renewables",
        "opener": "Which should be prioritized, renewables or fossil fuels?",
        "opener_swapped": "Which should be prioritized, fossil fuels or renewables?",
    },
    "coffee_tea": {
        "label": "Coffee vs. tea",
        "issue": "hot drinks",
        "text": "Coffee is better than tea",
        "opposite": "Tea is better than coffee",
        "opener": "Which is better, coffee or tea?",
        "opener_swapped": "Which is better, tea or coffee?",
    },
}
TOPIC_KEY = os.getenv("SIM_TOPIC", "nuclear")
if TOPIC_KEY not in TOPICS:
    raise ValueError(f"SIM_TOPIC={TOPIC_KEY!r} is not a key of TOPICS: {sorted(TOPICS)}")
TOPIC_LABEL          = TOPICS[TOPIC_KEY]["label"]
TOPIC_ISSUE          = TOPICS[TOPIC_KEY]["issue"]
TOPIC_TEXT           = TOPICS[TOPIC_KEY]["text"]
TOPIC_TEXT_OPPOSITE  = TOPICS[TOPIC_KEY]["opposite"]
TOPIC_OPENER         = TOPICS[TOPIC_KEY]["opener"]
TOPIC_OPENER_SWAPPED = TOPICS[TOPIC_KEY]["opener_swapped"]

# -- Model --------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11500")
# Overridable so the audit's cells queue beside default-model jobs. -nodraft tags:
# CLAUDE.md § Environment, RUNBOOK "Pulling models".
LLM_MODEL = os.getenv("SIM_LLM_MODEL", "qwen3.5-nodraft:4b")
# Prompt plus output, every call; an unused context costs only memory. The largest
# prompt is the final declaration, which shows every entry of the run: up to ~300 entries at ~150
# tokens each, ~46k (the five-position pilot's busiest agent and longest turns).
# Every call logs prompt_eval_count, and the pilot checks the largest against this.
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", 65536))
# Re-asks of an unparseable schema reply, each under a fresh seed, before the run
# aborts. The audit sets it (analysis/schedule.md § Exchange audit).
PARSE_MAX_RETRIES = 1

# -- Memory -------------------------------------------------------------------
# False: the declaration shows no window and nothing is stored.
MEMORY_ENABLED = os.getenv("SIM_MEMORY_ENABLED", "1") == "1"
# What a declaration shows of each entry (README "Memory"): "full" both turns, "own"
# the agent's own turn alone. A declaration shows every entry stored so far.
CONDITIONS = ("full", "own")
CONDITION = os.getenv("SIM_CONDITION", "full")
if CONDITION not in CONDITIONS:
    raise ValueError(f"SIM_CONDITION={CONDITION!r}; expected full or own")
for _retired in ("SIM_FEEDBACK_MEMORY",):
    if _retired in os.environ:
        raise ValueError(f"{_retired} is retired; set SIM_CONDITION (full / own)")
# -- SFT model (benchmark only) -----------------------------------------------
# α of the canon's update Q(o) <- (1 - α) Q(o) + α r (Banisch & Olbrich 2019, § 3.1).
# Runs carry no Q; a fitted rate is read against this one.
LEARNING_RATE = 0.05

# -- Reproducibility ----------------------------------------------------------
# Seeds the global `random` stream (names, starts, graph, pairing, exploration) and
# the per-call streams of agents.prompts and agents.llm. Drawn from OS entropy
# when unset and logged, so replicates stay independent.
_env_seed = os.getenv("SIM_SEED")
SEED = int(_env_seed) if _env_seed is not None else secrets.randbits(64)
