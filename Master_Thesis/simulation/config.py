"""
Global configuration: single source of truth for all tunable parameters.

Terminology: turn = one respond() call; exchange = both agents speak once;
interaction = one expresser, one responder, one exchange, one Q-update (expresser
only); round = INTERACTIONS_PER_ROUND interactions, the snapshot unit.

Environment overrides
---------------------
Every name in run_sweep.SUPPORTED_OVERRIDES is read here as SIM_<NAME>. On the
cluster run_stage.sh sets them per sub-stage, because this file is read when a
job *starts*, possibly days after qsub; the defaults below document interactive
runs only. What a run computed is its params.json (logger.log_params).

Deliberately without override: NUM_AGENTS and NETWORK_MAX_ROUNDS (fixed for the
whole confirmatory schedule, edited here before submitting), STRENGTH_DELTA
(derived) and INITIAL_EDGE_STRENGTH (the unit of the valuation axis).
SIM_STRENGTH_CAP, SIM_REWARD_WINDOW_M and SIM_NEW_EDGE_STRENGTH exist only for the
benchmark's edge-layer screens (D-04 - D-06); no sub-stage sets them.
SIM_SBM_P_INTER / SIM_SBM_P_INTRA raise at import: the topology takes counts.
"""

import os
import secrets

# -- Models -------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11500")
# One model per run: agent, generator and grader at once. Overridable so Stage 8
# can queue beside default-model jobs. -nodraft = qwen3.8:27b without speculative
# decoding, whose load crashes (RUNBOOK "Modelle pullen", step 3).
LLM_MODEL   = os.getenv("SIM_LLM_MODEL", "qwen3.8-nodraft:27b")
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", 4096))

# -- Memory -------------------------------------------------------------------
# Stage 4's ablation arm. False: no retrieval into prompts, no reflection; writing
# continues, so the arms differ only in what reaches the LLM. A flag of its own so
# the arm is visible in params.json.
MEMORY_ENABLED        = os.getenv("SIM_MEMORY_ENABLED", "1") == "1"

REFLECT_EVERY         = 5    # trigger reflection every N simulation rounds
MAX_MEMORIES_SEED     = 15   # recent memories fed into the reflection prompt
MAX_MEMORIES_RETRIEVE = 5    # relevant memories retrieved per agent

MEMORY_PERSIST = False         # persist ChromaDB to disk (True) or keep in-memory (False)
MEMORY_DIR     = "./memory.db" # path used when MEMORY_PERSIST is True

# -- Retrieval scoring weights -------------------------------------------------
#   Composite score =
#   recency·w + importance·w + relevance·w + stance_congruence·w
WEIGHT_RECENCY           = 0.25
WEIGHT_IMPORTANCE        = 0.25
WEIGHT_RELEVANCE         = 0.25
# Motivated-recall knob: biases retrieval toward memories tagged with the
# stance being expressed (see README "Memory system").
WEIGHT_STANCE_CONGRUENCE = 0.25

# -- Social Feedback Theory (SFT): Q-value learning ---------------------------
# TD update from Banisch & Olbrich (2019): Q(o_i) ← (1 − α) · Q(o_i) + α · r
LEARNING_RATE = 0.1  # α: step size for the Q-value TD update

# SFT expression gate: rejection-sampling enforcement of the drawn stance (README
# "SFT expression gate"). False = one ungated draft, the ablation arm.
SFT_GATE_ENABLED      = True
SFT_GATE_MAX_ATTEMPTS = 5   # attempt budget per cascade stage (soft prompt / context-free fallback)

# Conviction cue: the stance header carries max(0, stance * q_gap) / 2, banded
# (README "The conviction cue"). On for every arm; SIM_CONVICTION_CUE=0 is the
# ablation and reproduces the uncued prompt byte for byte.
CONVICTION_CUE_ENABLED = os.getenv("SIM_CONVICTION_CUE", "1") == "1"

# Stance bridge. Off (default): the raw persona sits on the identity line and the
# gate arbitrates when it disagrees with the anchor. On: the persona reaches
# generation only via _mold_stance's cached per-stance justification.
STANCE_BRIDGE_ENABLED = os.getenv("SIM_STANCE_BRIDGE", "0") == "1"

# Softmax inverse temperature β for opinion expression.
# p(+1) = exp(β·q_pos) / (exp(β·q_pos) + exp(β·q_neg))
# β = 0  → 50/50 random
# β > 0  → preference for the higher Q-value
# β → ∞  → deterministic argmax
OPINION_BETA = float(os.getenv("SIM_OPINION_BETA", "5.0"))

# Responder-selection inverse temperature β_sel: weight exp(β_sel · mean reward the
# expresser received on that edge). 0 = uniform (Banisch & Olbrich 2019); an empty
# history counts as 0.0.
RESPONDER_SELECTION_BETA = float(os.getenv("SIM_RESPONDER_SELECTION_BETA", "0.0"))

# -- Network simulation --------------------------------------------------------
NUM_AGENTS         = 20  # total agents in the network graph (pairwise mode always uses 2)
NETWORK_MAX_ROUNDS = 100  # total snapshot rounds

# Network mode only (pairwise fixes 2 in main_pairwise.py): one expected
# interaction per agent per round, the random sequential update convention.
INTERACTIONS_PER_ROUND = NUM_AGENTS

# Graph topology: two-block model with an EXACT edge count per layer (G(n, M) per
# stratum), not per-pair probabilities. With N fixed the coupling axis is a finite
# integer one, and a probabilistic rung would be a mixture over topology sizes
# (at p = 0.02 over 100 cross pairs, 13.3 % of draws have no cross edge).
# SBM_M_INTER is the primary experimental variable (README "Initial topology").
SBM_NUM_COMMUNITIES = 2    # required by the count parameterisation


def _block_sizes(n: int, k: int) -> list[int]:
    """Divide n agents as evenly as possible into k blocks."""
    base, remainder = divmod(n, k)
    return [base + (1 if i < remainder else 0) for i in range(k)]


def _pair_counts(n: int, k: int) -> tuple[int, int]:
    """(within-block pairs per block, cross-block pairs); needs two equal blocks."""
    sizes = _block_sizes(n, k)
    if k != 2 or len(set(sizes)) != 1:
        raise ValueError(
            f"the count parameterisation needs exactly 2 equal blocks, got "
            f"k={k} sizes={sizes}. Either restore a probability-based topology "
            f"or define one M per block pair."
        )
    s = sizes[0]
    return s * (s - 1) // 2, s * s


SBM_PAIRS_INTRA, SBM_PAIRS_INTER = _pair_counts(NUM_AGENTS, SBM_NUM_COMMUNITIES)

# A silently ignored p-based override would run a whole job at the wrong physics.
for _legacy in ("SIM_SBM_P_INTER", "SIM_SBM_P_INTRA"):
    if os.getenv(_legacy) is not None:
        raise RuntimeError(
            f"{_legacy} is set but no longer read: the topology is parameterised "
            f"by exact edge counts. Use SIM_SBM_M_INTER / SIM_SBM_M_INTRA "
            f"(counts, not probabilities)."
        )

# Within-block edges PER BLOCK, of 45: 27/45 = 0.6, so the ER rung is exactly
# m_inter = 60 (final_schedule.md, "Why m_intra = 27").
SBM_M_INTRA = int(os.getenv("SIM_SBM_M_INTRA", "27"))
# Cross-block edges in total, of 100. 0 is legal: two disconnected blocks.
SBM_M_INTER = int(os.getenv("SIM_SBM_M_INTER", "15"))

for _name, _m, _pairs in (
    ("SBM_M_INTRA", SBM_M_INTRA, SBM_PAIRS_INTRA),
    ("SBM_M_INTER", SBM_M_INTER, SBM_PAIRS_INTER),
):
    if not 0 <= _m <= _pairs:
        raise ValueError(f"{_name}={_m} outside [0, {_pairs}]")

# Derived densities, not inputs: kept for params.json and p-keyed analysis.
SBM_P_INTRA = SBM_M_INTRA / SBM_PAIRS_INTRA
SBM_P_INTER = SBM_M_INTER / SBM_PAIRS_INTER

# Graph dynamics: False holds the graph fixed; True enables endogenous rewiring
# (README "Extension: endogenous tie formation").
GRAPH_DYNAMIC = os.getenv("SIM_GRAPH_DYNAMIC", "0") == "1"

# Edge layer, read only when GRAPH_DYNAMIC (REWARD_WINDOW_M also by
# select_responder at β_sel > 0).
STRENGTH_CAP     = float(os.getenv("SIM_STRENGTH_CAP", "3.0"))  # ceiling on each agent's edge valuation
STRENGTH_FLOOR   = 0.0  # edge severed when either valuation falls to or below this
REWARD_WINDOW_M  = int(os.getenv("SIM_REWARD_WINDOW_M", "5"))  # rolling reward window per edge (interactions)

# Valuation at creation: founding edges at the unit, rewired ties weaker. With the
# floor at 0 and STRENGTH_DELTA derived from INITIAL_EDGE_STRENGTH, the dynamics
# depend only on CAP / INITIAL and NEW / INITIAL, so INITIAL is fixed at 1.
INITIAL_EDGE_STRENGTH = 1.0
NEW_EDGE_STRENGTH     = float(os.getenv("SIM_NEW_EDGE_STRENGTH", "0.5"))

# Rewiring rule, the RQ2 factor. A severed tie is replaced in the same step, so |E|
# is conserved.
#   "random"        blind draw over all non-neighbours (churn-only null)
#   "reward_local"  2-hop pool only: every new tie closes a triangle
#   "off"           sever without replacement
REWIRE_RULE = os.getenv("SIM_REWIRE_RULE", "reward_local")

# Partner-choice decisiveness, weight exp(REWIRE_BETA · referral estimate).
# 0 = uniform over the pool; large values risk hub collapse.
REWIRE_BETA = float(os.getenv("SIM_REWIRE_BETA", "5.0"))

# Timescale ratio w = τ_opinion / τ_rewire = STRENGTH_DELTA / (LEARNING_RATE ·
# INITIAL_EDGE_STRENGTH). w ≪ 1 recovers the static graph; w ≫ 1 lets structure
# outrun opinion. Nominal (assumes |mean reward| = 1); report the realised rewire
# rate beside it (README "Timescale separation").
REWIRE_TIMESCALE_RATIO = float(os.getenv("SIM_REWIRE_TIMESCALE_RATIO", "3.0"))

# Derived, not set directly: valuation change per interaction = mean_reward × this.
STRENGTH_DELTA = REWIRE_TIMESCALE_RATIO * LEARNING_RATE * INITIAL_EDGE_STRENGTH

# -- Discussion topic ---------------------------------------------------------
# One fixed topic per run, so the ±1 stance dimension keeps one meaning.
# Each entry is a pairwise comparison: `text` (+1) and `opposite` (-1) are the same
# comparison reversed, no trailing period (callers append one). `opener` /
# `opener_swapped` are the moderator's two wordings, differing only in which object
# comes first. Rationale: topic_candidates.md.
TOPICS: dict[str, dict] = {
    "nuclear": {
        "label": "Nuclear power",
        "text": "Nuclear power does more good than harm",
        "opposite": "Nuclear power does more harm than good",
        "opener": "Does nuclear power do more good than harm, or more harm than good?",
        "opener_swapped": "Does nuclear power do more harm than good, or more good than harm?",
    },
    "renewables": {
        "label": "Renewables vs. fossil fuels",
        "text": "Renewables should be prioritized over fossil fuels",
        "opposite": "Fossil fuels should be prioritized over renewables",
        "opener": "Which should be prioritized now, renewables or fossil fuels?",
        "opener_swapped": "Which should be prioritized now, fossil fuels or renewables?",
    },
    "coffee_tea": {
        "label": "Coffee vs. tea",
        "text": "Coffee is better than tea",
        "opposite": "Tea is better than coffee",
        "opener": "Overall, which is better, coffee or tea?",
        "opener_swapped": "Overall, which is better, tea or coffee?",
    },
}
TOPIC_KEY = os.getenv("SIM_TOPIC", "nuclear")
if TOPIC_KEY not in TOPICS:
    raise ValueError(f"SIM_TOPIC={TOPIC_KEY!r} is not a key of TOPICS: {sorted(TOPICS)}")
_topic = TOPICS[TOPIC_KEY]

# Per-field overrides beat the entry; only Stage 3 uses one (the opposite pole as
# SIM_TOPIC_TEXT).
TOPIC_LABEL = os.getenv("SIM_TOPIC_LABEL", _topic["label"])
TOPIC_TEXT  = os.getenv("SIM_TOPIC_TEXT", _topic["text"])

# The -1 pole, the classifier's answer B. Derived from TOPIC_TEXT so Stage 3's flip
# swaps both answers; an unregistered SIM_TOPIC_TEXT needs SIM_TOPIC_TEXT_OPPOSITE,
# since a guessed contrary could make A and B near-identical.
_POLE_PAIR = {_topic["text"]: _topic["opposite"], _topic["opposite"]: _topic["text"]}
_env_opposite = os.getenv("SIM_TOPIC_TEXT_OPPOSITE")
if _env_opposite is None and TOPIC_TEXT not in _POLE_PAIR:
    raise ValueError(
        f"SIM_TOPIC_TEXT={TOPIC_TEXT!r} is neither pole of TOPICS[{TOPIC_KEY!r}], so the "
        f"-1 pole the classifier contrasts it against cannot be derived. Pass "
        f"SIM_TOPIC_TEXT_OPPOSITE as well, or use one of: "
        f"{[_topic['text'], _topic['opposite']]}")
TOPIC_TEXT_OPPOSITE = _env_opposite or _POLE_PAIR[TOPIC_TEXT]

if TOPIC_TEXT_OPPOSITE == TOPIC_TEXT:
    raise ValueError(f"TOPIC_TEXT and TOPIC_TEXT_OPPOSITE are the same sentence "
                     f"({TOPIC_TEXT!r}): the classifier's two answers must differ.")

# The moderator's opener, drawn per exchange between the two wordings
# (agents/agent.py draw_opener) so its first-named pole is not a constant nudge.
# The classifier never sees it, and only the expresser on a first meeting reads it.
# An unregistered SIM_TOPIC_TEXT opens with itself.
_registered = TOPIC_TEXT in (_topic["text"], _topic["opposite"])
TOPIC_OPENER = _topic["opener"] if _registered else f"{TOPIC_TEXT}."
TOPIC_OPENER_SWAPPED = _topic["opener_swapped"] if _registered else TOPIC_OPENER

# Re-anchored to TOPIC_TEXT, so the logged flag means "+1 pole named first" under
# Stage 3 too.
if _registered and TOPIC_TEXT == _topic["opposite"]:
    TOPIC_OPENER_TOPIC_FIRST, TOPIC_OPENER_TOPIC_SECOND = TOPIC_OPENER_SWAPPED, TOPIC_OPENER
else:
    TOPIC_OPENER_TOPIC_FIRST, TOPIC_OPENER_TOPIC_SECOND = TOPIC_OPENER, TOPIC_OPENER_SWAPPED

if _registered and TOPIC_OPENER_SWAPPED == TOPIC_OPENER:
    raise ValueError(
        f"TOPICS[{TOPIC_KEY!r}] has identical 'opener' and 'opener_swapped': the "
        f"order draw would be a no-op and the position bias would stay constant.")

# -- Reproducibility ----------------------------------------------------------
# Seeds the global `random` stream before the persona sample and the topology, so
# it buys back initial conditions only: the gate's retries desync the stream from
# round 1 (final_schedule.md § Seeding). Drawn from OS entropy when unset and
# logged, so replicates stay independent.
_env_seed = os.getenv("SIM_SEED")
SEED = int(_env_seed) if _env_seed is not None else secrets.randbits(64)

