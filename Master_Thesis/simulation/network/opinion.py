"""
Agent opinion state and Q-value dynamics for Social Feedback Theory (SFT).

Core reinforcement-learning mechanism from Banisch & Olbrich (2019): each agent
holds Q-values over the two stances (+1 / −1), draws a stance by softmax (inverse
temperature β), and updates that Q-value via a TD rule after each interaction.

Public API
----------
AgentOpinionState             dataclass: Q-values + argmax_stance + q_gap
init_opinion_states()         initialise one state per agent
draw_softmax_stance()         stochastic stance draw (β-parameterised)
conviction_cue()              conviction in a stance, on [0, 1], for the prompt
update_q_value()              apply TD update for a given stance
opinion_states_to_dict()      serialise for logging (argmax + optional softmax)
compute_polarization_metrics()  population-level SFT metrics

Argmax stance (metrics, logging) vs. softmax stance (the per-interaction draw that
anchors respond()): README "Terminology".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AgentOpinionState:
    """SFT internal state: Q(+1) and Q(-1), both starting at 0.0.

    Neutral start keeps model priors out of the first step: any divergence is
    caused by social feedback.
    """

    q_pos: float = 0.0
    q_neg: float = 0.0

    @property
    def argmax_stance(self) -> int:
        """Deterministic argmax over Q-values; ties resolve to +1."""
        return 1 if self.q_pos >= self.q_neg else -1

    @property
    def q_gap(self) -> float:
        """Signed confidence: positive = leaning pro, negative = leaning contra."""
        return self.q_pos - self.q_neg


def init_opinion_states(agent_names: list[str]) -> dict[str, AgentOpinionState]:
    """Initialise one neutral AgentOpinionState for each agent."""
    return {name: AgentOpinionState() for name in agent_names}


def update_q_value(
    opinion: AgentOpinionState,
    stance: int,
    reward: float,
    alpha: float,
) -> None:
    """TD update Q(stance) ← (1 − α) · Q(stance) + α · r (Banisch & Olbrich 2019).

    ``stance`` must be the stance that entered the interaction: callers pass the
    expresser's non-zero ``gate_final_expressed``. Mutates ``opinion``.
    """
    if stance == 1:
        opinion.q_pos = (1 - alpha) * opinion.q_pos + alpha * reward
    else:
        opinion.q_neg = (1 - alpha) * opinion.q_neg + alpha * reward


def conviction_cue(opinion: AgentOpinionState, stance: int) -> float:
    """How settled the agent is about ``stance``: max(0, stance * q_gap) / 2 in [0, 1].

    Carries the magnitude of Q into the prompt where the softmax draw carries only
    its sign (README "The conviction cue"). ``stance`` is required: an off-argmax
    draw argues the lower-valued pole, and the clamp sends it to band 0. Linear,
    because q_gap is bounded in [-2, 2] by the update rule itself.
    """
    return max(0.0, stance * opinion.q_gap) / 2.0


def draw_softmax_stance(opinion: AgentOpinionState, beta: float) -> int:
    """Draw +1 or -1 with p(+1) = σ(β · q_gap).

    β = 0 gives a fair coin; β = inf is special-cased as sign(q_gap), because
    the sigmoid is nan there at q_gap = 0, and an exact tie stays a fair draw
    rather than argmax_stance's tie-break to +1. A large finite β can overflow
    math.exp.
    """
    import random
    if math.isinf(beta):
        if opinion.q_gap == 0.0:
            return random.choice((1, -1))
        return 1 if opinion.q_gap > 0 else -1
    p_pos = 1.0 / (1.0 + math.exp(-beta * opinion.q_gap))
    return 1 if random.random() < p_pos else -1


def opinion_states_to_dict(
    opinion_states: dict[str, AgentOpinionState],
    softmax_stances: dict[str, int] | None = None,
) -> dict[str, dict]:
    """Serialise opinion states for the round snapshot.

    ``softmax_stances`` adds each agent's last draw as expresser this round;
    agents that never expressed get no ``"softmax"`` key.
    """
    result = {}
    for name, s in opinion_states.items():
        entry: dict = {
            "q_pos":  round(s.q_pos, 4),
            "q_neg":  round(s.q_neg, 4),
            "argmax": s.argmax_stance,
        }
        if softmax_stances is not None and name in softmax_stances:
            entry["softmax"] = softmax_stances[name]
        entry["q_gap"] = round(s.q_gap, 4)
        result[name] = entry
    return result


def compute_polarization_metrics(
    opinion_states: dict[str, AgentOpinionState],
) -> dict:
    """Population-level metrics (Banisch & Olbrich 2019).

    Returns n_pos / n_neg (strict Q-gap sign; tied agents in neither), dispersion
    (variance of argmax stances, 0 = consensus, 1 = even split) and mean_q_gap
    (mean |q_gap|).
    """
    if not opinion_states:
        return {}

    stances = [s.argmax_stance for s in opinion_states.values()]
    n = len(stances)
    n_pos = sum(1 for s in opinion_states.values() if s.q_pos > s.q_neg)
    n_neg = sum(1 for s in opinion_states.values() if s.q_neg > s.q_pos)
    mean_stance = sum(stances) / n
    dispersion = sum((o - mean_stance) ** 2 for o in stances) / n
    mean_q_gap = sum(abs(s.q_gap) for s in opinion_states.values()) / n

    return {
        "n_pos":      n_pos,
        "n_neg":      n_neg,
        "dispersion": round(dispersion, 4),
        "mean_q_gap": round(mean_q_gap, 4),
    }
