"""
The canon's Q-learning (Banisch & Olbrich 2019) and the rules the runs, the
benchmark, analysis/fit.py and the surrogate share. LLM-free and config-free, so
every reader imports the one definition.

Public API
----------
AgentOpinionState             the canon's Q(+1), Q(-1) (benchmark, surrogate)
init_opinion_states()         Q = 0 for every agent (benchmark)
update_q_value()              the canon's TD update (benchmark, surrogate)
argmax_pole()                 the opinion sign(ΔQ), with the tie rule
compute_polarization_metrics()  population metrics of Q states (benchmark)
explore()                     ε-greedy expression of the other pole (runs, benchmark)
perceived_reward()            r̂ = ŝ·ô, the one definition
trace()                       T <- T + a(v - T) from 0
own_pole_lags()               per entry, the later entries on its own pole
delta_q()                     ΔQ over a record, closed form (fit)
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class AgentOpinionState:
    """Q(+1) and Q(-1), both from 0.0: the canon's state, kept by the benchmark only."""

    q_pos: float = 0.0
    q_neg: float = 0.0

    @property
    def argmax_stance(self) -> int:
        """+1 or -1 by the larger Q; 0 on a tie."""
        return 1 if self.q_pos > self.q_neg else -1 if self.q_neg > self.q_pos else 0

    @property
    def q_gap(self) -> float:
        """ΔQ = Q(+1) - Q(-1), the canon's conviction."""
        return self.q_pos - self.q_neg


def init_opinion_states(agent_names: list) -> dict:
    """One AgentOpinionState per agent, both Q at 0."""
    return {name: AgentOpinionState() for name in agent_names}


def update_q_value(opinion: AgentOpinionState, stance: int, reward: float, alpha: float) -> None:
    """Q(stance) <- (1 - α) Q(stance) + α r, the expressed pole only (eq. 1)."""
    if stance == 1:
        opinion.q_pos = (1 - alpha) * opinion.q_pos + alpha * reward
    else:
        opinion.q_neg = (1 - alpha) * opinion.q_neg + alpha * reward


def argmax_pole(q_gap: float, held: int | None = None) -> int:
    """The canon's opinion, sign(ΔQ). A tie keeps ``held``, else is a coin from the
    global stream, a coin start's rule at Q = (0, 0)."""
    if q_gap == 0.0:
        return held if held is not None else random.choice((1, -1))
    return 1 if q_gap > 0 else -1


def compute_polarization_metrics(opinion_states: dict) -> dict:
    """n_pos / n_neg (strict sign of ΔQ), dispersion (variance of argmax stances, a
    tie at 0; 0 = consensus, 1 = an even division) and mean_q_gap (mean |ΔQ|)."""
    if not opinion_states:
        return {}
    stances = [s.argmax_stance for s in opinion_states.values()]
    n = len(stances)
    mean_stance = sum(stances) / n
    return {
        "n_pos": stances.count(1),
        "n_neg": stances.count(-1),
        "dispersion": round(sum((o - mean_stance) ** 2 for o in stances) / n, 4),
        "mean_q_gap": round(sum(abs(s.q_gap) for s in opinion_states.values()) / n, 4),
    }


def explore(position: int, epsilon: float) -> tuple[int, bool]:
    """(argued position, explored): with probability ε the other pole. Draws from the
    global stream, and none at ε = 0."""
    if epsilon == 0.0 or random.random() >= epsilon:
        return position, False
    return -position, True


def perceived_reward(s: int, o: int) -> int:
    """r̂ = ŝ·ô, ``s`` the self-read and ``o`` the probe, ±1: the canon's r = o_i o_j
    as the agent reads both positions (exact in the benchmark); logged, never shown."""
    return s * o


def trace(values, rate: float) -> float:
    """The canon's update form T <- T + a(v - T), run over ``values`` from 0."""
    t = 0.0
    for v in values:
        t += rate * (v - t)
    return t


def own_pole_lags(poles) -> list:
    """Per entry, the later entries on the same own pole."""
    seen = {1: 0, -1: 0}
    lags: list = []
    for p in reversed(poles):
        lags.append(seen[p])
        seen[p] += 1
    return lags[::-1]


def delta_q(poles, rewards, rate: float) -> float:
    """ΔQ over a record: each pole's Q is the trace of r̂ over its own entries,
    Σ_k s_k a(1 - a)^(m_k) r̂_k, m_k from own_pole_lags; equal to update_q_value
    run over the same entries."""
    return sum(p * rate * (1 - rate) ** m * r
               for p, m, r in zip(poles, own_pole_lags(poles), rewards))
