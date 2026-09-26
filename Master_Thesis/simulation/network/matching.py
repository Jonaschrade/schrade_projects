"""
Partner selection and tie formation: one reward primitive at two ranges.

``select_responder`` (1-hop) weights neighbours by the expresser's own past
reward from them. ``expected_reward_via_referral`` (2-hop) estimates a
non-neighbour's reward by chaining it through mutual contacts; it drives
``rewire_after_drop``. No opinion state is read anywhere here: partner choice is
experience-based, so homophily is not assumed (README "Extension: endogenous tie
formation").

An isolated agent is never reconnected; it sits out until another agent's rewire
picks it, which "reward_local" can never do.
"""

from __future__ import annotations

import math
import random

import networkx as nx

from config import (
    NEW_EDGE_STRENGTH,
    REWIRE_BETA,
    REWIRE_RULE,
)
from network.state import NetworkState, new_edge_data


def select_responder(
    expresser: str,
    neighbours: list[str],
    graph: nx.Graph,
    beta: float,
) -> str:
    """Draw a responder with weight exp(β_sel · mean reward the expresser got from it).

    β_sel = 0 is uniform (Banisch & Olbrich 2019); an empty history counts as 0.0.
    Standalone so a multi-platform neighbour set (Jacob & Banisch 2023) can swap in.
    """
    if beta == 0.0:
        return random.choice(neighbours)

    def expected_reward(neighbour: str) -> float:
        history = graph[expresser][neighbour]["data"].reward_history[expresser]
        return sum(history) / len(history) if history else 0.0

    return _softmax_choice(neighbours, [expected_reward(n) for n in neighbours], beta)


def _softmax_choice(names: list[str], scores: list[float], beta: float) -> str:
    """Draw one name with probability ∝ exp(β · score); shared by both ranges."""
    if beta == 0.0:
        return random.choice(names)

    weights = [math.exp(beta * s) for s in scores]
    total = sum(weights)
    r = random.random() * total
    cumulative = 0.0
    for name, w in zip(names, weights):
        cumulative += w
        if r <= cumulative:
            return name
    return names[-1]


def _mean_history(graph: nx.Graph, holder: str, other: str) -> float | None:
    """Mean of ``holder``'s reward history on (holder, other); None when empty.

    The referral estimate drops an empty channel rather than averaging in a 0.0.
    """
    history = graph[holder][other]["data"].reward_history[holder]
    return sum(history) / len(history) if history else None


def expected_reward_via_referral(
    state: NetworkState,
    initiator: str,
    candidate: str,
) -> float:
    """Reward ``initiator`` would expect from non-neighbour ``candidate``.

        Ê[r_ij] = mean over m ∈ N(i) ∩ N(j) of  r̄_im · r̄_mj

    Since reward is a product of graded stances, r̄_im · r̄_mj ≈ g_i · g_j · g_m²:
    the intermediary only shrinks the estimate, never flips its sign. 0.0 when no
    mutual contact has history in both channels.
    """
    graph = state.graph
    estimates = []

    for contact in graph.neighbors(initiator):
        if not graph.has_edge(contact, candidate):
            continue
        to_me = _mean_history(graph, initiator, contact)
        to_them = _mean_history(graph, contact, candidate)
        if to_me is None or to_them is None:
            continue
        estimates.append(to_me * to_them)

    return sum(estimates) / len(estimates) if estimates else 0.0


def _candidate_pool(
    state: NetworkState,
    initiator: str,
    dropped_partner: str | None,
    rule: str,
) -> list[str]:
    """Non-neighbours ``initiator`` may attach to under ``rule``, sorted.

    Excludes the dropped partner, so a cut tie cannot re-form in the same step.
    "reward_local": the 2-hop set only; "random": every non-neighbour.
    """
    graph = state.graph
    excluded = set(graph.neighbors(initiator)) | {initiator}
    if dropped_partner is not None:
        excluded.add(dropped_partner)

    if rule == "reward_local":
        two_hop = {
            far for near in graph.neighbors(initiator) for far in graph.neighbors(near)
        }
        return sorted(two_hop - excluded)

    return sorted(set(state.agents) - excluded)


def rewire_after_drop(
    state: NetworkState,
    initiator: str,
    dropped_partner: str,
) -> str | None:
    """Replace a severed tie in the same step, so the cut is a relocation.

    Only the initiator rewires: it keeps its degree, so |E| and mean degree are
    conserved while the degree distribution moves. An empty "reward_local" pool
    does not fall back to a global draw, which would leak the reach the arm
    withholds.

    Returns
    -------
    The new partner, or None if the pool was empty (the degree drop stands).
    """
    pool = _candidate_pool(state, initiator, dropped_partner, REWIRE_RULE)
    if not pool:
        return None

    if REWIRE_RULE == "random":
        partner = random.choice(pool)
    else:
        scores = [expected_reward_via_referral(state, initiator, c) for c in pool]
        partner = _softmax_choice(pool, scores, REWIRE_BETA)

    state.graph.add_edge(
        initiator, partner, data=new_edge_data(initiator, partner, NEW_EDGE_STRENGTH)
    )
    return partner
