"""
Edge lifecycle: reward history and tie valuation.

``record_reward`` appends the expresser's reward to the edge's rolling window on
every measured-reward round, in both graph modes (it feeds select_responder).
``update_edge`` (GRAPH_DYNAMIC only) steps each valuation by the window mean ×
STRENGTH_DELTA and severs the edge at STRENGTH_FLOOR; the caller then rewires via
``matching.rewire_after_drop``. Both are skipped on a no-signal round. With |E|
conserved, the strength parameters set only how often a relocation fires, hence
the single timescale ratio (config.REWIRE_TIMESCALE_RATIO).
"""

from __future__ import annotations

from config import STRENGTH_CAP, STRENGTH_DELTA, STRENGTH_FLOOR
from network.state import NetworkState


def record_reward(
    state: NetworkState,
    agent_a: str,
    agent_b: str,
    reward_a: float | None = None,
    reward_b: float | None = None,
) -> None:
    """Append each non-None reward to its agent's window on edge (agent_a, agent_b).

    The main loops pass reward_a only (asymmetric update); reward_b is the hook
    for a symmetric extension.
    """
    edge = state.graph[agent_a][agent_b]["data"]

    if reward_a is not None:
        edge.reward_history[agent_a].append(reward_a)
    if reward_b is not None:
        edge.reward_history[agent_b].append(reward_b)


def rolling_signal(history) -> float:
    """Mean reward over a rolling window; 0.0 if empty.

    Shared with main_network, which logs the signal a severance was based on.
    """
    return sum(history) / len(history) if history else 0.0


def update_edge(
    state: NetworkState,
    agent_a: str,
    agent_b: str,
) -> str | None:
    """Step both valuations and sever the edge if either reaches the floor.

    strength += rolling_signal × STRENGTH_DELTA, clamped to [0, STRENGTH_CAP].

    Returns
    -------
    None if the edge survived, else the severing agent (the one that rewires).
    If both cross, the lower valuation initiates, ties to agent_a.
    """
    edge = state.graph[agent_a][agent_b]["data"]

    signal_a = rolling_signal(edge.reward_history[agent_a])
    signal_b = rolling_signal(edge.reward_history[agent_b])

    edge.strengths[agent_a] = max(0.0, min(STRENGTH_CAP,
        edge.strengths[agent_a] + signal_a * STRENGTH_DELTA))
    edge.strengths[agent_b] = max(0.0, min(STRENGTH_CAP,
        edge.strengths[agent_b] + signal_b * STRENGTH_DELTA))

    below_a = edge.strengths[agent_a] <= STRENGTH_FLOOR
    below_b = edge.strengths[agent_b] <= STRENGTH_FLOOR

    if below_a or below_b:
        state.graph.remove_edge(agent_a, agent_b)
        if below_a and below_b:
            return agent_a if edge.strengths[agent_a] <= edge.strengths[agent_b] else agent_b
        return agent_a if below_a else agent_b

    edge.rounds_active += 1
    return None
