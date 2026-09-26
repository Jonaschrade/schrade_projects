"""
Data classes for the network simulation state: NetworkState (agents, graph,
opinion states, round) and the per-edge EdgeData.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import networkx as nx

from config import REWARD_WINDOW_M


@dataclass
class EdgeData:
    """Metadata stored on every edge under key "data".

    Attributes
    ----------
    strengths:
        Per-agent valuation of the tie; stepped by update_edge, severed at
        STRENGTH_FLOOR (GRAPH_DYNAMIC only).
    reward_history:
        Per-agent rolling reward window (deque, maxlen REWARD_WINDOW_M). Only the
        expresser's side is filled; an empty window reads as 0.0.
    rounds_active:
        Discussions the edge has survived under update_edge.
    """

    strengths: dict = field(default_factory=dict)       # {agent_name: float}
    reward_history: dict = field(default_factory=dict)  # {agent_name: deque[float]}
    rounds_active: int = 0


def new_edge_data(agent_a: str, agent_b: str, strength: float) -> EdgeData:
    """The one constructor for every edge: both endpoints at ``strength``, empty histories.

    Used for founding edges (INITIAL_EDGE_STRENGTH) and rewired ties
    (NEW_EDGE_STRENGTH). An empty history is a neutral signal, so a new tie
    cannot drift toward the floor before it is used.
    """
    return EdgeData(
        strengths={agent_a: strength, agent_b: strength},
        reward_history={
            agent_a: deque(maxlen=REWARD_WINDOW_M),
            agent_b: deque(maxlen=REWARD_WINDOW_M),
        },
    )


@dataclass
class NetworkState:
    """Complete mutable state of one run.

    Attributes
    ----------
    agents:
        Agent name -> Agent; read-only after initialisation.
    graph:
        Undirected graph over agent names; edges carry EdgeData under "data".
    round:
        Current round, set by the main loop.
    max_rounds:
        The run's horizon, for the record.
    opinion_states:
        Agent name -> AgentOpinionState.
    """

    agents: dict
    graph: nx.Graph
    round: int = 0
    max_rounds: int = 30
    opinion_states: dict = field(default_factory=dict)
