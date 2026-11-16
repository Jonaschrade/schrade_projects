"""
The mutable state of one run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass
class NetworkState:
    """
    Attributes
    ----------
    agents:
        Name -> Agent.
    graph:
        Undirected graph over the names, static.
    round:
        Current round, set by the main loop.
    positions:
        Name -> the latest declared position, seeded with the starts.
    pending_start:
        Agents whose start still stands as their next declaration.
    """

    agents: dict
    graph: nx.Graph
    round: int = 0
    positions: dict = field(default_factory=dict)
    pending_start: set = field(default_factory=set)
