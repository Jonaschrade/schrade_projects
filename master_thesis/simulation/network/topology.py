"""
The canon's two-block graph (Banisch & Olbrich 2019, § 5.4), shared by
main_network.py and the binary benchmark so neither can drift from the other.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import networkx as nx


def distribute_sizes(n: int, k: int) -> list[int]:
    """Divide n agents as evenly as possible into k communities."""
    base, remainder = divmod(n, k)
    return [base + (1 if i < remainder else 0) for i in range(k)]


def build_graph(
    nodes: Iterable,
    k: int,
    p_inter: float,
    rng: random.Random | None = None,
) -> nx.Graph:
    """Cross pairs linked at ``p_inter``, within pairs at ``1 - p_inter``.

    Nodes carry a "community" attribute (the first half 0, the second 1). Every pair draws one uniform u, in a fixed
    order and whatever the coupling, and is linked iff (u < p_inter) == (it is a
    cross pair), so one seed's graphs nest across the coupling
    (analysis/schedule.md § Set-up). ``rng`` None draws from the global stream,
    which main_network seeds with config.SEED.
    """
    nodes = list(nodes)
    sizes = distribute_sizes(len(nodes), k)
    if k != 2 or len(set(sizes)) != 1:
        raise ValueError(f"the canon's topology needs exactly 2 equal blocks, got "
                         f"k={k} sizes={sizes}")
    uniform = rng.random if rng is not None else random.random

    G = nx.Graph()
    G.add_nodes_from(nodes)
    offset = 0
    for comm_idx, size in enumerate(sizes):
        for node in nodes[offset:offset + size]:
            G.nodes[node]["community"] = comm_idx
        offset += size

    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            cross = G.nodes[u]["community"] != G.nodes[v]["community"]
            if (uniform() < p_inter) == cross:
                G.add_edge(u, v)
    return G


def count_edge_types(G: nx.Graph) -> tuple[int, int]:
    """Return (n_intra, n_inter) from the "community" node attribute."""
    intra = sum(1 for u, v in G.edges() if G.nodes[u]["community"] == G.nodes[v]["community"])
    return intra, G.number_of_edges() - intra
