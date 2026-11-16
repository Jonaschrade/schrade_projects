"""
Pairing, shared by main_network.py and the benchmark (Banisch & Olbrich 2019, § 3.1).
"""

from __future__ import annotations

import random


def draw_pair(eligible: list, neighbours: dict, rng=random) -> tuple:
    """(agent i, agent j): i uniform over ``eligible`` (degree > 0) from ``rng``,
    j uniform over i's neighbours from the global stream. The benchmark passes its
    per-run stream for i; the simulation draws both from the global one."""
    i = rng.choice(eligible)
    return i, random.choice(neighbours[i])
