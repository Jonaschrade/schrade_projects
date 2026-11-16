"""
The agents' names: the 2020 Census's ten most common male and ten most common
female first names (U.S. Census Bureau 2026). At N = 20 every run holds all
twenty; only their placement on the nodes is drawn (README "Initialization").
"""

from __future__ import annotations

import random

NAMES = (
    "Michael", "John", "James", "David", "Robert",
    "William", "Daniel", "Joseph", "Richard", "Thomas",
    "Mary", "Maria", "Jennifer", "Elizabeth", "Patricia",
    "Linda", "Susan", "Jessica", "Barbara", "Karen",
)


def draw_names(n: int) -> list[str]:
    """``n`` distinct names in a drawn order, from the global stream."""
    if n > len(NAMES):
        raise ValueError(f"{n} agents, {len(NAMES)} names")
    return random.sample(NAMES, n)
