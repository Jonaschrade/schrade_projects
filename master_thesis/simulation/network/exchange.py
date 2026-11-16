"""
One exchange (README "The exchange", whose step numbers the comments follow):
declarations, exploration, turns, self-reads and probes, rewards, stored
entries. main_network.py and the exchange audit both run it, so the audit measures
the exchange a run holds.
"""

from __future__ import annotations

from agents import prompts
from agents.agent import Agent
from config import CONDITION, EXPLORATION_RATE, MEMORY_ENABLED
from network.opinion import explore, perceived_reward


def run_exchange(
    i: Agent,
    j: Agent,
    round_: int,
    *,
    starts: tuple = (None, None),
    argued: tuple | None = None,
    epsilon: float = EXPLORATION_RATE,
    store: bool = MEMORY_ENABLED,
    condition: str = CONDITION,
) -> dict:
    """Run one exchange; returns its record.

    ``starts``: a side's start, standing as its declaration.
    ``argued``: both argued positions, assigned (the audit's strata), which skips
    declarations and exploration.
    """
    agents = {"i": i, "j": j}
    partner = {"i": j, "j": i}
    sides: dict = {"i": {"agent": i.name}, "j": {"agent": j.name}}
    record = {"round": round_, "i": i.name, "j": j.name, "sides": sides}

    # 2-3. Declarations, partner-blind, then exploration, i's draw before j's.
    for role in ("i", "j"):
        if argued is not None:
            sides[role]["s"], sides[role]["explored"] = argued[0 if role == "i" else 1], False
            continue
        start = starts[0 if role == "i" else 1]
        if start is not None:
            declaration = {"x": start, "drawn": True}
        else:
            agent = agents[role]
            declaration = agent.declare(agent.memory.entries if store else [], condition)
        sides[role]["declaration"] = declaration
    if argued is None:
        for role in ("i", "j"):
            sides[role]["s"], sides[role]["explored"] = explore(sides[role]["declaration"]["x"],
                                                                epsilon)

    # 4. Turns, i's then j's, partner-blind: both answer the one opener.
    # 5. Each self-read blind: s_hat, a measurement no prompt shows.
    record["opener"], record["opener_topic_first"] = prompts.draw_opener()
    message = prompts.opener_message(record["opener"])
    for role in ("i", "j"):
        sides[role]["turn"] = agents[role].speak(sides[role]["s"], message)
    for role in ("i", "j"):
        sides[role]["turn"]["self_read"] = agents[role].self_read(sides[role]["turn"]["text"])
    s_hat = {role: sides[role]["turn"]["self_read"]["x"] for role in ("i", "j")}

    # 5-6. Probes, each shown the partner's turn, and the reward r̂ = ŝ·ô from the
    # side's own self-read and probe.
    for role in ("i", "j"):
        other = "j" if role == "i" else "i"
        probe = agents[role].perceive(partner[role].name, sides[other]["turn"]["text"])
        sides[role]["probe"] = probe
        sides[role]["r_hat"] = perceived_reward(s_hat[role], probe["x"])
    # 7. One entry each: the two turns' text.
    if store:
        for role in ("i", "j"):
            other = "j" if role == "i" else "i"
            sides[role]["stored"] = agents[role].memory.add(
                round_, partner[role].name, sides[role]["turn"]["text"],
                sides[other]["turn"]["text"])
    return record
