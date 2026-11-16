"""
Network simulation entry point (README "Design").

Before round 1 the global stream draws the names' placement, the starting
positions (a coin start only; the default polarized start is read off the graph's communities)
and the graph, in that order. A round is EXCHANGES_PER_ROUND exchanges
(network.exchange.run_exchange), then a snapshot; after round H every agent
declares once more, logged as round H + 1. The run computes no metric. A
ParseAbort ends the run, which is replaced.
"""

import os
import random
import time

from agents.agent import Agent
from agents.llm import LLM
from agents.names import draw_names
from config import (
    CONDITION,
    EXCHANGES_PER_ROUND,
    MEMORY_ENABLED,
    NETWORK_MAX_ROUNDS,
    NUM_AGENTS,
    P_START_PLUS,
    SBM_NUM_COMMUNITIES,
    SBM_P_INTER,
    SEED,
    START,
)
from network.exchange import run_exchange
from network.logger import SimulationLogger
from network.matching import draw_pair
from network.state import NetworkState
from network.topology import build_graph


def draw_start() -> int:
    """The canon's start: pole +1 with P_START_PLUS (README "Initialization")."""
    return 1 if random.random() < P_START_PLUS else -1


def main() -> None:
    random.seed(SEED)
    print(f"Seed: {SEED}")
    logger = SimulationLogger(run_id=os.getenv("SIM_RUN_ID"))
    llm = LLM(on_parse=logger.log_parse)

    names = draw_names(NUM_AGENTS)
    starts = {name: draw_start() for name in names} if START == "coin" else {}
    graph = build_graph(names, SBM_NUM_COMMUNITIES, SBM_P_INTER)
    if START == "polarized":
        starts = {name: 1 if graph.nodes[name]["community"] == 0 else -1 for name in names}
    state = NetworkState(agents={name: Agent(name, llm) for name in names}, graph=graph,
                         positions=dict(starts), pending_start=set(names))
    logger.log_params()
    for name in names:
        logger.log_declaration(0, name, {"x": starts[name], "drawn": True})
    logger.snapshot(state)

    eligible = [n for n in names if graph.degree(n) > 0]
    neighbours = {n: list(graph.neighbors(n)) for n in eligible}
    for round_n in range(1, NETWORK_MAX_ROUNDS + 1):
        state.round = round_n
        round_start = time.time()
        for _ in range(EXCHANGES_PER_ROUND if eligible else 0):
            a, b = draw_pair(eligible, neighbours)
            record = run_exchange(
                state.agents[a], state.agents[b], round_n,
                starts=tuple(state.positions[x] if x in state.pending_start else None
                             for x in (a, b)))
            state.pending_start -= {a, b}
            for role, name in (("i", a), ("j", b)):
                state.positions[name] = record["sides"][role]["declaration"]["x"]
            logger.log_exchange(record)
        logger.snapshot(state)
        print(f"Round {round_n}/{NETWORK_MAX_ROUNDS}: {time.time() - round_start:.0f}s, "
              f"elapsed {logger.elapsed():.0f}s", flush=True)

    # The final declaration, over the whole run, in round H + 1.
    state.round = NETWORK_MAX_ROUNDS + 1
    for name, agent in state.agents.items():
        declaration = agent.declare(agent.memory.entries if MEMORY_ENABLED else [],
                                    CONDITION)
        state.positions[name] = declaration["x"]
        logger.log_declaration(state.round, name, declaration)
    logger.snapshot(state)
    print(f"Done: {NETWORK_MAX_ROUNDS} rounds in {logger.elapsed():.0f}s")


if __name__ == "__main__":
    main()
