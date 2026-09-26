"""
Network simulation entry point: SFT opinion dynamics (Banisch & Olbrich 2019) with
the asymmetric interaction rule (Jacob & Banisch 2023) on a two-block graph.

Per interaction (README "Interaction loop"):
  1. expresser drawn uniformly from agents with a neighbour;
  2. responder via select_responder (β_sel);
  3. softmax stance by softmax(β), and the conviction cue;
  4. one gated exchange (expresser speaks, responder reacts);
  5. reward r = graded(expression) × graded(reaction);
  6-8. Q-update, reward record and (GRAPH_DYNAMIC) edge update, all only on a
     measured-reward round;
  9. (GRAPH_DYNAMIC) the severing agent rewires in the same step.
Reflection every REFLECT_EVERY rounds (MEMORY_ENABLED), then a snapshot.
"""

import os
import random
import time

import networkx as nx
from langchain_ollama import OllamaLLM

from agents.agent import Agent, draw_opener
from agents.personas import sample_personas
from config import (
    CONVICTION_CUE_ENABLED,
    GRAPH_DYNAMIC,
    INITIAL_EDGE_STRENGTH,
    INTERACTIONS_PER_ROUND,
    LEARNING_RATE,
    LLM_MODEL,
    LLM_NUM_CTX,
    NETWORK_MAX_ROUNDS,
    NUM_AGENTS,
    OLLAMA_HOST,
    MEMORY_ENABLED,
    OPINION_BETA,
    REFLECT_EVERY,
    RESPONDER_SELECTION_BETA,
    REWIRE_BETA,
    REWIRE_RULE,
    REWIRE_TIMESCALE_RATIO,
    SBM_NUM_COMMUNITIES,
    SBM_M_INTER,
    SBM_M_INTRA,
    SBM_P_INTER,
    SBM_P_INTRA,
    SEED,
    STRENGTH_DELTA,
    TOPIC_LABEL,
)
from network.discussion import gate_summary, run_discussion
from network.edges import record_reward, rolling_signal, update_edge
from network.logger import SimulationLogger
from network.matching import rewire_after_drop, select_responder
from network.opinion import (
    compute_polarization_metrics,
    conviction_cue,
    draw_softmax_stance,
    init_opinion_states,
    opinion_states_to_dict,
    update_q_value,
)
from network.state import NetworkState, new_edge_data


def _distribute_sizes(n: int, k: int) -> list[int]:
    """Divide n agents as evenly as possible into k communities."""
    base, remainder = divmod(n, k)
    return [base + (1 if i < remainder else 0) for i in range(k)]


def _build_initial_graph(agent_names: list[str]) -> nx.Graph:
    """Fixed-edge-count two-block graph: SBM_M_INTRA edges per block, SBM_M_INTER across.

    Nodes carry a "community" attribute; edges start at INITIAL_EDGE_STRENGTH.
    The within-block layer is drawn first, so one seed gives the same
    within-block graph at every coupling (final_schedule.md § Seeding);
    benchmark_sft.build_graph mirrors this order.
    """
    n = len(agent_names)
    sizes = _distribute_sizes(n, SBM_NUM_COMMUNITIES)

    G = nx.Graph()
    G.add_nodes_from(agent_names)
    offset = 0
    blocks: list[list[str]] = []
    for comm_idx, size in enumerate(sizes):
        block = agent_names[offset:offset + size]
        blocks.append(block)
        for name in block:
            G.nodes[name]["community"] = comm_idx
        offset += size

    # Pass 1: exactly SBM_M_INTRA edges inside each block.
    for block in blocks:
        pairs = [(u, v) for i, u in enumerate(block) for v in block[i + 1:]]
        G.add_edges_from(random.sample(pairs, SBM_M_INTRA))

    # Pass 2: exactly SBM_M_INTER edges across the two blocks.
    cross_pairs = [(u, v) for u in blocks[0] for v in blocks[1]]
    G.add_edges_from(random.sample(cross_pairs, SBM_M_INTER))

    for u, v in G.edges():
        G[u][v]["data"] = new_edge_data(u, v, INITIAL_EDGE_STRENGTH)

    return G


def _count_edge_types(G: nx.Graph) -> tuple[int, int]:
    """Return (n_intra, n_inter) based on node 'community' attribute."""
    n_intra = sum(
        1 for u, v in G.edges()
        if G.nodes[u].get("community") == G.nodes[v].get("community")
    )
    return n_intra, G.number_of_edges() - n_intra


def main() -> None:
    """Run the full network simulation."""
    # Initial conditions only (persona records, not their texts; topology); config.py "Reproducibility".
    random.seed(SEED)
    print(f"Seed: {SEED}")

    llm = OllamaLLM(model=LLM_MODEL, base_url=f"http://{OLLAMA_HOST}", reasoning=False, num_ctx=LLM_NUM_CTX)
    logger = SimulationLogger(run_id=os.getenv("SIM_RUN_ID"))

    # -- Agent initialisation ---------------------------------------------
    print(f"\nSampling {NUM_AGENTS} personas...")
    personas = sample_personas(NUM_AGENTS, llm)
    agents: dict[str, Agent] = {
        p["name"]: Agent(name=p["name"], persona=p["persona"], llm=llm)
        for p in personas
    }

    print(f"\n{'━' * 60}")
    print("Participants")
    for name, agent in agents.items():
        print(f"\n  {name}: {agent.persona}")
        # Bridges exist only with STANCE_BRIDGE_ENABLED; stdout is their only record.
        for s, label in ((1, "Pro"), (-1, "Contra")):
            if s in agent._stance_bridges:
                print(f"      {label}-Stance-Bridge: {agent._stance_bridges[s]}")
    print(f"\n{'━' * 60}\n")

    logger.log_personas(agents)

    # -- Network initialisation -------------------------------------------
    G = _build_initial_graph(list(agents.keys()))
    state = NetworkState(agents=agents, graph=G, max_rounds=NETWORK_MAX_ROUNDS)
    state.opinion_states = init_opinion_states(list(agents.keys()))

    logger.log_params()              # the run's own parameter record
    logger.snapshot_network(state)   # round-0 baseline

    community_sizes = _distribute_sizes(NUM_AGENTS, SBM_NUM_COMMUNITIES)
    print(
        f"Model: {LLM_MODEL}\n"
        f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges  "
        f"(fixed-count block model: {SBM_NUM_COMMUNITIES} communities "
        f"{community_sizes}, m_intra={SBM_M_INTRA}/block, m_inter={SBM_M_INTER} "
        f"-> densities {SBM_P_INTRA:.3f} / {SBM_P_INTER:.3f})\n"
        f"α={LEARNING_RATE}  β={OPINION_BETA}  β_sel={RESPONDER_SELECTION_BETA}  "
        f"interactions/round={INTERACTIONS_PER_ROUND}  dynamic={GRAPH_DYNAMIC}\n"
        + (
            f"rewire={REWIRE_RULE}  β_rw={REWIRE_BETA}  "
            f"w={REWIRE_TIMESCALE_RATIO} (→ STRENGTH_DELTA={STRENGTH_DELTA:.3f})\n"
            if GRAPH_DYNAMIC else ""
        )
    )

    # last message each agent sent to a specific partner, keyed by (speaker, listener)
    last_message_to: dict[tuple[str, str], str] = {}

    # -- Main simulation loop ---------------------------------------------
    for round_n in range(1, NETWORK_MAX_ROUNDS + 1):
        state.round = round_n
        round_start = time.time()

        n_pos = sum(1 for s in state.opinion_states.values() if s.q_pos > s.q_neg)
        n_neg = sum(1 for s in state.opinion_states.values() if s.q_neg > s.q_pos)
        n_intra, n_inter = _count_edge_types(state.graph)

        print(f"\n{'━' * 60}")
        print(
            f"Round {round_n} / {NETWORK_MAX_ROUNDS}  "
            f"│  topic: {TOPIC_LABEL}  "
            f"│  edges: {state.graph.number_of_edges()} (intra: {n_intra}, inter: {n_inter})  "
            f"│  opinions: +{n_pos} / −{n_neg}  "
            f"│  elapsed: {logger.elapsed():.0f}s"
        )
        print(f"{'━' * 60}")

        # -- INTERACTIONS_PER_ROUND asymmetric interactions ---------------
        softmax_stances: dict[str, int] = {}    # last softmax stance per agent this round
        for interaction_i in range(INTERACTIONS_PER_ROUND):

            # 1. Draw expresser uniformly from agents with at least one neighbour
            eligible = [n for n in agents if state.graph.degree(n) > 0]
            if not eligible:
                break
            expresser_name = random.choice(eligible)

            # 2. Draw responder weighted by expresser's expected reward
            neighbours = list(state.graph.neighbors(expresser_name))
            responder_name = select_responder(
                expresser_name, neighbours, state.graph, RESPONDER_SELECTION_BETA
            )

            # 3. Draw softmax stances via softmax(β)
            softmax_a = draw_softmax_stance(state.opinion_states[expresser_name], OPINION_BETA)
            softmax_b = draw_softmax_stance(state.opinion_states[responder_name], OPINION_BETA)

            # 3b. Conviction cue (None = uncued ablation)
            cue_a = (conviction_cue(state.opinion_states[expresser_name], softmax_a)
                     if CONVICTION_CUE_ENABLED else None)
            cue_b = (conviction_cue(state.opinion_states[responder_name], softmax_b)
                     if CONVICTION_CUE_ENABLED else None)

            print(f"\n  [{interaction_i + 1}/{INTERACTIONS_PER_ROUND}]  "
                  f"{expresser_name} → {responder_name}  "
                  f"(softmax stances: {softmax_a:+d} / {softmax_b:+d})")

            # 4. One exchange: expresser speaks, responder reacts
            result = run_discussion(
                agents[expresser_name],
                agents[responder_name],
                draw_opener()[0],   # per exchange, not fixed
                topic_label=TOPIC_LABEL,
                turns_per_agent=1,
                softmax_a=softmax_a,
                softmax_b=softmax_b,
                cue_a=cue_a,
                cue_b=cue_b,
                prior_b_message=last_message_to.get((responder_name, expresser_name)),
            )

            # record each agent's last utterance so future exchanges can continue from it
            for turn in result["turns"]:
                other = responder_name if turn["speaker"] == expresser_name else expresser_name
                last_message_to[(turn["speaker"], other)] = turn["content"]
            result["argmax_a"] = state.opinion_states[expresser_name].argmax_stance
            result["argmax_b"] = state.opinion_states[responder_name].argmax_stance
            logger.log_discussion(round_n, expresser_name, responder_name, result)

            # 5-8. Reward consequences fire only on a measured-reward round. An
            #    abstention (expressed 0) and a None reward both freeze all three;
            #    a fabricated r = 0 would falsely decay Q (README "Two reward signals").
            expressed_a = result["gate_final_expressed_a"]
            abstained = expressed_a == 0
            apply_reward = not abstained and result["reward_a"] is not None
            if apply_reward:
                update_q_value(
                    state.opinion_states[expresser_name],
                    expressed_a,
                    result["reward_a"],
                    LEARNING_RATE,
                )
            elif abstained:
                print(f"    ⚠️  {expresser_name} abstained (ambiguous expression): Q-update, reward record & edge dynamics frozen this round")
            else:
                print(f"    ⚠️  {expresser_name}: reward unparseable. Q-update, reward record & edge dynamics frozen this round")
            softmax_stances[expresser_name] = softmax_a

            q = state.opinion_states[expresser_name]
            idx_note = f"Q-update index→{expressed_a:+d}" if apply_reward else "reward consequences frozen"
            print(
                f"    {expresser_name}: argmax→{result['argmax_a']:+d}  "
                f"softmax→{softmax_a:+d}  "
                f"{gate_summary(result['gate_attempts_a'], result['gate_fallback_a'])}"
            )
            print(
                f"    {responder_name}: argmax→{result['argmax_b']:+d}  "
                f"softmax→{softmax_b:+d}  "
                f"{gate_summary(result['gate_attempts_b'], result['gate_fallback_b'])}"
            )
            print(
                f"    Updated Q-gap {expresser_name}: {q.q_gap:+.3f} ({idx_note})"
            )

            if apply_reward:
                record_reward(
                    state,
                    expresser_name,
                    responder_name,
                    reward_a=result["reward_a"],
                )

                # 8-9. Edge dynamics; a severance is a relocation (|E| conserved).
                if GRAPH_DYNAMIC and state.graph.has_edge(expresser_name, responder_name):
                    # Held before update_edge may remove the edge, so the logged
                    # decision reads the post-step valuation that crossed the floor.
                    edge_data = state.graph[expresser_name][responder_name]["data"]
                    severed_by = update_edge(state, expresser_name, responder_name)
                    if severed_by is None:
                        logger.log_edge_event(
                            round_n, "edge_maintained", expresser_name, responder_name
                        )
                    else:
                        dropped = (
                            responder_name if severed_by == expresser_name else expresser_name
                        )
                        logger.log_edge_event(
                            round_n, "edge_dropped", severed_by, dropped,
                            signal_severed=round(
                                rolling_signal(edge_data.reward_history[severed_by]), 4
                            ),
                            signal_other=round(
                                rolling_signal(edge_data.reward_history[dropped]), 4
                            ),
                            strength_severed=round(edge_data.strengths[severed_by], 4),
                            strength_other=round(edge_data.strengths[dropped], 4),
                        )
                        print(f"    ✂  {severed_by} dropped {dropped}")

                        if REWIRE_RULE != "off":
                            new_partner = rewire_after_drop(state, severed_by, dropped)
                            if new_partner is None:
                                logger.log_edge_event(
                                    round_n, "rewire_failed", severed_by, dropped,
                                    rule=REWIRE_RULE,
                                )
                                print(f"    ⚠️  {severed_by}: no candidate under "
                                      f"REWIRE_RULE={REWIRE_RULE}; degree drops")
                            else:
                                logger.log_edge_event(
                                    round_n, "edge_rewired", severed_by, new_partner,
                                    replaced=dropped, rule=REWIRE_RULE,
                                )
                                print(f"    ↻  {severed_by} → {new_partner} (replaces {dropped})")

        # -- Reflection phase ---------------------------------------------
        if MEMORY_ENABLED and round_n % REFLECT_EVERY == 0:
            print(f"\n-- Reflection phase (round {round_n}) --")
            for agent in agents.values():
                agent.reflect(
                    state.opinion_states[agent.name].argmax_stance,
                    conviction_cue(state.opinion_states[agent.name],
                                   state.opinion_states[agent.name].argmax_stance)
                    if CONVICTION_CUE_ENABLED else None,
                )
                logger.log_reflection(round_n, agent.name)

        # -- Snapshot -----------------------------------------------------
        print(f"\n  Round {round_n} completed in {time.time() - round_start:.1f}s  "
              f"│  total elapsed: {logger.elapsed():.0f}s")
        pol_metrics = compute_polarization_metrics(state.opinion_states)
        logger.snapshot_network(
            state,
            extra_metrics=pol_metrics,
            opinion_states=opinion_states_to_dict(state.opinion_states, softmax_stances),
        )

    # -- Summary ----------------------------------------------------------
    print(f"\n\n{'━' * 60}")
    print("Simulation complete")
    print(f"  Rounds run    : {NETWORK_MAX_ROUNDS}")
    print(f"  Interactions  : {NETWORK_MAX_ROUNDS * INTERACTIONS_PER_ROUND}")
    print(f"  Final edges   : {state.graph.number_of_edges()}")
    print(f"  Components    : {nx.number_connected_components(state.graph)}")

    final_pol = compute_polarization_metrics(state.opinion_states)
    print(f"  Opinions +1   : {final_pol.get('n_pos', '?')}  /  "
          f"Opinions −1: {final_pol.get('n_neg', '?')}")
    print(f"  Dispersion    : {final_pol.get('dispersion', '?')}")
    print(f"  Mean |Q-gap|  : {final_pol.get('mean_q_gap', '?')}")
    print(f"  Logs written  : {logger.run_dir}")
    print(f"  Total time    : {logger.elapsed():.1f}s")
    print(f"{'━' * 60}\n")


if __name__ == "__main__":
    main()
