"""
Pairwise entry point: network mode's mechanism for two agents on one edge
(README "Pairwise mode"). Responder selection degenerates to the other agent; a
severance under GRAPH_DYNAMIC ends the run, since there is nobody to rewire to.
"""

import random

import networkx as nx
from langchain_ollama import OllamaLLM

from agents.agent import Agent, draw_opener
from agents.personas import sample_personas
from config import (
    CONVICTION_CUE_ENABLED,
    GRAPH_DYNAMIC,
    LEARNING_RATE,
    LLM_MODEL,
    LLM_NUM_CTX,
    MEMORY_ENABLED,
    NETWORK_MAX_ROUNDS,
    OLLAMA_HOST,
    OPINION_BETA,
    REFLECT_EVERY,
    RESPONDER_SELECTION_BETA,
    INITIAL_EDGE_STRENGTH,
    TOPIC_LABEL,
)
from network.discussion import gate_summary, run_discussion
from network.edges import record_reward, update_edge
from network.logger import SimulationLogger
from network.matching import select_responder
from network.opinion import (
    compute_polarization_metrics,
    conviction_cue,
    draw_softmax_stance,
    init_opinion_states,
    update_q_value,
)
from network.state import NetworkState, new_edge_data

# One expected interaction per agent per round; config's value is network-mode only.
INTERACTIONS_PER_ROUND = 2


def main() -> None:
    """Run the two-agent pairwise simulation with full SFT Q-value tracking."""
    llm = OllamaLLM(model=LLM_MODEL, base_url=f"http://{OLLAMA_HOST}", reasoning=False, num_ctx=LLM_NUM_CTX)
    logger = SimulationLogger()

    # -- Agent initialisation ---------------------------------------------
    print("\nSampling 2 personas...")
    personas = sample_personas(2, llm)
    agents: dict[str, Agent] = {
        p["name"]: Agent(name=p["name"], persona=p["persona"], llm=llm)
        for p in personas
    }
    a_name, b_name = list(agents.keys())

    print(f"\n{'━' * 50}")
    print("Participants")
    for name, agent in agents.items():
        print(f"\n\n  {name}: {agent.persona}")
        # Bridges exist only with STANCE_BRIDGE_ENABLED; stdout is their only record.
        for s, label in ((1, "Pro"), (-1, "Contra")):
            if s in agent._stance_bridges:
                print(f"\n     {label}-Stance-Bridge: {agent._stance_bridges[s]}")
    print(f"\n{'━' * 50}\n")

    logger.log_personas(agents)

    # -- SFT opinion state initialisation --------------------------------
    opinion_states = init_opinion_states([a_name, b_name])

    # -- Single-edge graph (identical EdgeData to network mode) -----------
    G = nx.Graph()
    G.add_nodes_from([a_name, b_name])
    G.add_edge(a_name, b_name, data=new_edge_data(a_name, b_name, INITIAL_EDGE_STRENGTH))
    state = NetworkState(agents=agents, graph=G, max_rounds=NETWORK_MAX_ROUNDS)
    state.opinion_states = opinion_states

    # last message each agent sent to a specific partner, keyed by (speaker, listener)
    last_message_to: dict[tuple[str, str], str] = {}

    print(f"β={OPINION_BETA}  α={LEARNING_RATE}  β_sel={RESPONDER_SELECTION_BETA}  dynamic={GRAPH_DYNAMIC}\n")

    # -- Main simulation loop ---------------------------------------------
    terminated_early = False
    for round_n in range(1, NETWORK_MAX_ROUNDS + 1):
        state.round = round_n

        n_pos = sum(1 for s in opinion_states.values() if s.q_pos > s.q_neg)
        n_neg = sum(1 for s in opinion_states.values() if s.q_neg > s.q_pos)
        edge_data = G[a_name][b_name]["data"]
        strength_str = (
            f"  │  strengths: {a_name} {edge_data.strengths[a_name]:.2f} "
            f"/ {b_name} {edge_data.strengths[b_name]:.2f}"
            if GRAPH_DYNAMIC else ""
        )

        print(f"\n{'━' * 50}")
        print(
            f"Round {round_n} / {NETWORK_MAX_ROUNDS}  "
            f"│  topic: {TOPIC_LABEL}  "
            f"│  opinions: +{n_pos} / −{n_neg}"
            f"{strength_str}"
        )
        print(f"{'━' * 50}")

        # INTERACTIONS_PER_ROUND asymmetric interactions; expresser drawn uniformly each time
        for interaction_i in range(INTERACTIONS_PER_ROUND):
            expresser_name = random.choice([a_name, b_name])

            # 1. Draw responder (degenerates to the only neighbour)
            neighbours = list(state.graph.neighbors(expresser_name))
            responder_name = select_responder(
                expresser_name, neighbours, state.graph, RESPONDER_SELECTION_BETA
            )

            # 2. Softmax draw
            softmax_a = draw_softmax_stance(opinion_states[expresser_name], OPINION_BETA)
            softmax_b = draw_softmax_stance(opinion_states[responder_name], OPINION_BETA)
            cue_a = (conviction_cue(opinion_states[expresser_name], softmax_a)
                     if CONVICTION_CUE_ENABLED else None)
            cue_b = (conviction_cue(opinion_states[responder_name], softmax_b)
                     if CONVICTION_CUE_ENABLED else None)

            print(
                f"\n  [{interaction_i + 1}/{INTERACTIONS_PER_ROUND}]  "
                f"{expresser_name} → {responder_name}  "
                f"(softmax stances: {softmax_a:+d} / {softmax_b:+d})"
            )

            # 3. Exchange (both turns gated), 4. reward, 5. Q-update
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
            result["argmax_a"] = opinion_states[expresser_name].argmax_stance
            result["argmax_b"] = opinion_states[responder_name].argmax_stance
            logger.log_discussion(round_n, expresser_name, responder_name, result)

            # 5-7. Reward consequences only on a measured-reward round (see main_network).
            expressed_a = result["gate_final_expressed_a"]
            abstained = expressed_a == 0
            apply_reward = not abstained and result["reward_a"] is not None
            if apply_reward:
                update_q_value(
                    opinion_states[expresser_name],
                    expressed_a,
                    result["reward_a"],
                    LEARNING_RATE,
                )
            elif abstained:
                print(f"    ⚠️  {expresser_name} abstained (ambiguous expression): Q-update, reward record & edge dynamics frozen this round")
            else:
                print(f"    ⚠️  {expresser_name}: reward unparseable. Q-update, reward record & edge dynamics frozen this round")

            q = opinion_states[expresser_name]
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

                # Edge update (GRAPH_DYNAMIC only): may sever the edge
                if GRAPH_DYNAMIC:
                    severed_by = update_edge(state, expresser_name, responder_name)
                    print(
                        f"    Strength  {expresser_name}: {edge_data.strengths[expresser_name]:.2f}  "
                        f"│  {responder_name}: {edge_data.strengths[responder_name]:.2f}"
                    )
                    if severed_by is not None:
                        print(f"\n  ✋ Conversation ended: {severed_by} let relationship "
                              f"strength reach floor.")
                        terminated_early = True
                        break

        if terminated_early:
            break

        # -- Reflection phase ---------------------------------------------
        if MEMORY_ENABLED and round_n % REFLECT_EVERY == 0:
            print(f"\n-- Reflection phase (round {round_n}) --")
            for agent in agents.values():
                agent.reflect(
                    opinion_states[agent.name].argmax_stance,
                    conviction_cue(opinion_states[agent.name],
                                   opinion_states[agent.name].argmax_stance)
                    if CONVICTION_CUE_ENABLED else None,
                )
                logger.log_reflection(round_n, agent.name)

    # -- Summary ----------------------------------------------------------
    print(f"\n\n{'━' * 50}")
    print("Simulation complete")
    completed_rounds = state.round if terminated_early else NETWORK_MAX_ROUNDS
    print(f"  Interactions  : {completed_rounds * INTERACTIONS_PER_ROUND}"
          + (" (early termination)" if terminated_early else ""))

    final_pol = compute_polarization_metrics(opinion_states)
    print(
        f"  Opinions +1   : {final_pol.get('n_pos', '?')}  /  "
        f"Opinions −1: {final_pol.get('n_neg', '?')}"
    )
    print(f"  Dispersion    : {final_pol.get('dispersion', '?')}")
    print(f"  Mean |Q-gap|  : {final_pol.get('mean_q_gap', '?')}")
    if GRAPH_DYNAMIC:
        edge_data = G[a_name][b_name]["data"]
        print(
            f"  Final strengths: {a_name} {edge_data.strengths[a_name]:.2f} "
            f"/ {b_name} {edge_data.strengths[b_name]:.2f}"
        )
    print(f"  Logs written  : {logger.run_dir}")
    print(f"{'━' * 50}\n")


if __name__ == "__main__":
    main()
