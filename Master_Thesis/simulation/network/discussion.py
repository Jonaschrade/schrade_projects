"""
Discussion runner: alternating gated turns (Agent.respond), then the reward.

Each reward is the product of two graded stances already in the gate records,
r = graded(expresser) * graded(reaction), so no extra LLM call is made. The
responder is gated too, so the reaction comes from a stance-coherent partner.
See README "Two reward signals".
"""

from __future__ import annotations

from agents.agent import Agent, reward_from_expressions
from config import TOPIC_OPENER_TOPIC_FIRST


def gate_summary(attempts: int, fallback: bool) -> str:
    """One-line console summary of a turn's gate outcome, shared by both entry points."""
    return f"gate: {attempts} attempt(s){', fallback' if fallback else ''}"


def run_discussion(
    agent_a: Agent,
    agent_b: Agent,
    topic: str,
    topic_label: str = "",
    turns_per_agent: int = 1,
    verbose: bool = True,
    *,
    softmax_a: int,
    softmax_b: int,
    cue_a: float | None = None,
    cue_b: float | None = None,
    prior_b_message: str | None = None,
) -> dict:
    """Run one discussion and collect the rewards.

    The moderator's ``topic`` opens the transcript; with ``prior_b_message`` the
    expresser answers agent_b's last message instead (a repeat meeting). Agents
    alternate for ``turns_per_agent × 2`` gated turns. Each speaker's cue reaches
    only its own turns.

    Returns
    -------
    dict with ``turns`` ({speaker, content, gate} each), ``reward_a`` /
    ``reward_b`` (float or None; reward_b needs turns_per_agent >= 2, since with
    one turn each agent_a cannot react to agent_b), ``opener_topic_first``, the
    echoed ``softmax_*`` / ``cue_*``, and for each agent's first turn
    ``gate_attempts_*``, ``gate_fallback_*``, ``gate_final_expressed_*`` (the
    Q-update index) and ``gate_final_graded_*`` (the reward factor).
    """
    transcript: list[dict] = [{"speaker": "Moderator", "content": topic}]
    if prior_b_message is not None:
        transcript.append({"speaker": agent_b.name, "content": prior_b_message})

    total_turns = turns_per_agent * 2

    gate_a_first: dict | None = None
    gate_b_first: dict | None = None

    turns: list[dict] = []
    for i in range(total_turns):
        if i % 2 == 0:
            speaker = agent_a
            softmax_stance = softmax_a
            cue = cue_a
        else:
            speaker = agent_b
            softmax_stance = softmax_b
            cue = cue_b

        last = transcript[-1]
        reply, gate = speaker.respond(
            last["content"], last["speaker"],
            softmax_stance=softmax_stance,
            cue=cue,
        )
        if i == 0:
            gate_a_first = gate
        elif i == 1:
            gate_b_first = gate
        if verbose:
            print(f"\n{speaker.name}: {reply}")
        transcript.append({"speaker": speaker.name, "content": reply})
        turns.append({"speaker": speaker.name, "content": reply, "gate": gate})

    # reward_a: agent_a's first turn × agent_b's last reaction. reward_b mirrors it
    # and is None at one turn each, where agent_a's turn precedes agent_b's.
    graded_a_first = gate_a_first["final_expressed_graded"]
    graded_b_last = next(
        t["gate"]["final_expressed_graded"]
        for t in reversed(turns) if t["speaker"] == agent_b.name
    )
    reward_a: float | None = reward_from_expressions(graded_a_first, graded_b_last)

    def _fmt_g(g: float | None) -> str:
        return f"{g:+.2f}" if g is not None else "None"

    def _fmt_reward(r: float | None, g_expr: float | None, g_react: float | None) -> str:
        """Reward with its composition: r (= expr × react)."""
        factors = f"expr {_fmt_g(g_expr)} × resp {_fmt_g(g_react)}"
        if r is None:
            return f"None ({factors}; no signal, TD update skipped)"
        return f"{r:+.2f} (= {factors})"

    if turns_per_agent >= 2:
        graded_b_first = gate_b_first["final_expressed_graded"]
        graded_a_last = next(
            t["gate"]["final_expressed_graded"]
            for t in reversed(turns) if t["speaker"] == agent_a.name
        )
        reward_b: float | None = reward_from_expressions(graded_b_first, graded_a_last)
        print(
            f"\n  🎯 Reward  {agent_a.name}: {_fmt_reward(reward_a, graded_a_first, graded_b_last)}  |  "
            f"{agent_b.name}: {_fmt_reward(reward_b, graded_b_first, graded_a_last)}"
        )
    else:
        reward_b = None
        print(f"\n  🎯 Reward  {agent_a.name}: {_fmt_reward(reward_a, graded_a_first, graded_b_last)}")

    return {
        "topic_label":            topic_label,
        "opener_topic_first":     topic == TOPIC_OPENER_TOPIC_FIRST,
        "turns":                  turns,
        "reward_a":               reward_a,
        "reward_b":               reward_b,
        "softmax_a":              softmax_a,
        "softmax_b":              softmax_b,
        "cue_a":                  cue_a,
        "cue_b":                  cue_b,
        "gate_attempts_a":        gate_a_first["n_attempts"],
        "gate_fallback_a":        gate_a_first["fallback_used"],
        "gate_final_expressed_a": gate_a_first["final_expressed"],
        "gate_final_graded_a":    graded_a_first,
        "gate_attempts_b":        gate_b_first["n_attempts"],
        "gate_fallback_b":        gate_b_first["fallback_used"],
        "gate_final_expressed_b": gate_b_first["final_expressed"],
        "gate_final_graded_b":    gate_b_first["final_expressed_graded"],
    }
