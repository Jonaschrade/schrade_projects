"""
Agent: a name and a memory; its four calls (README "The exchange", "Self-read").

Only ``declare`` shows memory. Every position call returns the position x, its
logprobs keyed by x, the drawn ``topic_first``, the call's seed and
prompt_eval_count.
"""

from __future__ import annotations

from agents import prompts
from agents.llm import LLM
from memory.store import Entry, MemoryStore, render_window


class Agent:
    def __init__(self, name: str, llm: LLM) -> None:
        self.name = name
        self.llm = llm
        self.memory = MemoryStore()

    def _position(self, prompt_for) -> dict:
        """One schema call over the two poles; ``prompt_for(options)`` builds the prompt."""
        options, letters, topic_first = prompts.draw_options()
        out = self.llm.choose(prompt_for(options), prompts.ANSWER_SYSTEM,
                              prompts.POSITION_SCHEMA, "position")
        return {"x": letters[out["answer"]],
                "logprobs": {str(letters[c]): lp for c, lp in out["logprobs"].items()},
                "topic_first": topic_first, "seed": out["seed"],
                "prompt_eval_count": out["prompt_eval_count"], "retries": out["retries"]}

    def declare(self, entries: list[Entry], condition: str) -> dict:
        """The position the agent takes, partner-blind, from ``entries`` rendered under
        ``condition``; adds the shown ids."""
        window = render_window(entries, condition)
        out = self._position(lambda options: prompts.declaration(self.name, window, options))
        out["shown"] = [e.id for e in entries]
        return out

    def speak(self, position: int, message: str) -> dict:
        """A turn arguing ``position``, one draft, unchecked: {text, seed, prompt_eval_count}."""
        return self.llm.text(prompts.turn(self.name, position, message), prompts.TURN_SYSTEM)

    def self_read(self, text: str) -> dict:
        """The position the agent's own turn ``text`` takes, read blind; shown to no later call."""
        return self._position(lambda options: prompts.self_read(self.name, text, options))

    def perceive(self, partner: str, partner_turn: str) -> dict:
        """The probe: the position ``partner`` took, shown its turn."""
        return self._position(
            lambda options: prompts.probe(self.name, partner, partner_turn, options))
