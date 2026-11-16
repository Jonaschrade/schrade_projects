"""
Every prompt text, the position schema and the two presentation draws (README "Positions",
"LLM calls"). Every prompt opens with the identity line; only the declaration
shows the setting and memory.
"""

from __future__ import annotations

import random

from config import (
    SEED,
    TOPIC_ISSUE,
    TOPIC_OPENER,
    TOPIC_OPENER_SWAPPED,
    TOPIC_TEXT,
    TOPIC_TEXT_OPPOSITE,
)

# Which pole a position list starts from, drawn per call: a fixed order would
# alias the LLM's lean toward the first answer onto one pole (`topic_first`).
_SLOT_RNG = random.Random(f"classifier-slot:{SEED}")
# Which pole the moderator's opener names first, drawn per exchange. Separate
# from _SLOT_RNG, so the two orders stay independent and each estimable.
_OPENER_RNG = random.Random(f"opener-order:{SEED}")

# A persona line after Piao et al.'s (2026) opens every system prompt; it names no
# topic, so no system prompt primes the subject.
PERSONA = "Assume you are someone who participates in a discussion."
TURN_SYSTEM = f"{PERSONA} Reply only with what you say next."
ANSWER_SYSTEM = f"{PERSONA} Reply only in the requested format."

LETTERS = ("A", "B")
POSITION_SCHEMA = {"type": "object",
                   "properties": {"position": {"type": "string", "enum": list(LETTERS)}},
                   "required": ["position"]}


def wording(x: int) -> str:
    """A position's words in every prompt and list: its pole's sentence as it stands,
    no stance verb around it, never the other pole's negation."""
    return TOPIC_TEXT if x > 0 else TOPIC_TEXT_OPPOSITE


def draw_options() -> tuple[str, dict[str, int], bool]:
    """(the option list with its instruction, letter -> position, topic_first):
    the two poles, TOPIC_TEXT's first when topic_first."""
    topic_first = _SLOT_RNG.random() < 0.5
    order = (1, -1) if topic_first else (-1, 1)
    lines = "\n".join(f"{c}. {wording(x)}" for c, x in zip(LETTERS, order))
    return f"{lines}\nAnswer with the letter of one option.", dict(zip(LETTERS, order)), topic_first


def draw_opener() -> tuple[str, bool]:
    """(the moderator's opener, topic_first): topic_first when it names TOPIC_TEXT's pole first."""
    topic_first = _OPENER_RNG.random() < 0.5
    return (TOPIC_OPENER if topic_first else TOPIC_OPENER_SWAPPED), topic_first


def identity(name: str) -> str:
    return f"You are {name}."


# The declaration's setting (README "Memory"), the same in either condition; it states
# no rule, goal or reward. Only the declaration shows it.
SETTING = (f"You talk about {TOPIC_ISSUE} with other people, one conversation at a time. "
           "In each conversation you state your position and your partner states theirs.")


def declaration(name: str, window: str, options: str) -> str:
    memory = f"Memory of your past conversations, oldest first:\n\n{window}\n\n" if window else ""
    return (f"{identity(name)}\n\n{SETTING}\n\n{memory}"
            f"Which position do you take in the next conversation?\n{options}")


def opener_message(opener: str) -> str:
    """A turn's message, both sides': the opener, never the partner's turn or name."""
    return f'A moderator asks: "{opener}"'


def turn(name: str, position: int, message: str) -> str:
    return (f"{identity(name)}\n\n{message}\n\nYour position: {wording(position)}.\n\n"
            f"Reply as {name}, in two or three sentences, from this position.")


def self_read(name: str, text: str, options: str) -> str:
    return (f"{identity(name)}\n\nYou said: '{text}'\n\n"
            f"Which position did you take in this message?\n{options}")


def probe(name: str, partner: str, partner_turn: str, options: str) -> str:
    return (f"{identity(name)}\n\n{partner} said: '{partner_turn}'\n\n"
            f"Which position did {partner} take in this message?\n{options}")
