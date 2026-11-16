"""
Each agent's exchange entries and their rendering (README "Memory").

Entries are the only memory: no retrieval, no importance, no reflection. A
declaration shows every entry stored so far, in the order stored, unscored.
``render`` is the one place the conditions' table is applied; the audit renders its
exchanges and constructed windows through it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Entry:
    """One exchange as one agent stores it: both turns' text; ``id`` indexes the
    agent's store."""

    id: int
    round: int
    partner: str
    own_text: str
    partner_text: str

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryStore:
    def __init__(self) -> None:
        self.entries: list[Entry] = []

    def add(self, round_: int, partner: str, own_text: str, partner_text: str) -> int:
        """Store one entry; returns its id."""
        entry = Entry(len(self.entries), round_, partner, own_text, partner_text)
        self.entries.append(entry)
        return entry.id


def render(entry: Entry, condition: str) -> str:
    """One entry under ``condition`` ("full" both turns, "own" the own turn), the same in
    either role, since neither turn answers the other. It states no position or reward,
    marks no explored turn and stores no declaration."""
    said = f"You said: '{entry.own_text}'"
    lines = {"full": [said, f"{entry.partner} said: '{entry.partner_text}'"],
             "own": [said]}
    return "\n".join(lines[condition])


def render_window(entries: list[Entry], condition: str) -> str:
    """The window as a declaration shows it, entries separated by a blank line."""
    return "\n\n".join(render(e, condition) for e in entries)
