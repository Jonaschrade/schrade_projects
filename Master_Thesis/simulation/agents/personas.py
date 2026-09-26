"""
Persona sampling from data/gss_personas.json: GSS survey records (question text
-> answer), exported by notebooks/data_processing.ipynb with attitude items
excluded (README "Persona pool").

Public API:
    sample_personas(n, llm) -> list[{"name", "persona"}]
"""

import json
import random
import re
from pathlib import Path

_POOL_PATH = Path(__file__).parent.parent / "data" / "gss_personas.json"


def _load_pool() -> list[dict]:
    """Load persona records from disk. Each record is a plain dict."""
    with open(_POOL_PATH, encoding="utf-8") as f:
        return json.load(f)


def _clean_value(value: str) -> str:
    """Strip leading answer codes such as '(2) '."""
    return re.sub(r"^\(\d+\)\s*", "", value).strip()


def _format_record(record: dict) -> str:
    """Format a survey record dict as a human-readable attribute block."""
    return "\n".join(f"- {q}: {_clean_value(str(a))}" for q, a in record.items())


def _expand(record: dict, llm) -> dict:
    """LLM-derived first name and second-person persona for one record; generic fallback if unparseable."""
    attributes = _format_record(record)

    raw = llm.invoke(
        f"You are given the survey profile of a US resident:\n\n"
        f"{attributes}\n\n"
        f"Your task:\n"
        f"1. Invent a fitting American first name for this person.\n"
        f"2. Write a 2-3-sentence persona description in the second person "
        f"(start with 'You are ...') that reflects the person's life situation, professional "
        f"background, social environment, and communication style.\n\n"
        f"Reply exclusively in this format:\n"
        f"NAME: <first name>\n"
        f"PERSONA: <description>"
    ).strip()

    name, persona = None, None
    for line in raw.splitlines():
        if line.upper().startswith("NAME:"):
            name = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PERSONA:"):
            persona = line.split(":", 1)[1].strip()

    return {
        "name": name or "Alex",
        "persona": persona or "You are a US resident with your own opinions.",
    }


def sample_personas(n: int, llm) -> list[dict]:
    """Draw records at random and expand them until n personas with unique names exist."""
    pool = _load_pool()
    personas = []
    used_names: set[str] = set()

    remaining = random.sample(pool, len(pool))

    for record in remaining:
        if len(personas) == n:
            break
        p = _expand(record, llm)
        if p["name"].lower() in used_names:
            continue
        used_names.add(p["name"].lower())
        print(f"  Sampled persona: {p['name']}")
        personas.append(p)

    return personas
