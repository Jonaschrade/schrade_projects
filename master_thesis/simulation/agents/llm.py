"""
The direct Ollama client every run, the audit and the check job call through
(README "LLM calls").

A free-text call (a turn) samples under the Modelfile's sampler. A schema
call (declaration, probe, self-read) decodes under its JSON schema at
min_p = 0, so no answer is cut before the grammar, and returns the answer's
logprobs. Every call samples under the next seed of _SAMPLE_RNG and returns it
with Ollama's prompt_eval_count. Responses are read by subscript, so a dict stub
(the audit's --stub) serves as well as ollama's response models.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable

import ollama

from config import LLM_MODEL, LLM_NUM_CTX, OLLAMA_HOST, PARSE_MAX_RETRIES, SEED

# Each call's sampling seed, drawn anew: one fixed seed would hand every re-ask of
# a prompt the same text. Its own stream, so per-call draws cannot shift the global
# one (names, starts, graph, pairing, exploration).
_SAMPLE_RNG = random.Random(f"ollama-sample:{SEED}")

# Ollama's maximum; both letters must be among them (RUNBOOK "Pulling
# models" checks it per tag).
TOP_LOGPROBS = 20


class ParseAbort(RuntimeError):
    """A schema reply failed to parse PARSE_MAX_RETRIES + 1 times; the run aborts."""


class LLM:
    """One model behind one client. ``on_parse(kind, record)`` receives every
    ``parse_retry`` and the ``parse_abort``."""

    def __init__(self, model: str = LLM_MODEL, client=None,
                 on_parse: Callable[[str, dict], None] | None = None) -> None:
        self.model = model
        self.client = client if client is not None else ollama.Client(host=OLLAMA_HOST)
        self.on_parse = on_parse

    def _generate(self, prompt: str, system: str, schema: dict | None = None):
        seed = _SAMPLE_RNG.getrandbits(31)
        options = {"seed": seed, "num_ctx": LLM_NUM_CTX}
        extra: dict = {}
        if schema is not None:
            options["min_p"] = 0.0
            extra = {"format": schema, "logprobs": True, "top_logprobs": TOP_LOGPROBS}
        response = self.client.generate(model=self.model, prompt=prompt, system=system,
                                        think=False, stream=False, options=options, **extra)
        return response, seed

    def text(self, prompt: str, system: str) -> dict:
        """{text, seed, prompt_eval_count}."""
        response, seed = self._generate(prompt, system)
        return {"text": response["response"].strip(), "seed": seed,
                "prompt_eval_count": response["prompt_eval_count"]}

    def choose(self, prompt: str, system: str, schema: dict, field: str) -> dict:
        """{answer, logprobs (label -> logprob, labels outside the top absent), seed,
        prompt_eval_count, retries}. An unparseable reply is asked again under the
        next seed, PARSE_MAX_RETRIES times, then ParseAbort."""
        labels = schema["properties"][field]["enum"]
        for attempt in range(PARSE_MAX_RETRIES + 1):
            response, seed = self._generate(prompt, system, schema)
            answer = _parse(response["response"], field, labels)
            if answer is not None:
                return {"answer": answer,
                        "logprobs": _answer_logprobs(response["logprobs"] or [], answer, labels),
                        "seed": seed, "prompt_eval_count": response["prompt_eval_count"],
                        "retries": attempt}
            record = {"model": self.model, "field": field, "attempt": attempt, "seed": seed,
                      "reply": response["response"][:500]}
            if attempt < PARSE_MAX_RETRIES and self.on_parse:
                self.on_parse("parse_retry", record)
        if self.on_parse:
            self.on_parse("parse_abort", record)
        raise ParseAbort(f"{self.model}: no parseable '{field}' in {PARSE_MAX_RETRIES + 1} tries")


def _parse(reply: str, field: str, labels: list[str]) -> str | None:
    try:
        value = json.loads(reply).get(field)
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if value in labels else None


def _label(token: str, labels: list[str]) -> str | None:
    t = token.strip().strip('"').strip()
    return t if t in labels else None


def _answer_logprobs(tokens: list, answer: str, labels: list[str]) -> dict:
    """Logprob per label at the answer's token: the first token after the field key
    that decodes to the answer; alternatives decoding to one label are summed."""
    text = ""
    for tok in tokens:
        if ":" in text and _label(tok["token"], labels) == answer:
            mass: dict[str, list[float]] = {answer: [tok["logprob"]]}
            for alt in tok["top_logprobs"] or []:
                lab = _label(alt["token"], labels)
                if lab is not None and alt["token"] != tok["token"]:
                    mass.setdefault(lab, []).append(alt["logprob"])
            return {lab: _logsumexp(v) for lab, v in mass.items()}
        text += tok["token"]
    return {}


def _logsumexp(values: list[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))
