"""
Structured run output under ``logs/run_<id>/``, or a given directory (README "What a run logs"):

* ``params.json``: resolved parameters plus commit, model digest, Ollama
  version and node; the authority on what a run computed.
* ``events.jsonl``: one record per exchange, per declaration outside an exchange
  (the starts, round 0; the final ones, round H + 1), per parse retry and
  parse abort.
* ``network_rounds/round_NNNN.json``: each agent's latest declared position after
  each round; round 0 also holds the graph.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from network.state import NetworkState

_REPO = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    """The commit the run started from, suffixed "-dirty" on uncommitted changes.

    Never raises: provenance must not be able to fail a run.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True, check=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=_REPO,
                               capture_output=True, text=True, check=True,
                               timeout=10).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _model_digest(model: str, host: str) -> str:
    """Digest of the served weights from /api/tags, or "unreachable" / "not-served"."""
    try:
        with urllib.request.urlopen(f"http://{host}/api/tags", timeout=5) as fh:
            served = json.loads(fh.read()).get("models", [])
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return "unreachable"
    # Exact tag only: a family match could record another size's digest.
    wanted = {model, f"{model}:latest"} if ":" not in model else {model}
    for entry in served:
        if entry.get("name", "") in wanted:
            return entry.get("digest", "unknown")
    return "not-served"


def _ollama_version(host: str) -> str:
    """The Ollama server version (the runtime the digest does not pin), or "unreachable"."""
    try:
        with urllib.request.urlopen(f"http://{host}/api/version", timeout=5) as fh:
            return json.loads(fh.read()).get("version", "unknown")
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return "unreachable"


class SimulationLogger:
    """Writes one run's output to ``logs/run_<run_id>/`` (run_id defaults to a
    timestamp); ``run_dir`` replaces that path, as the exchange audit's cells do."""

    def __init__(self, run_id: str | None = None, run_dir: Path | None = None) -> None:
        ts = run_id or str(int(time.time()))
        self.run_dir = Path(run_dir) if run_dir is not None else Path("logs") / f"run_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rounds_dir = self.run_dir / "network_rounds"
        self.events_path = self.run_dir / "events.jsonl"
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def write(self, record: dict) -> None:
        """Append one JSON record to ``events.jsonl``."""
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def params(self) -> dict:
        """The resolved parameters and provenance: no run reproduces bitwise, so code,
        weights, runtime and node (its GPU's arithmetic) travel with them."""
        import config as _c

        keys = [
            "NUM_AGENTS", "EXCHANGES_PER_ROUND", "NETWORK_MAX_ROUNDS", "EXPLORATION_RATE",
            "START", "P_START_PLUS", "SBM_NUM_COMMUNITIES", "SBM_PAIRS_INTRA", "SBM_PAIRS_INTER",
            "SBM_P_INTRA", "SBM_P_INTER", "TOPIC_KEY", "TOPIC_LABEL", "TOPIC_ISSUE",
            "TOPIC_TEXT", "TOPIC_TEXT_OPPOSITE", "TOPIC_OPENER", "TOPIC_OPENER_SWAPPED",
            "LLM_MODEL", "LLM_NUM_CTX", "PARSE_MAX_RETRIES", "MEMORY_ENABLED",
            "CONDITION", "SEED",
        ]
        params = {k: getattr(_c, k) for k in keys}
        params["run_id"] = self.run_dir.name
        params["git_commit"] = _git_commit()
        params["model_digest"] = _model_digest(_c.LLM_MODEL, _c.OLLAMA_HOST)
        params["ollama_version"] = _ollama_version(_c.OLLAMA_HOST)
        params["node"] = socket.gethostname()
        return params

    def log_params(self, extra: dict | None = None) -> None:
        """Write ``params()``, plus ``extra``, to ``params.json``."""
        (self.run_dir / "params.json").write_text(
            json.dumps({**self.params(), **(extra or {})}, ensure_ascii=False, indent=2,
                       default=str),
            encoding="utf-8",
        )

    def log_exchange(self, record: dict) -> None:
        """One exchange: network.exchange.run_exchange's record verbatim."""
        self.write({"type": "exchange", **record})

    def log_declaration(self, round_n: int, agent: str, declaration: dict) -> None:
        """A declaration outside an exchange: a start or a final declaration."""
        self.write({"type": "declaration", "round": round_n, "agent": agent, **declaration})

    def log_parse(self, kind: str, record: dict) -> None:
        """``parse_retry`` or ``parse_abort``: agents.llm.LLM's ``on_parse``."""
        self.write({"type": kind, **record})

    def snapshot(self, state: NetworkState) -> None:
        """``network_rounds/round_NNNN.json``: the latest declared positions; round 0
        adds the graph (each node's community, the edges)."""
        snap: dict = {"round": state.round, "elapsed_s": round(self.elapsed(), 2),
                      "positions": state.positions}
        if state.round == 0:
            snap["communities"] = dict(state.graph.nodes(data="community"))
            snap["edges"] = [list(e) for e in state.graph.edges()]
        self.rounds_dir.mkdir(exist_ok=True)
        (self.rounds_dir / f"round_{state.round:04d}.json").write_text(
            json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
