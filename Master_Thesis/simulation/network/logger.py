"""
Structured run output under ``logs/run_<id>/`` (README "Output"):

* ``params.json``: resolved parameters plus commit, model digest and Ollama
  version; the authority on what a run computed.
* ``personas.json``: the agents and their persona texts.
* ``events.jsonl``: one record per discussion, edge event and reflection; turns
  carry their gate records (read back by analysis/metrics.load_gate).
* ``network_rounds/round_NNNN.json``: graph, metrics and Q-states after each round.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import networkx as nx

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
    """Writes one run's output to ``logs/run_<run_id>/`` (run_id defaults to a timestamp)."""

    def __init__(self, run_id: str | None = None) -> None:
        ts = run_id or str(int(time.time()))
        self.run_dir = Path("logs") / f"run_{ts}"
        self.rounds_dir = self.run_dir / "network_rounds"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.start_time = time.time()

    def elapsed(self) -> float:
        """Return seconds elapsed since the logger was created."""
        return time.time() - self.start_time

    # -- internal helpers ----------------------------------------------------

    def _write(self, record: dict) -> None:
        """Append a JSON record to the events log."""
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- public logging methods ----------------------------------------------

    def log_personas(self, agents: dict) -> None:
        """Write each agent's name and persona to ``personas.json``."""
        records = [
            {"name": name, "persona": agent.persona}
            for name, agent in agents.items()
        ]
        path = self.run_dir / "personas.json"
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_params(self) -> None:
        """Write the resolved parameters and provenance to ``params.json``.

        No run reproduces bitwise, so code, weights and runtime travel with the
        parameters (final_schedule.md § Reproducibility contract).
        """
        import config as _c

        keys = [
            "NUM_AGENTS", "INTERACTIONS_PER_ROUND", "NETWORK_MAX_ROUNDS",
            "SBM_NUM_COMMUNITIES", "SBM_M_INTRA", "SBM_M_INTER",
            "SBM_PAIRS_INTRA", "SBM_PAIRS_INTER",
            "SBM_P_INTRA", "SBM_P_INTER",
            "LEARNING_RATE", "OPINION_BETA", "RESPONDER_SELECTION_BETA",
            "GRAPH_DYNAMIC", "REWIRE_RULE", "REWIRE_BETA",
            "REWIRE_TIMESCALE_RATIO", "STRENGTH_DELTA", "STRENGTH_CAP",
            "STRENGTH_FLOOR", "INITIAL_EDGE_STRENGTH", "NEW_EDGE_STRENGTH",
            "REWARD_WINDOW_M", "MEMORY_ENABLED", "REFLECT_EVERY",
            "MAX_MEMORIES_RETRIEVE", "MAX_MEMORIES_SEED",
            "SFT_GATE_ENABLED", "CONVICTION_CUE_ENABLED", "STANCE_BRIDGE_ENABLED",
            "LLM_MODEL", "LLM_NUM_CTX",
            "TOPIC_KEY", "TOPIC_LABEL", "TOPIC_TEXT", "TOPIC_TEXT_OPPOSITE",
            "TOPIC_OPENER", "TOPIC_OPENER_SWAPPED", "SEED",
        ]
        params = {k: getattr(_c, k) for k in keys if hasattr(_c, k)}
        params["run_id"] = self.run_dir.name
        params["git_commit"] = _git_commit()
        params["model_digest"] = _model_digest(_c.LLM_MODEL, _c.OLLAMA_HOST)
        params["ollama_version"] = _ollama_version(_c.OLLAMA_HOST)
        (self.run_dir / "params.json").write_text(
            json.dumps(params, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def log_discussion(
        self,
        round_n: int,
        agent_a: str,
        agent_b: str,
        result: dict,
    ) -> None:
        """Log one discussion: expresser agent_a, responder agent_b, and run_discussion's result verbatim."""
        self._write({
            "type":    "discussion",
            "round":   round_n,
            "agent_a": agent_a,
            "agent_b": agent_b,
            **result,
        })

    def log_edge_event(
        self,
        round_n: int,
        event_type: str,
        agent_a: str,
        agent_b: str,
        **kwargs,
    ) -> None:
        """Log an edge event (types: README "Output").

        For edge_dropped / edge_rewired, agent_a is the initiator. edge_dropped's
        kwargs record the decision: each side's rolling signal and post-step
        valuation, so a cut on negative feedback is distinguishable from drift.
        """
        self._write({
            "type":    event_type,
            "round":   round_n,
            "agent_a": agent_a,
            "agent_b": agent_b,
            **kwargs,
        })

    def log_reflection(self, round_n: int, agent_name: str) -> None:
        """Log a reflection trigger; the insights themselves go to stdout only."""
        self._write({
            "type":  "reflection",
            "round": round_n,
            "agent": agent_name,
        })

    def snapshot_network(
        self,
        state: NetworkState,
        extra_metrics: dict | None = None,
        opinion_states: dict | None = None,
    ) -> None:
        """Write ``network_rounds/round_NNNN.json`` for the current round.

        Edges, graph metrics merged with ``extra_metrics``, and
        ``opinion_states``; with "community" node attributes (network mode) also
        the community map and intra/inter edge counts.
        """
        edges = [
            {
                "a":             u,
                "b":             v,
                "strengths":     state.graph[u][v]["data"].strengths,
                "rounds_active": state.graph[u][v]["data"].rounds_active,
            }
            for u, v in state.graph.edges()
        ]

        degrees = dict(state.graph.degree())
        n_agents = max(1, len(state.agents))

        metrics: dict = {
            "density":      nx.density(state.graph),
            "n_components": nx.number_connected_components(state.graph),
            "avg_degree":   sum(degrees.values()) / n_agents,
            "n_edges":      state.graph.number_of_edges(),
        }

        communities = nx.get_node_attributes(state.graph, "community")
        if communities:
            n_intra = sum(
                1 for u, v in state.graph.edges()
                if communities.get(u) == communities.get(v)
            )
            metrics["n_intra_edges"] = n_intra
            metrics["n_inter_edges"] = state.graph.number_of_edges() - n_intra

        if extra_metrics:
            metrics.update(extra_metrics)

        snapshot: dict = {
            "round":      state.round,
            "elapsed_s":  round(self.elapsed(), 2),
            "nodes":      list(state.agents.keys()),
            "edges":      edges,
            "metrics":    metrics,
        }
        if communities:
            snapshot["communities"] = communities
        if opinion_states:
            snapshot["opinion_states"] = opinion_states

        path = self.rounds_dir / f"round_{state.round:04d}.json"
        path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
