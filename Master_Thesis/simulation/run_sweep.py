"""
Local grid sweep: one ``main_network.py`` subprocess per ``PARAM_GRID`` combination,
parameters passed as SIM_* environment variables. On the cluster, run_stage.sh
replaces it.

Each run logs to ``logs/run_sweep_<ts>_<i>/`` and writes its own params.json; the
grid goes to ``logs/sweeps/<ts>/manifest.json``. The grid is checked against
``SUPPORTED_OVERRIDES`` and aborts on any other name, because an unread override
yields a complete run at the wrong parameters.

Usage
-----
    python run_sweep.py --dry-run        # print the grid
    python run_sweep.py --workers 3      # up to 3 runs in parallel (limit: Ollama's concurrency)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# -- Grid definition ----------------------------------------------------------
# Each key maps to the SIM_* environment variable read by config.py.
# Edit these lists to change the sweep; values are cast to str() when set.
PARAM_GRID: dict[str, list] = {
    "SIM_SBM_M_INTER": [5, 10, 30],   # cross-edge counts, of 100 possible pairs
    "SIM_RESPONDER_SELECTION_BETA": [0.0, 2.0],
    "SIM_OPINION_BETA": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "SIM_GRAPH_DYNAMIC": ["0", "1"],
}

# Applied to every combination. Empty by default.
COMMON_OVERRIDES: dict[str, str] = {}

# Exactly the SIM_* names config.py reads; keep in step with it. Deliberately
# absent: NUM_AGENTS and NETWORK_MAX_ROUNDS (config.py docstring).
SUPPORTED_OVERRIDES: frozenset[str] = frozenset({
    "SIM_SBM_M_INTER", "SIM_SBM_M_INTRA", "SIM_OPINION_BETA",
    "SIM_RESPONDER_SELECTION_BETA", "SIM_GRAPH_DYNAMIC", "SIM_REWIRE_RULE",
    "SIM_REWIRE_BETA", "SIM_REWIRE_TIMESCALE_RATIO", "SIM_MEMORY_ENABLED",
    "SIM_TOPIC", "SIM_TOPIC_LABEL", "SIM_TOPIC_TEXT", "SIM_TOPIC_TEXT_OPPOSITE",
    "SIM_LLM_MODEL", "SIM_SEED", "SIM_STRENGTH_CAP", "SIM_REWARD_WINDOW_M",
    "SIM_NEW_EDGE_STRENGTH", "SIM_CONVICTION_CUE", "SIM_STANCE_BRIDGE",
})


def check_overrides() -> None:
    """Abort on any SIM_* name config.py does not read."""
    unknown = sorted((set(PARAM_GRID) | set(COMMON_OVERRIDES)) - SUPPORTED_OVERRIDES)
    if unknown:
        raise SystemExit(
            "config.py does not read: " + ", ".join(unknown)
            + "\nThese would be exported and ignored. Supported: "
            + ", ".join(sorted(SUPPORTED_OVERRIDES))
        )


def build_grid() -> list[dict[str, str]]:
    """Expand ``PARAM_GRID`` into one env-var dict per combination."""
    keys = list(PARAM_GRID.keys())
    combos = []
    for values in itertools.product(*PARAM_GRID.values()):
        combo = {key: str(value) for key, value in zip(keys, values)}
        combos.append(combo)
    return combos


def run_one(combo: dict[str, str], sweep_dir: Path, index: int) -> dict:
    """Run a single grid point in a subprocess and return its manifest entry."""
    run_id = f"sweep_{sweep_dir.name}_{index:03d}"
    env = os.environ.copy()
    env.update(COMMON_OVERRIDES)
    env.update(combo)
    env["SIM_RUN_ID"] = run_id

    start = time.time()
    result = subprocess.run(
        [sys.executable, "main_network.py"],
        env=env,
        capture_output=True,
        text=True,
    )
    duration = time.time() - start

    return {
        "run_id": run_id,
        "params": combo,
        "common_overrides": COMMON_OVERRIDES,
        "returncode": result.returncode,
        "duration_seconds": round(duration, 1),
        "log_dir": f"logs/run_{run_id}",
        "stderr_tail": result.stderr[-2000:] if result.returncode != 0 else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a parameter grid sweep over main_network.py")
    parser.add_argument("--workers", type=int, default=1, help="Number of grid points to run in parallel")
    parser.add_argument("--dry-run", action="store_true", help="Print the grid without running anything")
    args = parser.parse_args()

    check_overrides()
    combos = build_grid()
    # ASCII only: --dry-run runs on Windows consoles (cp1252).
    print(f"Grid size: {len(combos)} combinations  |  workers={args.workers}")
    for i, combo in enumerate(combos):
        print(f"  [{i:03d}] {combo}")

    if args.dry_run:
        return

    sweep_dir = Path("logs") / "sweeps" / str(int(time.time()))
    sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, combo, sweep_dir, i) for i, combo in enumerate(combos)]
        for future in futures:
            entry = future.result()
            status = "ok" if entry["returncode"] == 0 else f"FAILED (rc={entry['returncode']})"
            print(f"  {entry['run_id']}: {status}  ({entry['duration_seconds']}s)  -> {entry['log_dir']}")
            manifest.append(entry)

    manifest_path = sweep_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
