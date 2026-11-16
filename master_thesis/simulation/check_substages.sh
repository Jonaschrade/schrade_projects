#!/bin/bash
# Resolve every sub-stage of run_stage.sh and assert the table is well formed.
#
# No two sub-stages of a stage may resolve to the same parameter point: a duplicate
# means a point is paid for twice and a "contrast" is one configuration against
# itself -- the cheapest defect to introduce and the most expensive to notice late.
# Pilot phase 2 may share the runs' points: pooled with them, it adds a seed at a
# point, not a contrast. Phase 2 differs from phase 1 in SIM_START alone. The run
# totals per stage are printed.
#
# Takes the sub-stage list from run_stage.sh (LIST=1) and runs each under
# DRY_RUN=1, which prints the resolved SIM_* set and exits before the lock, the
# Ollama server and any run. Structural errors (N or H off the preamble) abort.
#
#   bash simulation/check_substages.sh          # all sub-stages
#   bash simulation/check_substages.sh P-nqf-1  # just one, printing its physics
#
# Runs locally in seconds and needs no GPU.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# run_stage.sh refuses any SIM_* passed in, so a variable left in this shell
# would abort every sub-stage.
for v in $(compgen -e | grep -E '^(SIM_|LLM_NUM_CTX$)' || true); do unset "$v"; done

SUBSTAGES="${*:-$(LIST=1 bash "$HERE/run_stage.sh")}"

resolved=""
declare -A runs
n=0
for s in $SUBSTAGES; do
  line="$(DRY_RUN=1 SUBSTAGE="$s" bash "$HERE/run_stage.sh")"
  echo "$line"
  # Compare the physics within a stage: drop the ID and NREP.
  physics="${s%%-*} $(echo "$line" | sed -E 's/^[^ ]+ NREP=[0-9]+ //')"
  resolved="$resolved$physics
"
  nrep="$(echo "$line" | sed -E 's/^[^ ]+ NREP=([0-9]+) .*/\1/')"
  runs[${s%%-*}]=$(( ${runs[${s%%-*}]:-0} + nrep ))
  n=$((n + 1))
done

distinct="$(printf '%s' "$resolved" | sort -u | grep -c . || true)"
echo
for st in $(printf '%s\n' "${!runs[@]}" | sort); do
  echo "Stage $st: ${runs[$st]} runs"
done
echo "$n sub-stages, $distinct distinct parameter points"
if [ "$n" -ne "$distinct" ]; then
  echo "FAIL: $((n - distinct)) sub-stage(s) collide. Duplicated points:"
  printf '%s' "$resolved" | sort | uniq -d
  exit 1
fi
echo "ok   every sub-stage resolves to its own parameter point"
