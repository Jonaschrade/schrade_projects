#!/bin/bash
# Resolve every sub-stage of run_stage.sh and assert the table is well formed.
#
# The check analysis/final_schedule.md mandates after any edit to that table: no
# two sub-stages may resolve to the same parameter point. A duplicate means an
# arm is paid for twice and a "contrast" is one configuration against itself --
# the cheapest defect to introduce and the most expensive to notice late.
#
# Runs run_stage.sh under DRY_RUN=1, which prints the resolved SIM_* set and
# exits before the lock, the Ollama server and any run. Placeholder values are
# allowed here and reported as OWED=<screen>; structural errors (an anchor that
# is not a rung, a non-ascending beta grid, an unscreened w) still abort.
#
#   bash simulation/check_substages.sh          # all sub-stages
#   bash simulation/check_substages.sh 1-02     # just one, printing its physics
#
# Runs locally in a second and needs no GPU.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# Stage 0 is a screen, not an arm: 0-01 is the horizon probe and shares L3 with
# 1-03 by design, differing only in NETWORK_MAX_ROUNDS, which has no SIM_*
# override and so cannot appear in a resolved parameter line. Including it would
# report a false duplicate. Resolve it explicitly with `check_substages.sh 0-01`.
ALL="1-00 1-01 1-02 1-03 1-04 1-05 1-06
     2-01 2-02 2-03
     3-01 3-02 3-03
     4-01 4-02 4-03
     5-01 5-02 5-03 5-04 5-05 5-06
     6-01 6-02 6-03 6-04 6-05
     7-01 7-02 7-03
     8-01 8-02 8-03 8-04 8-05 8-06"
SUBSTAGES="${*:-$ALL}"

# Stage 3 takes its proposition at submission time, so only it gets throwaway
# text; every other sub-stage must resolve with no topic override at all (Stage 5
# carries its topic in the table), or run_stage.sh's guard aborts it.

resolved=""
n=0
for s in $SUBSTAGES; do
  if [[ $s == 3-* ]]; then
    line="$(SIM_TOPIC_TEXT="<check>" DRY_RUN=1 SUBSTAGE="$s" bash "$HERE/run_stage.sh")"
  else
    line="$(env -u SIM_TOPIC_TEXT DRY_RUN=1 SUBSTAGE="$s" bash "$HERE/run_stage.sh")"
  fi
  echo "$line"
  # Compare the physics only: drop the ID, NREP and the OWED marker, since two
  # arms may legitimately differ in replication count but never in parameters.
  physics="$(echo "$line" | sed -E 's/^[0-9]+-[0-9]+ NREP=[0-9]+ //; s/OWED=[^ ]* //')"
  resolved="$resolved$physics
"
  n=$((n + 1))
done

distinct="$(printf '%s' "$resolved" | sort -u | grep -c . || true)"
echo
echo "$n sub-stages, $distinct distinct parameter points"
if [ "$n" -ne "$distinct" ]; then
  echo "FAIL: $((n - distinct)) sub-stage(s) collide. Duplicated points:"
  printf '%s' "$resolved" | sort | uniq -d
  exit 1
fi
echo "ok   every sub-stage resolves to its own parameter point"
