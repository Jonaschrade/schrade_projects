#!/bin/bash
# Batch runner for one sub-stage: ONE GPU Ollama server,
# NREP runs of main_network.py against it in sequence (the model stays loaded).
#
# The sub-stage comes with the submission; only the preamble (H, R) is edited.
# The table below sets every SIM_* parameter, because config.py is read only at
# job start; where a document disagrees, this table wins. Design:
# analysis/schedule.md § Pilot, § Runs.
#
# Sub-stages:  P-<cell><cond>-<k>     pilot phase 1, p_inter = 0.125 * k, k = 0 .. 4
#              P2-<cell><cond>-<k>    pilot phase 2: phase 1 from the polarized start
#              M-<cell><cond>-<kk>    runs, p_inter = 0.5 * kk / (R - 1), kk = 00 .. R-1
#   Phase 1 alone starts by coin; phase 2 and the runs start polarized.
#   <cell> = topic n (nuclear) / r (renewables) / c (coffee_tea)
#            + model q (qwen3.5) / g (gemma4) / i (granite4.2); <cond> = f / o
#            (SIM_CONDITION full: both turns shown / own: own turn only).
#            Every sub-stage is one run.
#
# Submit:    S=P2-nqf-0; qsub -v SUBSTAGE=$S -N substage_$S ~/projects/thesis/simulation/run_stage.sh
#   Replace: move the failed run to $RUNDIR/failed/, then submit the sub-stage again
#   Partial: -v SUBSTAGE=$S,REP_OFFSET=<n>,NREP_JOB=<k> computes r=n+1..n+k
# List:      LIST=1 bash run_stage.sh   prints every sub-stage (check_substages.sh)
# Follow:    tail -f /data/scc/jonas.schrade/thesis-runs/substage_<S>.o<jobid>
#
# STAGE=<value> stays a deprecated alias for SUBSTAGE, for jobs already queued.

#$ -q gpu@scc214
#$ -l h_vmem=96G,rtx_6000=1,gpu=1
#$ -l h_rt=168:00:00
#$ -N thesis_stage
#$ -o /data/scc/jonas.schrade/thesis-runs/ -j y

set -euo pipefail
RUNDIR=/data/scc/jonas.schrade/thesis-runs

# ===================== Operating points (preamble) ========================
# Empty = not yet set. A job that needs one aborts instead of computing a
# placeholder: a placeholder run produces a complete, plausible log, and nothing in
# params.json says afterwards that the value was only a candidate.
H=100  # the horizon, fixed by design for the pilot and the runs (analysis/schedule.md § Runs)
R=""   # runs per group, set after recovery (analysis/schedule.md § Runs)

TOPIC_CODES="n:nuclear r:renewables c:coffee_tea"
MODEL_CODES="q:qwen3.5-nodraft:4b g:gemma4-nodraft:e4b i:granite4.2-nodraft:3b"  # RUNBOOK § Pulling models
COND_CODES="f:full o:own"
PILOT_P_INTER="0 0.125 0.25 0.375 0.5"

lookup() {  # lookup <code> <code:value list>
  local kv
  for kv in $2; do [ "${kv%%:*}" = "$1" ] && { echo "${kv#*:}"; return 0; }; done
  return 1
}

if [ -n "${LIST:-}" ]; then
  for t in $TOPIC_CODES; do for m in $MODEL_CODES; do
    for c in $COND_CODES; do
      for p in P P2; do
        for k in 0 1 2 3 4; do echo "$p-${t%%:*}${m%%:*}${c%%:*}-$k"; done
      done
    done
  done; done
  if [ -n "$R" ]; then
    for t in $TOPIC_CODES; do for m in $MODEL_CODES; do for c in $COND_CODES; do
      for k in $(seq 0 $((R - 1))); do
        printf 'M-%s%s%s-%02d\n' "${t%%:*}" "${m%%:*}" "${c%%:*}" "$k"
      done
    done; done; done
  fi
  exit 0
fi

if [ -n "${SUBSTAGE:-}" ]; then
  :
elif [ -n "${STAGE:-}" ]; then
  echo "NOTE: STAGE=$STAGE is deprecated; use SUBSTAGE=$STAGE."
  SUBSTAGE="$STAGE"
else
  echo "ABORT: SUBSTAGE not set."
  echo "  S=P2-nqf-0; qsub -v SUBSTAGE=\$S -N substage_\$S ~/projects/thesis/simulation/run_stage.sh"
  exit 1
fi

REP_OFFSET="${REP_OFFSET:-0}"

# Physics comes from this file only. A SIM_* value from the submission would be
# silently overwritten by the exports below or, for a name the table does not set
# (seed), silently run along. LLM_NUM_CTX likewise: too small, and Ollama cuts the
# prompt's start. Seeds come as SEED_BLOCK (guard 2).
PASSED="$(env | grep -o '^\(SIM_[A-Z_]*\|LLM_NUM_CTX\)=' | tr -d '=' | tr '\n' ' ' || true)"
if [ -n "$PASSED" ]; then
  echo "ABORT: passed at submission: $PASSED"
  echo "  The sub-stage table sets every parameter; seeds via SEED_BLOCK."
  exit 1
fi

# ========================= Sub-stage table ================================
# EVERY value is set, even where config.py has the same default: otherwise a job's
# physics depends on the checkout's state at job start.
export SIM_MEMORY_ENABLED=1
NREP=1
bad() { echo "ABORT: unknown sub-stage '$SUBSTAGE'"; exit 1; }
IFS=- read -r STG CELL K <<< "$SUBSTAGE"
SIM_TOPIC="$(lookup "${CELL:0:1}" "$TOPIC_CODES")" || bad
SIM_LLM_MODEL="$(lookup "${CELL:1:1}" "$MODEL_CODES")" || bad
case "$STG" in
  P|P2)
    [ "${#CELL}" = 3 ] || bad
    SIM_CONDITION="$(lookup "${CELL:2:1}" "$COND_CODES")" || bad
    case "$K" in 0|1|2|3|4) ;; *) bad ;; esac
    if [ "$STG" = P ]; then SIM_START=coin; else SIM_START=polarized; fi
    set -- $PILOT_P_INTER; shift "$K"; P="$1" ;;
  M)
    [ "${#CELL}" = 3 ] || bad
    SIM_CONDITION="$(lookup "${CELL:2:1}" "$COND_CODES")" || bad
    [ -n "$R" ] || { echo "ABORT: $SUBSTAGE needs R; it is not yet set."; exit 1; }
    case "$K" in [0-9][0-9]) ;; *) bad ;; esac
    [ $((10#$K)) -lt "$R" ] || bad
    SIM_START=polarized
    P="$(awk -v k="$((10#$K))" -v r="$R" 'BEGIN { printf "%.6g", 0.5 * k / (r - 1) }')" ;;
  *) bad ;;
esac
export SIM_TOPIC SIM_LLM_MODEL SIM_CONDITION SIM_START SIM_SBM_P_INTER="$P"

# =============================== Guards ===================================
# Each of these errors would otherwise show only when the comparison is made
# months later.

# 1. N and H have no override (CLAUDE.md); N = 20, H from the preamble. A hand
#    edit in config.py would otherwise show only in the analysis. Via sed, NOT
#    python: before the env setup `python` is system Python 2.x and does not
#    import config.py (RUNBOOK § Troubleshooting).
if [ -n "${DRY_RUN:-}" ]; then
  SIMDIR="$(cd "$(dirname "$0")" && pwd)"
else
  SIMDIR=~/projects/thesis/simulation   # under SGE, $0 is a spool copy
fi
for kv in NUM_AGENTS=20 NETWORK_MAX_ROUNDS=$H; do
  have="$(sed -n "s/^${kv%%=*}[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p" "$SIMDIR/config.py")"
  if [ "$have" != "${kv#*=}" ]; then
    echo "ABORT: config.py has ${kv%%=*}='$have', the schedule needs ${kv#*=}."
    exit 1
  fi
done

# 2. One seed per replicate, never one per job: SIM_SEED for every run of a
#    sub-stage would condition it on ONE graph (CLAUDE.md, guard above).
#    Nothing is blocked at present; SEED_BLOCK stays the way to do it.
SEED_BLOCK="${SEED_BLOCK:-}"

# 3. Job splitting: this job computes r = REP_OFFSET+1 .. REP_OFFSET+NREP_JOB,
#    without NREP_JOB the rest up to NREP. Never beyond NREP: the table fixes the
#    replicate count.
for v in REP_OFFSET NREP_JOB; do
  eval "val=\${$v:-0}"
  case "$val" in ''|*[!0-9]*)
    echo "ABORT: $v must be a whole number (was '$val')."; exit 1 ;;
  esac
done
NREP_JOB="${NREP_JOB:-$((NREP - REP_OFFSET))}"
if [ "$NREP_JOB" -lt 1 ] || [ $((REP_OFFSET + NREP_JOB)) -gt "$NREP" ]; then
  echo "ABORT: r$((REP_OFFSET + 1))..r$((REP_OFFSET + NREP_JOB)) is not within r1..r$NREP of $SUBSTAGE."
  exit 1
fi

# DRY_RUN: one machine-readable line with the resolved parameter point, then
# stop — before lock, Ollama and any run. The basis of check_substages.sh.
if [ -n "${DRY_RUN:-}" ]; then
  echo "$SUBSTAGE NREP=$NREP" \
       "$(env | grep '^SIM_' | sort | tr '\n' ' ')"
  exit 0
fi

echo "Sub-stage $SUBSTAGE  r$((REP_OFFSET + 1))..r$((REP_OFFSET + NREP_JOB)) of $NREP" \
     "p_inter=$SIM_SBM_P_INTER  model=$SIM_LLM_MODEL  topic=$SIM_TOPIC" \
     "condition=$SIM_CONDITION  start=$SIM_START  H=$H  seeds=${SEED_BLOCK:-<OS entropy>}"

# Guard against double runs: if several jobs of the same sub-stage wait on
# different nodes, they would otherwise write into the same
# logs/run_<SUBSTAGE>_r*/ at once. mkdir is atomic (over NFS too).
# REP_OFFSET belongs in the lock name: partial jobs of one sub-stage write
# disjoint replicates and may run in parallel.
SUBSTAGE_LOCK="$RUNDIR/.substage_${SUBSTAGE}_o${REP_OFFSET}.lock"
mkdir "$SUBSTAGE_LOCK" 2>/dev/null || {
  echo "ABORT: sub-stage $SUBSTAGE (offset $REP_OFFSET) is already running. Lock: $SUBSTAGE_LOCK"
  echo "If orphaned (job hard-killed): rmdir '$SUBSTAGE_LOCK'"
  exit 1
}
# Cleans up lock and Ollama daemon — on qdel too (SIGTERM → exit → EXIT trap).
trap 'kill "${OLLAMA_PID:-}" 2>/dev/null; rmdir "$SUBSTAGE_LOCK" 2>/dev/null' EXIT
trap 'exit 143' TERM INT

# 4. No replicate twice: the logger appends to an existing events.jsonl and
#    would so mix two runs under one name. A failed replicate moves to
#    $RUNDIR/failed/ before its replacement (RUNBOOK § Replacing a failed replicate).
for r in $(seq $((REP_OFFSET + 1)) $((REP_OFFSET + NREP_JOB))); do
  if [ -e "$RUNDIR/logs/run_${SUBSTAGE}_r$r" ]; then
    echo "ABORT: $RUNDIR/logs/run_${SUBSTAGE}_r$r already exists."; exit 1
  fi
done

# setup_thesis_env.sh appends to partly unset variables (LD_LIBRARY_PATH)
# and therefore dies under 'set -u'. Disable it for the sourcing only.
set +u
source ~/setup_thesis_env.sh
set -u
which ollama | grep -q '/data/scc' || { echo "ABORT: wrong ollama binary"; exit 1; }

# Adopt the starter_method's GPU allocation: it reserves the card (lock under
# /tmp/sge-gpu-lock-*) and puts its index in SGE_GPU. If it finds no free card,
# the job dies before this line with "Could only reserve 0 of 1 requested
# devices" (see RUNBOOK § Background: GPU allocation on the gpu queue).
if [ -n "${SGE_GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SGE_GPU// /,}"
else
  # Fallback without starter_method: pick a free GPU, skipping index 0 (discovery trap)
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi \
    --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1!=0 && $2<1000 {print $1; exit}')
  [ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "ABORT: no free GPU (other than 0)"; exit 1; }
fi
echo "node $(hostname), GPU $CUDA_VISIBLE_DEVICES (SGE_GPU='${SGE_GPU:-unset}')"

# Own port per job: several own jobs regularly share a node.
export OLLAMA_HOST=127.0.0.1:$((11500 + JOB_ID % 400))
# Otherwise one llama-server thread per node core on the job's slots (RUNBOOK § Troubleshooting).
export LLAMA_ARG_THREADS=${NSLOTS:-1}
# A model load can outlast Ollama's 5-min default (RUNBOOK § Troubleshooting).
export OLLAMA_LOAD_TIMEOUT=30m
GGML_CUDA_NO_VMM=1 ollama serve > "$HOME/ollama_$JOB_ID.log" 2>&1 &
OLLAMA_PID=$!

# HARD CPU-fallback check: wait for the discovery line, do not sleep blindly.
for _ in $(seq 1 60); do
  grep -q "inference compute" "$HOME/ollama_$JOB_ID.log" && break
  sleep 2
done
# Positive test: missing evidence is no licence. Ollama writes
# 'library=CUDA' (GPU) or 'library=cpu' (discovery trap) — hence -i.
if ! grep -qi "library=cuda" "$HOME/ollama_$JOB_ID.log"; then
  echo "ABORT: Ollama not on CUDA (discovery trap? the node's GPU 0 taken?)."
  echo "  See ollama_$JOB_ID.log. Move to a node with a free GPU 0."
  kill "$OLLAMA_PID" 2>/dev/null || true; exit 1
fi
echo "Ollama on GPU, ready."

cd "$RUNDIR"
for r in $(seq $((REP_OFFSET + 1)) $((REP_OFFSET + NREP_JOB))); do
  if [ -n "$SEED_BLOCK" ]; then
    set -- $SEED_BLOCK
    if [ "$r" -gt "$#" ]; then
      echo "ABORT: SEED_BLOCK has only $# seeds, replicate r$r needs one."; exit 1
    fi
    eval "export SIM_SEED=\${$r}"
    echo "=== Run $r (SIM_RUN_ID=${SUBSTAGE}_r${r}, seed=$SIM_SEED) ==="
  else
    echo "=== Run $r (SIM_RUN_ID=${SUBSTAGE}_r${r}) ==="
  fi
  if SIM_RUN_ID=${SUBSTAGE}_r${r} python -u ~/projects/thesis/simulation/main_network.py; then
    echo "Run $r OK"
  else
    echo "Run $r FAILED (exit $?), continuing with the next"
  fi
done

echo "Sub-stage $SUBSTAGE done."
