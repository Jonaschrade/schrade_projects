#!/bin/bash
# The exchange audit, one cell's sub-phase per job, both conditions in it: one GPU
# Ollama server, build_audit.py writing
# $RUNDIR/logs/audit/<topic>_<model>/<phase>_<sub>/ against it, the calibration under logs/audit/calibration/ (analysis/schedule.md § Exchange audit). The tags must be built first
# (simulation/pull_models.sh, RUNBOOK § Pulling models).
#
# Submit one phase for the nine cells, each cell's windows held until its meetings end
# (RUNBOOK § Exchange audit):
#   A=~/projects/thesis/analysis/audit/run_audit.sh; P=calibration
#   for M in qwen3.5-nodraft:4b gemma4-nodraft:e4b granite4.2-nodraft:3b; do
#     for T in nuclear renewables coffee_tea; do
#       J=audit_${P}_meetings_${T}_${M%%:*}
#       qsub -v MODEL=$M,TOPIC=$T,PHASE=$P,SUB=meetings -N $J $A
#       qsub -hold_jid $J -v MODEL=$M,TOPIC=$T,PHASE=$P,SUB=windows -N audit_${P}_windows_${T}_${M%%:*} $A
#     done
#   done
# A job cut short resumes when resubmitted with the same line at the same commit, under
# a new seed; its units run replicate-major, so the part written stays balanced. A windows
# job aborts while any of its meetings is missing or from another commit. A rerun at
# another commit moves the logs aside first (RUNBOOK § Exchange audit).
# Copy back once a phase is done, with rsync (RUNBOOK § Copying back).
# Follow: tail -f /data/scc/jonas.schrade/thesis-runs/audit_<phase>_<sub>_<topic>_<model>.o<jobid>
#
# Queue and resources as run_stage.sh, so pull_models.sh check's decode rates,
# measured there, carry over; h_rt is run_stage.sh's ceiling, not a cost estimate.

#$ -q gpu@scc214
#$ -l h_vmem=96G,rtx_6000=1,gpu=1
#$ -l h_rt=168:00:00
#$ -N thesis_audit
#$ -o /data/scc/jonas.schrade/thesis-runs/ -j y

set -euo pipefail
RUNDIR=/data/scc/jonas.schrade/thesis-runs

[ -n "${MODEL:-}" ] && [ -n "${TOPIC:-}" ] && [ -n "${PHASE:-}" ] && [ -n "${SUB:-}" ] || {
  echo "ABORT: needs -v MODEL=<run tag>,TOPIC=<config.TOPICS key>,PHASE=<phase>,SUB=<sub>; see this script's head."
  exit 1
}
export SIM_LLM_MODEL="$MODEL" SIM_TOPIC="$TOPIC"
JOBDIR="${TOPIC}_${MODEL//:/-}/${PHASE}_${SUB}"   # build_audit.py's directory
echo "Audit $JOBDIR"

# One job per directory: two would append to the same events.jsonl, and O_APPEND
# is not atomic over NFS. mkdir is.
LOCK="$RUNDIR/.audit_${JOBDIR//\//_}.lock"
mkdir "$LOCK" 2>/dev/null || {
  echo "ABORT: this sub-phase is already running. Lock: $LOCK"
  echo "If orphaned (qdel or a hard kill): rmdir '$LOCK'"
  exit 1
}
trap 'kill "${OLLAMA_PID:-}" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT
trap 'exit 143' TERM INT

# setup_thesis_env.sh appends to unset variables and dies under -u.
set +u
source ~/setup_thesis_env.sh
set -u
which ollama | grep -q '/data/scc' || { echo "ABORT: wrong ollama binary: $(which ollama)"; exit 1; }

# The starter method's GPU; without it a free card other than index 0 (the discovery trap).
if [ -n "${SGE_GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SGE_GPU// /,}"
else
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi \
    --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1!=0 && $2<1000 {print $1; exit}')
  [ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "ABORT: no free GPU other than 0"; exit 1; }
fi
echo "node $(hostname), GPU $CUDA_VISIBLE_DEVICES (SGE_GPU='${SGE_GPU:-unset}')"

# Own port per job (own jobs share nodes); threads capped to the job's slots
# (RUNBOOK § Troubleshooting).
export OLLAMA_HOST=127.0.0.1:$((11500 + JOB_ID % 400))
export LLAMA_ARG_THREADS=${NSLOTS:-1}
# A model load can outlast Ollama's 5-min default (RUNBOOK § Troubleshooting).
export OLLAMA_LOAD_TIMEOUT=30m
GGML_CUDA_NO_VMM=1 ollama serve > "$HOME/ollama_$JOB_ID.log" 2>&1 &
OLLAMA_PID=$!

# Wait for the discovery line and require CUDA positively; an empty log is no evidence.
for _ in $(seq 1 60); do
  grep -q "inference compute" "$HOME/ollama_$JOB_ID.log" && break
  sleep 2
done
if ! grep -qi "library=cuda" "$HOME/ollama_$JOB_ID.log"; then
  echo "ABORT: Ollama not on CUDA (discovery trap? GPU 0 busy?), see ollama_$JOB_ID.log"
  exit 1
fi
ollama show "$MODEL" > /dev/null 2>&1 || {
  echo "ABORT: model '$MODEL' is not pulled; run simulation/pull_models.sh pull on scc2."
  exit 1
}
echo "Ollama on GPU, $MODEL present."

# Output under $RUNDIR, as runs': the repo checkout stays clean, so params.json's
# git_commit is the tag's.
cd "$RUNDIR"
python -u ~/projects/thesis/analysis/audit/build_audit.py --phase "$PHASE" --sub "$SUB"
