#!/bin/bash
# Recovery on the SCC's CPUs (analysis/schedule.md § Recovery): one SGE array task per
# dataset, `recovery.py task` writing $RUNDIR/recovery/tasks/task_<id>.json. No queue,
# host or GPU is requested: the scheduler places each task on any node with a free slot.
# Dataset id = SGE_TASK_ID - 1, its seed; ids 0 … 35 hold each of the nine cells and four
# generating models once. Inputs and setup: RUNBOOK § Recovery on the CPUs.
#
#   S=~/projects/thesis/analysis/run_recovery.sh
#   qsub -t 1-36 -l h_vmem=4G,h_rt=24:00:00 $S        # the test batch
#   qacct -j <job> | grep -E 'maxvmem|ru_wallclock'    # sets the rest's h_vmem and h_rt
#   qsub -t 37-3600 -l h_vmem=<…>,h_rt=<…> $S
# The test batch's limits are generous guesses, there only until qacct replaces them with
# what its tasks used. A task whose file exists is skipped, so
# resubmitting a range redoes only the missing ones, which `recovery.py merge` lists.

#$ -N thesis_recovery
#$ -o /data/scc/jonas.schrade/thesis-runs/recovery/ -j y

set -euo pipefail
RUNDIR=/data/scc/jonas.schrade/thesis-runs
VENV=/data/scc/jonas.schrade/envs/thesis   # the simulation's, plus pandas (RUNBOOK)
REC="$RUNDIR/recovery"

[ "${SGE_TASK_ID:-undefined}" != undefined ] || { echo "ABORT: submit as an array (qsub -t)"; exit 1; }
[ -f "$REC/settings.json" ] || { echo "ABORT: no $REC/settings.json (RUNBOOK § Recovery on the CPUs)"; exit 1; }

# setup_thesis_env.sh appends to unset variables and dies under -u.
set +u
source ~/setup_thesis_env.sh
source "$VENV/bin/activate"
set -u
[ "$(command -v python)" = "$VENV/bin/python" ] || { echo "ABORT: python is $(command -v python), not $VENV's"; exit 1; }
# numpy's BLAS would otherwise start a thread per core of the node in a one-slot job.
export OMP_NUM_THREADS=${NSLOTS:-1} MKL_NUM_THREADS=${NSLOTS:-1} OPENBLAS_NUM_THREADS=${NSLOTS:-1}

cd ~/projects/thesis
ID=$((SGE_TASK_ID - 1))
echo "recovery task $ID on $(hostname), $(git describe --always --dirty)"
python analysis/recovery.py task "$ID" "$REC/settings.json" "$RUNDIR/logs/audit" "$REC/tasks"
