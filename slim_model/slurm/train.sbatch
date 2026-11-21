#!/bin/bash
#SBATCH --job-name=slim_train
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=23:59:59
#SBATCH --output=slim_model/logs/slim_train_%j.out
#SBATCH --error=slim_model/logs/slim_train_%j.err
#
# usage (from the repo root):
#   sbatch slim_model/slurm/train.sbatch slim_model/configs/mults1244.yaml mults1244 [extra train_slim.py args]
# Resumes from the latest checkpoint automatically. Near the time limit it saves, leaves
# <runs>/<run_name>/RESUME_NEEDED and resubmits itself (AUTO_RESUBMIT=0 turns that off).
# When training finishes it submits slurm/eval.sbatch for the run (AUTO_EVAL=0 turns that off).
CONFIG=${1:?usage: sbatch slim_model/slurm/train.sbatch <config.yaml> <run_name> [extra args]}
RUN=${2:?usage: sbatch slim_model/slurm/train.sbatch <config.yaml> <run_name> [extra args]}
shift 2
source "${REPO:-/cosmos/nfs/home/dyvo/NonlinearMechQuarter1}/slim_model/slurm/_env.sh"
set -o pipefail   # after _env.sh: module scripts do not like set -u
RUNS_DIR=$(python -c "import sys; sys.path.insert(0, 'slim_model'); from slim import paths; print(paths.RUNS_DIR)")

# stop ~1.5 h before the 24 h limit: the first validation pass can take ~35 min while MIOpen tunes
TIME_BUDGET_HOURS=${TIME_BUDGET_HOURS:-22.5}
PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
MULTI=""; [ "$NGPU" -gt 1 ] && MULTI="--multi_gpu"
echo "Starting slim training: config $CONFIG, run $RUN, runs dir $RUNS_DIR, extra: $*"

python -m accelerate.commands.launch $MULTI \
    --mixed_precision=fp16 \
    --num_processes="$NGPU" \
    --num_machines=1 \
    --dynamo_backend=no \
    --main_process_port="$PORT" \
    slim_model/train_slim.py --config "$CONFIG" --run_name "$RUN" --resume auto \
    --time_budget_hours "$TIME_BUDGET_HOURS" --num_workers 4 "$@"
EXIT_CODE=$?

if [ -f "$RUNS_DIR/$RUN/RESUME_NEEDED" ] && [ "${AUTO_RESUBMIT:-1}" = 1 ]; then
  echo "time budget reached; resubmitting to continue $RUN"
  sbatch slim_model/slurm/train.sbatch "$CONFIG" "$RUN" "$@"
  exit 0
fi
if [ $EXIT_CODE -eq 0 ] && [ -f "$RUNS_DIR/$RUN/TRAINING_DONE" ] && [ "${AUTO_EVAL:-1}" = 1 ]; then
  sbatch slim_model/slurm/eval.sbatch "$RUN" --sampling_steps "${EVAL_STEPS:-256,100,50}"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "Slim training PASSED"
else
    echo "Slim training FAILED: exit code $EXIT_CODE"
fi
exit $EXIT_CODE
