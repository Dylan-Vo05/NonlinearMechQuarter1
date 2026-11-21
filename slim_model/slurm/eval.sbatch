#!/bin/bash
#SBATCH --job-name=slim_eval
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=slim_model/logs/slim_eval_%j.out
#SBATCH --error=slim_model/logs/slim_eval_%j.err
#
# usage (from the repo root):
#   sbatch slim_model/slurm/eval.sbatch <run_name or run dir> [extra eval_slim.py args, e.g. --sampling_steps 256,100,50,25]
# Uses one GPU. The authors' model:  sbatch slim_model/slurm/eval.sbatch runs/pretrained/pretrained --step 200000
RUN_DIR=${1:?usage: sbatch slim_model/slurm/eval.sbatch <run_name or run dir> [extra args]}
shift
source "${REPO:-/cosmos/nfs/home/dyvo/NonlinearMechQuarter1}/slim_model/slurm/_env.sh"
set -o pipefail   # after _env.sh: module scripts do not like set -u
python slim_model/eval_slim.py --run_dir "$RUN_DIR" "$@"
EXIT_CODE=$?
python slim_model/compare_results.py
exit $EXIT_CODE
