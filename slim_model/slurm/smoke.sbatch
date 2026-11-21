#!/bin/bash
#SBATCH --job-name=slim_smoke
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=slim_model/logs/slim_smoke_%j.out
#SBATCH --error=slim_model/logs/slim_smoke_%j.err
# Run this FIRST, from the repo root: sbatch slim_model/slurm/smoke.sbatch
# Fake data + tiny model on all GPUs of the node: multi-GPU training, resume, time-limit stop,
# distillation and eval. Must end with "SMOKE TEST PASSED". Writes only under slim_model/_smoke/.
source "${REPO:-/cosmos/nfs/home/dyvo/NonlinearMechQuarter1}/slim_model/slurm/_env.sh"
bash slim_model/smoke_test.sh
