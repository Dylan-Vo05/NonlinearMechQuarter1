#!/bin/bash
#SBATCH --job-name=nonlinear_baseline
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=23:59:59
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err

module purge
module load rocm/7.0.0
module load cray-python/3.12.12

# Activate your existing Python environment
source ~/venvs/nonlinearmech/bin/activate

cd /cosmos/nfs/home/dyvo/NonlinearMechQuarter1

echo "=========================================="
echo "Starting NonlinearMech baseline test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1


accelerate launch main.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Baseline TEST PASSED"
else
    echo "Baseline TEST FAILED: exit code $EXIT_CODE"
fi

exit $EXIT_CODE