#!/bin/bash
#SBATCH --job-name=nonlinear_smoke
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err

module purge
module load rocm/7.0.0
module load cray-python/3.12.12

# Activate your existing Python environment
source ~/venvs/nonlinearmech/bin/activate

cd /cosmos/nfs/home/dyvo/NonlinearMechQuarter1

echo "=========================================="
echo "Starting NonlinearMech smoke test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

python -u smoke_test.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SMOKE TEST PASSED"
else
    echo "SMOKE TEST FAILED: exit code $EXIT_CODE"
fi

exit $EXIT_CODE