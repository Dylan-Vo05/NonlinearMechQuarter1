#!/bin/bash
#SBATCH --job-name=nonlinear_baseline_modern
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=23:59:59
#SBATCH --output=baseline_modern_%j.out
#SBATCH --error=baseline_modern_%j.err

module purge
module load rocm/7.0.0
module load cray-python/3.12.12

source ~/venvs/nonlinearmech_modern/bin/activate

cd /cosmos/nfs/home/dyvo/NonlinearMechQuarter1

echo "Starting NonlinearMech baseline test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

echo "Python executable:"
which python

echo "Python version:"
python --version

echo "Accelerate package:"
python -m pip show accelerate

echo "Accelerate executable:"
which accelerate

echo "PATH:"
echo $PATH

export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3
export MIOPEN_USER_DB_PATH=/tmp/$USER/my-miopen-cache
export PYTORCH_TUNABLEOP_ENABLED=1

python -m accelerate.commands.launch \
    --mixed_precision=fp16 \
    --num_processes=4 \
    --num_machines=1 \
    --dynamo_backend=no \
    main_modern.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Baseline TEST PASSED"
else
    echo "Baseline TEST FAILED: exit code $EXIT_CODE"
fi

exit $EXIT_CODE
