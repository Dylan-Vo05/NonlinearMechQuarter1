# Sourced by every slim_model job script. Same environment as run_modern.slurm.
# Override without editing: REPO=... VENV=... sbatch ...
REPO=${REPO:-/cosmos/nfs/home/dyvo/NonlinearMechQuarter1}
VENV=${VENV:-$HOME/venvs/nonlinearmech_rocm}

module purge
module load rocm/7.0.0
module load cray-python/3.12.12

source "$VENV/bin/activate"
cd "$REPO"

export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3
export MIOPEN_USER_DB_PATH=/tmp/$USER/my-miopen-cache
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"

# GPUs visible to this job (4 on a Cosmos node with --exclusive)
NGPU=${NGPU:-$(python -c "import torch; print(torch.cuda.device_count())")}
echo "GPUs: $NGPU"
