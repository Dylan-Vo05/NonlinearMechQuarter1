#!/bin/bash
#SBATCH --job-name=video_diffusion
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Load Singularity
module load singularitypro

# Define directories
PROJECT="$HOME/NonlinearMechQuarter1"
CONTAINER="$HOME/containers/nonlinearmech-rocm7.sif"
DATA="/cosmos/vast/scratch/$USER/data"

# Move to project directory
cd "$PROJECT"

# Run distributed training
singularity exec --rocm \
    --bind "$PROJECT":/workspace \
    --bind "$DATA":/data \
    --pwd /workspace \
    "$CONTAINER" \
    accelerate launch \
        --multi_gpu \
        --mixed_precision=fp16 \
        --num_processes 4 \
        --dynamo_backend=no \
        main.py