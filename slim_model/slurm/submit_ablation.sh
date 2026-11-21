#!/bin/bash
# Submits the whole ablation. Run from the repo root:  bash slim_model/slurm/submit_ablation.sh
#   1. eval of the authors' pretrained model at 256 / 100 / 50 / 25 sampler steps   (idea 1, no training)
#   2. short-budget training of full, mults1244 and dim48 with identical settings    (ideas 2 and 3)
#      each training job resubmits itself until done, then submits its own eval job
# Optional:  WITH_COMBINED=1 (dim48 + mults1244)   WITH_DISTILL=1 (mults1244 distilled from pretrained, idea 4)
set -euo pipefail
cd "${REPO:-/cosmos/nfs/home/dyvo/NonlinearMechQuarter1}"
PRE_RUN=${SLIM_PRETRAINED:-runs/pretrained/pretrained}
PRE=$PRE_RUN/model/step_200000/checkpoint.pt
# a Git LFS pointer is ~130 bytes; the real checkpoint is ~620 MB
if [ -f "$PRE" ] && [ "$(stat -c %s "$PRE")" -gt 1000000 ]; then
  sbatch slim_model/slurm/eval.sbatch "$PRE_RUN" --step 200000 --sampling_steps 256,100,50,25
else
  echo "skipping pretrained eval: $PRE missing or only a Git LFS pointer (run: git lfs pull)"
  PRE=""
fi
for v in full mults1244 dim48; do
  sbatch slim_model/slurm/train.sbatch "slim_model/configs/$v.yaml" "$v"
done
if [ "${WITH_COMBINED:-0}" = 1 ]; then
  sbatch slim_model/slurm/train.sbatch slim_model/configs/dim48_mults1244.yaml dim48_mults1244
fi
if [ "${WITH_DISTILL:-0}" = 1 ]; then
  [ -n "$PRE" ] || { echo "distillation needs the pretrained checkpoint"; exit 1; }
  sbatch slim_model/slurm/train.sbatch slim_model/configs/mults1244.yaml mults1244_distill \
      --teacher_run "$PRE_RUN" --teacher_step 200000
fi
squeue -u "$USER"
