#!/bin/bash
# End-to-end check on fake data with a tiny model: param counts, training, resume, time-budget stop,
# distillation and evaluation. A few minutes on CPU, about a minute on GPUs.
# Run from the repo root:  bash slim_model/smoke_test.sh        (on Cosmos: sbatch slim_model/slurm/smoke.sbatch)
# Writes only under slim_model/_smoke/ (git-ignored); your real runs and data are not touched.
set -euo pipefail
S=slim_model
OUT=${SMOKE_DIR:-$S/_smoke}
NGPU=${NGPU:-$(python -c "import torch; print(torch.cuda.device_count())")}
if [ "$NGPU" -gt 1 ]; then
  RUN="python -m accelerate.commands.launch --multi_gpu --num_processes $NGPU --num_machines 1 --mixed_precision fp16 --dynamo_backend no --main_process_port $((29500 + RANDOM % 1000))"
  MP=fp16; BACKEND=nccl
elif [ "$NGPU" -eq 1 ]; then
  RUN="python"; MP=fp16; BACKEND=nccl
else
  RUN="python"; MP=no; BACKEND=gloo
fi
COMMON="--config $S/configs/tiny_smoke.yaml --data_root $OUT/data --runs_dir $OUT/runs --log_every 1 --num_workers 0 --mixed_precision $MP --dist_backend $BACKEND --num_preview_samples 1"
echo "== smoke test with $NGPU GPU(s): $RUN"

echo "== 1. real parameter counts"
python $S/count_params.py

echo "== 2. fake data"
rm -rf "$OUT"
python $S/make_fake_data.py --out "$OUT/data" --n_train 8 --n_val 4

echo "== 3. train 2 steps, then resume and extend to 4"
$RUN $S/train_slim.py $COMMON --run_name smoke --steps 2 --save_every 2
$RUN $S/train_slim.py $COMMON --run_name smoke --steps 4 --save_every 2
test -f "$OUT/runs/smoke/model/step_4/checkpoint.pt"
test -f "$OUT/runs/smoke/training/step_2/gifs/prediction_channel_3.gif"   # preview samples landed in the run folder

echo "== 4. time-budget stop (expects a RESUME_NEEDED marker and a step-5 checkpoint)"
$RUN $S/train_slim.py $COMMON --run_name smoke --steps 8 --save_every 100 --time_budget_hours 0.0000001
test -f "$OUT/runs/smoke/RESUME_NEEDED"
test -f "$OUT/runs/smoke/model/step_5/checkpoint.pt"

echo "== 5. distillation from the smoke run"
$RUN $S/train_slim.py $COMMON --run_name smoke_distill --steps 2 --save_every 2 --teacher_run "$OUT/runs/smoke"

# the original 256-step sampler is slow on CPU, so it is only included when a GPU is present
if [ "$NGPU" -gt 0 ]; then STEPS=256,8,4; else STEPS=8,4; fi
echo "== 6. evaluation at sampler steps $STEPS (256 = authors' sampler, fewer = DDIM)"
python $S/eval_slim.py --run_dir smoke --runs_dir "$OUT/runs" --data_root "$OUT/data" --sampling_steps $STEPS \
    --n_targets 2 --n_samples 2 --batch_size 4 --val_batches 2
python $S/compare_results.py --runs_dir "$OUT/runs"

echo "== SMOKE TEST PASSED (numbers above are meaningless: random data, 5 training steps)"
