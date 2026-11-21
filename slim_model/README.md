# slim_model: training smaller versions of the diffusion model

This folder trains and compares slimmer versions of the Bastek & Kochmann video diffusion model, using this
repo's own `denoising_diffusion_pytorch/` and `src/` (imported, not copied). It only adds files: nothing
outside `slim_model/` is changed, and `main.py` / `main_modern.py` work exactly as before.

Defaults match `main_modern.py` on Cosmos: data in `/cosmos/vast/scratch/dyvo/data/lagrangian/`, runs written
to `/cosmos/vast/scratch/dyvo/nonlinearmech_runs/slim/<run_name>/`, the authors' checkpoint at
`runs/pretrained/pretrained/model/step_200000/checkpoint.pt`, and the same modules, venv, MIOpen settings and
`--partition=cluster --exclusive` as `run_modern.slurm`.

## What gets compared

| Variant | Idea | Denoiser params | vs full |
|---|---|---|---|
| `runs/pretrained/pretrained` (authors' 200k-step model) | 1. fewer sampler steps (DDIM), no training | 39,547,203 | 100% |
| `configs/full.yaml` | baseline, same short budget as the slim ones | 39,547,203 | 100% |
| `configs/mults1244.yaml` | 2. deepest level 256 instead of 512 channels | 23,284,291 | 59% |
| `configs/dim48.yaml` | 3. base width 48 instead of 64 | 23,733,891 | 60% |
| `configs/dim48_mults1244.yaml` (optional) | 2 + 3 together | 14,338,371 | 36% |
| `mults1244` + `--teacher_run runs/pretrained/pretrained` (optional) | 4. distillation from the full model | 23,284,291 | 59% |

`python slim_model/count_params.py --detail` prints these from the code, with a per-level breakdown.

Every trained variant uses the same settings: 30,000 steps (15% of the paper's 200k), batch 8 per GPU
(32 global on 4 GPUs, the paper's global batch), lr 1e-4, fp16, EMA 0.995, checkpoint + validation loss
every 5,000 steps. Change `train_num_steps` in the configs for a longer budget, but keep it equal across variants.

## Run it (from the repo root on Cosmos)

```bash
cd /cosmos/nfs/home/dyvo/NonlinearMechQuarter1
git lfs pull                                   # only if runs/pretrained/... is still a 130-byte pointer
sbatch slim_model/slurm/smoke.sbatch           # fake data + tiny model on 4 GPUs; log must end "SMOKE TEST PASSED"
bash slim_model/slurm/submit_ablation.sh       # pretrained eval + full, mults1244, dim48 training jobs
#   WITH_COMBINED=1 WITH_DISTILL=1 bash slim_model/slurm/submit_ablation.sh   # also the two optional runs
squeue -u $USER
tail -f slim_model/logs/slim_train_<jobid>.out # step, train loss, it/s every 100 steps; val loss every 5k
python slim_model/compare_results.py --plot val_loss.png   # comparison table, any time
```

Always `sbatch` from the repo root: the `#SBATCH --output` paths are relative to where you submit.

What happens without you: each training job resumes from its latest checkpoint, stops cleanly ~1.5 h before
the 24 h limit and resubmits itself until 30k steps are done, then submits its own eval job
(`AUTO_RESUBMIT=0` / `AUTO_EVAL=0` turn those off). Re-running the same `sbatch` line is always safe: it
resumes. One variant by hand:

```bash
sbatch slim_model/slurm/train.sbatch slim_model/configs/mults1244.yaml mults1244
sbatch slim_model/slurm/eval.sbatch mults1244 --sampling_steps 256,100,50
```

## Changing paths or environment

Nothing needs editing if your layout matches `main_modern.py`. Otherwise use environment variables
(they pass through `sbatch`):

* `SLIM_DATA_ROOT` (default `/cosmos/vast/scratch/dyvo/data/lagrangian`, must hold `training/` and `validation/`)
* `SLIM_RUNS_DIR` (default `/cosmos/vast/scratch/dyvo/nonlinearmech_runs/slim`)
* `SLIM_PRETRAINED` (default `runs/pretrained/pretrained`)
* `REPO` (default `/cosmos/nfs/home/dyvo/NonlinearMechQuarter1`), `VENV` (default `~/venvs/nonlinearmech_rocm`)
* `TIME_BUDGET_HOURS` (default 22.5), `EVAL_STEPS` (default `256,100,50`)

The Python scripts also take `--data_root`, `--runs_dir` and the rest as flags (`--help` lists them).

## What to send back

`<runs>/ablation_results.csv` (one row per run x sampler steps), the output of
`python slim_model/compare_results.py`, `val_loss.png`, and `<runs>/*/metrics.csv`. Columns that matter:

* `nrmse_mean`: NRMSE between the target stress-strain curve and the curve read off the generated video
  (the authors' pixel estimate, `slim/stress.py`), over 20 held-out validation targets x 3 samples. Lower is better.
* `nrmse_best_of_k`: the best of the 3 samples per target, averaged (closer to how the paper picks designs).
* `sec_per_sample`: generation time, the thing DDIM speeds up.
* `val_loss_ema`: denoising loss on validation data with the EMA weights.
* `empty_topology_frac`: share of samples with no material found. Should be ~0; if not, that variant is broken.

Every run is scored on the same 20 targets and the same starting noise (`--seed 0`).

## How this differs from main_modern.py

* `train_slim.py` replaces `main_modern.py` for training: `--config`, `--run_name`, `dim_mults` from the
  config, parameter count at start, `--resume auto`, `--time_budget_hours`, optional distillation. It uses
  the nccl backend (RCCL on ROCm) like `main_modern.py`; `--dist_backend gloo` is the fallback.
* `slim/trainer.py` subclasses the repo's `Trainer` (that file is not edited) and fixes: resuming advanced
  the step counter twice per batch; checkpoints were written only at the very end, by every GPU into the
  same file; validation loss only went to wandb. Checkpoints are now written by one process, atomically,
  without the DDP `module.` prefix (they load on any GPU count), every 5,000 steps. Losses go to
  `<run>/metrics.csv`.
* The validation pass (`eval_network`, commented out in the repo's `train()`) runs at every checkpoint. It
  writes preview gifs to `<run>/training/step_N/`; the trainer keeps the results path relative so the
  repo's `'./' + results_folder` paths land in the run folder, not in `./cosmos/...` under the repo.
* DDIM was already in the code (`sampling_timesteps < train_timesteps`); `eval_slim.py` just sweeps it.
* Distillation is output-level only (student matches the teacher's predicted noise, weight 1, plus the
  normal loss).

## Tested

CPU only, on fake data, against this repo's `denoising_diffusion_pytorch` (commit `5b0b43f`): parameter
counts, training, resume, time-limit stop, distillation, eval (`smoke_test.sh`), with run folders given as
absolute paths like the `/cosmos/vast/...` ones; plus 2-process training, resume and time-limit stop over gloo.
Checkpoints also load with `torch.load(..., weights_only=True)`, as the repo's `Trainer.load` does.
Not yet run on GPUs, with nccl/RCCL, or on the real data, and the authors' 620 MB checkpoint was not
available to the test, so run `smoke.sbatch` first.

## Files

| File | What it does |
|---|---|
| `train_slim.py` | train one variant (resumable, time-limit aware, optional distillation) |
| `eval_slim.py` | NRMSE on held-out targets at several sampler step counts |
| `count_params.py` | real parameter counts, optional forward/backward check |
| `compare_results.py` | comparison table and validation-loss plot |
| `configs/*.yaml` | the variants, plus `tiny_smoke.yaml` for testing |
| `slurm/*` | job scripts (`_env.sh` holds the modules/venv), one-command ablation submit |
| `slim/*` | shared code: paths, model building, fixed trainer, stress-curve estimate |
| `smoke_test.sh`, `make_fake_data.py` | end-to-end test on random data (writes only to `slim_model/_smoke/`) |
