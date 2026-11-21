"""Train one (slim) variant of the video diffusion model. Run from the repo root.

1 GPU / CPU:   python slim_model/train_slim.py --config slim_model/configs/mults1244.yaml --run_name mults1244
4 GPUs:        accelerate launch --multi_gpu --num_processes 4 slim_model/train_slim.py --config ... --run_name ...
On Cosmos use slim_model/slurm/train.sbatch, which does the above and resumes automatically.

Data and run folders default to the vast scratch layout main_modern.py uses (see slim/paths.py).
Distillation (idea 4): add --teacher_run runs/pretrained/pretrained (the authors' released 200k-step model).
"""
import slim  # noqa: F401  (first: puts the repo root on sys.path)
import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml
from torch import distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed

from denoising_diffusion_pytorch.video_denoising_diffusion_pytorch import GaussianDiffusion
from slim import paths
from slim.common import (build_diffusion, ckpt_path, is_multi_process, latest_step, load_config, load_weights,
                         param_report, resolve_step)
from slim.trainer import STOP_FOR_TIME, DistillDiffusion, SlimTrainer

RESUME_MARKER = 'RESUME_NEEDED'  # written when the time budget stopped training early
DONE_MARKER = 'TRAINING_DONE'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='e.g. slim_model/configs/mults1244.yaml')
    p.add_argument('--run_name', required=True, help='output goes to <runs_dir>/<run_name>/')
    p.add_argument('--data_root', default=paths.DATA_ROOT, help='folder holding training/ and validation/')
    p.add_argument('--runs_dir', default=paths.RUNS_DIR, help='each run goes to <runs_dir>/<run_name>/')
    p.add_argument('--resume', default='auto', help="'auto' = latest checkpoint if any, 'none' = must be a new run, or a step number")
    p.add_argument('--steps', type=int, default=None, help='override train_num_steps from the config')
    p.add_argument('--save_every', type=int, default=None, help='override save_and_sample_every')
    p.add_argument('--batch_size', type=int, default=None, help='override per-GPU batch size')
    p.add_argument('--lr', type=float, default=None, help='override learning rate')
    p.add_argument('--num_preview_samples', type=int, default=4,
                   help='validation conditions sampled at each checkpoint (min 1; repeated so every GPU gets one)')
    p.add_argument('--teacher_run', default=None,
                   help=f'run dir of a trained full model to distill from (idea 4), e.g. {paths.PRETRAINED_RUN}')
    p.add_argument('--teacher_step', default='latest')
    p.add_argument('--distill_weight', type=float, default=1.0)
    p.add_argument('--mixed_precision', default='fp16', choices=['no', 'fp16', 'bf16'])
    p.add_argument('--dist_backend', default='nccl', choices=['nccl', 'gloo'],
                   help="nccl (RCCL on ROCm) = what main_modern.py uses on Cosmos; gloo = the authors' original choice")
    p.add_argument('--num_workers', type=int, default=4, help='DataLoader workers per GPU')
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--time_budget_hours', type=float, default=0.0,
                   help='save and exit after this many hours, leaving <run>/RESUME_NEEDED (0 = off); slurm/train.sbatch resubmits')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wandb_user', default=None, help='wandb entity to log to (optional)')
    p.add_argument('--param_count_only', action='store_true', help='print the parameter count and exit')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    for key, val in (('train_num_steps', args.steps), ('save_and_sample_every', args.save_every),
                     ('batch_size', args.batch_size), ('learning_rate', args.lr)):
        if val is not None:
            cfg[key] = val

    run_dir = Path(args.runs_dir) / args.run_name
    saved_cfg_path = run_dir / 'model' / 'model.yaml'

    # resume logic: the saved config wins, so a resumed run cannot silently change architecture
    resume_step = None
    existing = latest_step(run_dir) if run_dir.exists() else None
    if args.resume == 'none' and existing is not None:
        sys.exit(f'{run_dir} already has checkpoints; use --resume auto or a new --run_name')
    if args.resume not in ('auto', 'none'):
        resume_step = int(args.resume)
    elif args.resume == 'auto' and existing is not None:
        resume_step = existing
    if resume_step is not None:
        saved = load_config(saved_cfg_path)
        for k in ('unet_dim', 'unet_dim_mults', 'selected_channels', 'train_timesteps'):
            if saved[k] != cfg[k]:
                sys.exit(f'{k} in {args.config} ({cfg[k]}) differs from the run being resumed ({saved[k]})')
        saved['train_num_steps'] = cfg['train_num_steps']  # allow extending a run
        cfg = saved

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ip_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60 * 10))
    if args.dist_backend == 'nccl' and torch.cuda.is_available():
        ip_kwargs = InitProcessGroupKwargs(backend='nccl', timeout=datetime.timedelta(seconds=60 * 10))  # as main_modern.py
    elif is_multi_process() and not dist.is_initialized():
        dist.init_process_group(backend='gloo')  # the authors' main.py; also the only option on CPU
    accelerator = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs, ip_kwargs],
                              log_with='wandb' if args.wandb_user else None)
    set_seed(args.seed)

    distill = args.teacher_run is not None
    diffusion = build_diffusion(cfg, cls=DistillDiffusion if distill else GaussianDiffusion,
                                **({'distill_weight': args.distill_weight} if distill else {}))
    accelerator.print(param_report(diffusion, cfg))
    if args.param_count_only:
        return 0

    if distill:
        t_cfg = load_config(Path(args.teacher_run) / 'model' / 'model.yaml')
        t_step = resolve_step(args.teacher_run, args.teacher_step)
        teacher = build_diffusion(t_cfg)
        load_weights(teacher, ckpt_path(args.teacher_run, t_step), use_ema=True)
        diffusion.set_teacher(teacher.to(accelerator.device))
        accelerator.print(f'distilling from {args.teacher_run} step {t_step} '
                          f'({sum(p.numel() for p in teacher.denoise_fn.parameters()) / 1e6:.2f}M params), '
                          f'weight {args.distill_weight}')

    if accelerator.is_main_process:
        (run_dir / 'model').mkdir(parents=True, exist_ok=True)
        if resume_step is None:
            with open(saved_cfg_path, 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            shutil.copy(args.config, run_dir / 'model' / 'source_config.yaml')
        with open(run_dir / 'launch_args.txt', 'a') as f:
            f.write(' '.join(sys.argv) + f'  # world_size={accelerator.num_processes}\n')
    accelerator.wait_for_everyone()

    data_root = args.data_root.rstrip('/') + '/'
    trainer = SlimTrainer(
        diffusion,
        folder=data_root + 'training/',
        validation_folder=data_root + 'validation/',
        results_folder=str(run_dir),
        selected_channels=cfg['selected_channels'],
        train_batch_size=cfg['batch_size'],
        test_batch_size=cfg['batch_size'],
        train_lr=cfg['learning_rate'],
        save_and_sample_every=cfg['save_and_sample_every'],
        train_num_steps=cfg['train_num_steps'],
        ema_decay=cfg['ema_decay'],
        log=True,
        null_cond_prob=cfg['null_cond_prob'],
        per_frame_cond=cfg['per_frame_cond'],
        reference_frame=cfg['reference_frame'],
        run_name=args.run_name,
        accelerator=accelerator,
        wandb_username=args.wandb_user,
        num_workers=args.num_workers,
        log_every=args.log_every,
        time_budget_hours=args.time_budget_hours,
    )
    accelerator.print(f'global batch size: {cfg["batch_size"] * accelerator.num_processes} '
                      f'({accelerator.num_processes} process(es) x {cfg["batch_size"]})')
    rc = trainer.train(resume_step=resume_step, num_samples=max(1, args.num_preview_samples))
    # Marker files tell slurm/train.sbatch what happened (torchrun/accelerate turn a worker's exit code 99
    # into 1, so the exit code alone is not reliable with several GPUs).
    if accelerator.is_main_process:
        resume_marker, done_marker = run_dir / RESUME_MARKER, run_dir / DONE_MARKER
        if rc == STOP_FOR_TIME:
            resume_marker.touch()
            done_marker.unlink(missing_ok=True)
        else:
            resume_marker.unlink(missing_ok=True)
            done_marker.write_text(f'trained to step {trainer.step}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
