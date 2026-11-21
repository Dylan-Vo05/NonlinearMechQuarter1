"""Score a trained run: stress-curve NRMSE on held-out validation targets, at several sampler step counts.

Run from the repo root:
    python slim_model/eval_slim.py --run_dir runs/pretrained/pretrained --sampling_steps 256,100,50,25   # idea 1
    python slim_model/eval_slim.py --run_dir mults1244 --sampling_steps 100,50                          # a slim variant
--run_dir is a path, or a bare run name looked up in the runs folder (slim/paths.py).

For each held-out target (a real stress-strain curve from the validation set) it samples --n_samples
designs with classifier-free guidance, reads the model's own predicted stress curve from each generated
video (slim/stress.py, the authors' pixel estimate), and reports NRMSE against the target.
'best_of_k' mimics the paper (generate several, keep the best). Targets are picked with --seed, so every
run is scored on the same targets. 256 steps = the authors' original sampler; fewer = DDIM (eta = 0).

Writes <run_dir>/eval_slim/step_<N>/{per_sample.csv,summary.json,curves.npz} and appends one row per
step count to --results_csv (shared by all runs, read by compare_results.py).
"""
import slim  # noqa: F401  (first: puts the repo root on sys.path)
import argparse
import csv
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

from denoising_diffusion_pytorch.video_denoising_diffusion_pytorch import Dataset
from slim import paths
from slim.common import build_diffusion, ckpt_path, count_params, load_config, load_weights, resolve_step, set_sampling_steps
from slim.stress import nrmse, predicted_stress_curve

SUMMARY_FIELDS = ['run', 'step', 'params_M', 'dim', 'dim_mults', 'sampling_steps', 'guidance', 'n_targets', 'n_samples',
                  'nrmse_mean', 'nrmse_median', 'nrmse_best_of_k', 'empty_topology_frac', 'sec_per_sample',
                  'val_loss_ema', 'device', 'weights']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', required=True, help='run folder, or a run name inside --runs_dir')
    p.add_argument('--runs_dir', default=paths.RUNS_DIR)
    p.add_argument('--step', default='latest')
    p.add_argument('--data_root', default=paths.DATA_ROOT)
    p.add_argument('--sampling_steps', default='256,100,50,25')
    p.add_argument('--guidance', type=float, default=5.0)
    p.add_argument('--n_targets', type=int, default=20)
    p.add_argument('--n_samples', type=int, default=3, help='designs per target')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--seed', type=int, default=0, help='same seed = same targets and same starting noise')
    p.add_argument('--val_batches', type=int, default=50, help='batches for the validation loss (0 = skip)')
    p.add_argument('--raw_weights', action='store_true', help='use raw instead of EMA weights')
    p.add_argument('--results_csv', default=None, help='default: <runs_dir>/ablation_results.csv')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def append_row(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # several eval jobs may finish at once
        f.seek(0)
        new = f.read(1) == ''
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


@torch.no_grad()
def validation_loss(diffusion, ds, batch_size, n_batches, null_cond_prob, seed, device):
    torch.manual_seed(seed)
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    losses = []
    for i, (x, cond) in enumerate(dl):
        if i >= n_batches:
            break
        losses.append(diffusion(x.to(device), cond=cond.to(device), null_cond_prob=null_cond_prob).item())
    return float(np.mean(losses)) if losses else float('nan')


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not (run_dir / 'model' / 'model.yaml').exists() and (Path(args.runs_dir) / args.run_dir).exists():
        run_dir = Path(args.runs_dir) / args.run_dir
    if args.results_csv is None:
        args.results_csv = str(Path(args.runs_dir) / 'ablation_results.csv')
    cfg = load_config(run_dir / 'model' / 'model.yaml')
    step = resolve_step(run_dir, args.step)
    device = torch.device(args.device)

    diffusion = build_diffusion(cfg)
    _, which = load_weights(diffusion, ckpt_path(run_dir, step), use_ema=not args.raw_weights)
    diffusion.to(device).eval()
    n_params = count_params(diffusion.denoise_fn)
    print(f'{run_dir.name} step {step}: {n_params / 1e6:.2f}M params, {which} weights, device {device}')

    root = args.data_root.rstrip('/') + '/'
    common = dict(selected_channels=cfg['selected_channels'], num_frames=cfg['num_frames'],
                  per_frame_cond=cfg['per_frame_cond'], reference_frame=cfg['reference_frame'])
    ds_train = Dataset(root + 'training/', cfg['image_size'], labels_scaling=None, **common)  # normalization only
    ds_val = Dataset(root + 'validation/', cfg['image_size'], labels_scaling=ds_train.labels_scaling, **common)
    s22_min, s22_max = float(ds_train.min_s_22), float(ds_train.max_s_22)
    zero_u2 = float(ds_train.zero_u_2)

    rng = np.random.default_rng(args.seed)
    n_t = min(args.n_targets, len(ds_val))
    idx = np.sort(rng.choice(len(ds_val), n_t, replace=False))
    targets = ds_val.detached_labels[idx]                       # physical stress at the frame strains
    cond_all = ds_val.labels[idx].repeat_interleave(args.n_samples, dim=0)  # normalized, as the model saw in training

    # the authors' model sits in runs/pretrained/pretrained, so name it by hand in the results table
    run_label = 'pretrained' if run_dir.resolve() == Path(paths.PRETRAINED_RUN).resolve() else run_dir.name
    out_dir = run_dir / 'eval_slim' / f'step_{step}'
    out_dir.mkdir(parents=True, exist_ok=True)

    val_loss = float('nan')
    if args.val_batches > 0:
        val_loss = validation_loss(diffusion, ds_val, args.batch_size, args.val_batches,
                                   cfg['null_cond_prob'], args.seed, device)
        print(f'validation loss ({which} weights, {args.val_batches} batches): {val_loss:.5f}')

    summaries, per_sample, curves_out = [], [], {'target_index': idx, 'targets': targets}
    for S in [int(s) for s in args.sampling_steps.split(',') if s.strip()]:
        set_sampling_steps(diffusion, S)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        videos = []
        for b in range(0, len(cond_all), args.batch_size):
            cond = cond_all[b:b + args.batch_size].to(device)
            videos.append(diffusion.sample(cond=cond, guidance_scale=args.guidance).float().cpu())
        if device.type == 'cuda':
            torch.cuda.synchronize()
        sec = (time.time() - t0) / len(cond_all)
        videos = torch.cat(videos).numpy()

        errs = np.zeros((n_t, args.n_samples))
        preds = np.zeros((n_t, args.n_samples, targets.shape[1]))
        empty = 0
        for i in range(n_t):
            for k in range(args.n_samples):
                curve, topo = predicted_stress_curve(videos[i * args.n_samples + k], cfg['selected_channels'],
                                                     s22_min, s22_max, zero_u2)
                empty += int(not topo.any())
                preds[i, k] = curve
                errs[i, k] = nrmse(targets[i], curve)
                per_sample.append(dict(sampling_steps=S, target_index=int(idx[i]), sample=k, nrmse=errs[i, k]))
        curves_out[f'pred_S{S}'] = preds

        row = dict(run=run_label, step=step, params_M=round(n_params / 1e6, 3), dim=cfg['unet_dim'],
                   dim_mults='-'.join(map(str, cfg['unet_dim_mults'])), sampling_steps=S, guidance=args.guidance,
                   n_targets=n_t, n_samples=args.n_samples, nrmse_mean=round(float(np.nanmean(errs)), 5),
                   nrmse_median=round(float(np.nanmedian(errs)), 5),
                   nrmse_best_of_k=round(float(np.nanmean(np.nanmin(errs, axis=1))), 5),
                   empty_topology_frac=round(empty / errs.size, 4), sec_per_sample=round(sec, 3),
                   val_loss_ema=round(val_loss, 6), device=str(device), weights=which)
        summaries.append(row)
        append_row(args.results_csv, row)
        print(f'S={S:>4}  NRMSE mean {row["nrmse_mean"]:.4f}  median {row["nrmse_median"]:.4f}  '
              f'best-of-{args.n_samples} {row["nrmse_best_of_k"]:.4f}  {sec:.2f} s/sample', flush=True)

    with open(out_dir / 'per_sample.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sampling_steps', 'target_index', 'sample', 'nrmse'])
        w.writeheader()
        w.writerows(per_sample)
    (out_dir / 'summary.json').write_text(json.dumps(summaries, indent=2))
    np.savez(out_dir / 'curves.npz', **curves_out)
    print(f'wrote {out_dir} and appended to {args.results_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
