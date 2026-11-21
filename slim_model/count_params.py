"""Print real parameter counts (and optionally a forward-pass check) for every config. No data needed.

    python slim_model/count_params.py                                    # all configs in slim_model/configs/
    python slim_model/count_params.py slim_model/configs/full.yaml --check   # also one forward/backward pass
"""
import slim  # noqa: F401  (first: puts the repo root on sys.path)
import argparse
import glob
import os
import time

import torch

from slim import paths
from slim.common import build_diffusion, count_params, load_config, param_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('configs', nargs='*', default=sorted(glob.glob(str(paths.SLIM_DIR / 'configs' / '*.yaml'))))
    p.add_argument('--check', action='store_true', help='one training-loss forward/backward on random input')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--detail', action='store_true', help='per-level breakdown')
    args = p.parse_args()

    rows = []
    for path in args.configs:
        cfg = load_config(path)
        d = build_diffusion(cfg)
        n = count_params(d.denoise_fn)
        rows.append((os.path.relpath(path), cfg['unet_dim'], tuple(cfg['unet_dim_mults']), n))
        if args.detail:
            print(param_report(d, cfg), '\n')
        if args.check:
            d.to(args.device)
            b = 2
            x = torch.rand(b, len(cfg['selected_channels']), cfg['num_frames'], cfg['image_size'], cfg['image_size'],
                           device=args.device)
            cond = torch.rand(b, cfg['num_frames'], device=args.device) * 2 - 1
            t0 = time.time()
            loss = d(x, cond=cond, null_cond_prob=0.1)
            loss.backward()
            print(f'{path}: forward+backward OK, loss {loss.item():.4f}, {time.time() - t0:.2f} s (batch {b}, {args.device})')
            d.cpu()

    base = next((n for path, *_, n in rows if path.endswith('full.yaml')), None)
    print(f'\n{"config":<44s} {"dim":>4s} {"dim_mults":<14s} {"params":>14s} {"vs full":>8s}')
    for path, dim, mults, n in rows:
        rel = f'{100 * n / base:.0f}%' if base else ''
        print(f'{path:<44s} {dim:>4d} {str(mults):<14s} {n:>14,} {rel:>8s}')


if __name__ == '__main__':
    main()
