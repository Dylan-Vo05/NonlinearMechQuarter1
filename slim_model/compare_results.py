"""Side-by-side table of every evaluated run, plus validation-loss curves from training.

    python slim_model/compare_results.py                      # reads <runs>/ablation_results.csv and <runs>/*/metrics.csv
    python slim_model/compare_results.py --plot val_loss.png  # also plot validation loss vs step (needs matplotlib)
"""
import slim  # noqa: F401
import argparse
import csv
import glob
import os
from collections import defaultdict

from slim import paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs_dir', default=paths.RUNS_DIR)
    p.add_argument('--results_csv', default=None, help='default: <runs_dir>/ablation_results.csv')
    p.add_argument('--plot', default=None)
    args = p.parse_args()
    if args.results_csv is None:
        args.results_csv = os.path.join(args.runs_dir, 'ablation_results.csv')

    if os.path.exists(args.results_csv):
        rows = list(csv.DictReader(open(args.results_csv)))
        latest = {}
        for r in rows:  # keep the newest row per (run, step, sampler steps)
            latest[(r['run'], r['step'], r['sampling_steps'])] = r
        rows = sorted(latest.values(), key=lambda r: (r['run'], int(r['step']), -int(r['sampling_steps'])))
        cols = ['run', 'step', 'params_M', 'sampling_steps', 'nrmse_mean', 'nrmse_best_of_k', 'sec_per_sample',
                'val_loss_ema', 'empty_topology_frac']
        widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
        print('  '.join(c.ljust(w) for c, w in zip(cols, widths)))
        for r in rows:
            print('  '.join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    else:
        print(f'no {args.results_csv} yet (run eval_slim.py first)')

    curves = defaultdict(list)
    for path in glob.glob(os.path.join(args.runs_dir, '*', 'metrics.csv')):
        run = os.path.basename(os.path.dirname(path))
        for r in csv.DictReader(open(path)):
            if r['metric'] == 'val_loss':
                curves[run].append((int(r['step']), float(r['value'])))
    if curves:
        print('\nvalidation loss during training (raw weights, step: loss)')
        for run, pts in sorted(curves.items()):
            pts = sorted(dict(pts).items())
            print(f'  {run}: ' + ', '.join(f'{s}: {v:.4f}' for s, v in pts))
    if args.plot and curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for run, pts in sorted(curves.items()):
            pts = sorted(dict(pts).items())
            ax.plot([s for s, _ in pts], [v for _, v in pts], marker='o', label=run)
        ax.set_xlabel('training step')
        ax.set_ylabel('validation loss (L1 on noise)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f'plot saved to {args.plot}')


if __name__ == '__main__':
    main()
