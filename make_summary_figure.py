import os
import sys
import shutil
import numpy as np
import imageio
from PIL import Image, ImageSequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import create_visualization, compute_NRMSE

'''
Builds the summary figure for an eval run: for each target, the coloured sigma_22 field at 0.2% and 20% strain
(Eulerian frame, jet colormap, grey box = compression) and the model's estimated stress-strain curve vs. the target.
Also writes one coloured deformation gif per target.

usage: python make_summary_figure.py <gif dir> <frame_range_data.csv> [out dir]
'''

def main():
    gif_dir = sys.argv[1].rstrip('/') + '/'
    path_frames_full = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(gif_dir, '..', 'results')
    os.makedirs(out_dir, exist_ok=True)

    targets = np.loadtxt('./data/target_responses.csv', delimiter=',')
    num_targets = imageio.get_reader(gif_dir + 'prediction_channel_3.gif').get_data(0).shape[0] // 100

    fig, axes = plt.subplots(num_targets, 3, figsize=(10.8, 3.6 * num_targets))
    nrmse = []
    for row in range(num_targets):
        create_visualization(gif_dir, path_frames_full, row, 0, ref_frame='lagrangian', atol=0.02)
        vis_dir = gif_dir + 'visualization_' + str(row) + '-0/'
        frames = [np.array(f.convert('RGBA')) for f in ImageSequence.Iterator(Image.open(vis_dir + 'visualization_conv_euler.gif'))]
        shutil.copy(vis_dir + 'visualization_conv_euler.gif', os.path.join(out_dir, 'target_' + str(row) + '_deformation.gif'))

        pred = np.loadtxt(vis_dir + 'stress_strain_estimate.csv', delimiter=',', skiprows=1)
        strain, curve = pred[:, 0], pred[:, 1]
        err = compute_NRMSE(targets[row], curve)
        nrmse.append(err)

        for ax, f, label in ((axes[row, 0], 0, 'undeformed (0.2% strain)'), (axes[row, 1], -1, 'compressed (20% strain)')):
            ax.imshow(frames[f], interpolation='nearest')
            ax.set_title('Target ' + str(row) + ': σ22, ' + label)
            ax.axis('off')

        ax = axes[row, 2]
        ax.plot(strain, targets[row], 'k-o', label='target')
        ax.plot(strain, curve, 'r--s', label='model estimate')
        ax.set_title('NRMSE ' + f'{100 * err:.1f}%')
        ax.set_xlabel('strain')
        ax.set_ylabel('stress')
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'summary.png'), dpi=150)
    np.savetxt(os.path.join(out_dir, 'nrmse.csv'), np.array(nrmse)[:, None], delimiter=',', header='nrmse', comments='')
    print('NRMSE per target:', ', '.join(f'{100 * e:.1f}%' for e in nrmse))

if __name__ == '__main__':
    main()
