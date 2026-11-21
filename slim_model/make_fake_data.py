"""Write a tiny random dataset in the authors' folder layout, for a smoke test of the pipeline only.

    python slim_model/make_fake_data.py --out slim_model/_smoke/data --n_train 8 --n_val 4
"""
import argparse
import os

import numpy as np
from PIL import Image

FIELDS = ['topo', 'u_1', 'u_2', 's_mises', 's_22', 'ener']


def write_gif(path, frames):
    imgs = [Image.fromarray(f, mode='L') for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=200, loop=0, optimize=False)


def make_split(root, n, rng, frames=11, px=96, n_stress=51):
    for fld in FIELDS:
        os.makedirs(os.path.join(root, 'gifs', fld), exist_ok=True)
    ranges, stress = [], []
    for i in range(n):
        # mirror-symmetric random topology so the u_2-based topology extraction has something to find
        q = (rng.random((px // 2, px // 2)) > 0.4).astype(np.uint8)
        topo = np.block([[q, q[:, ::-1]], [q[::-1], q[::-1, ::-1]]]) * 255
        for fld in FIELDS:
            if fld == 'topo':
                vids = topo[None].repeat(frames, 0).astype(np.uint8)
                vids[np.arange(frames), 0, np.arange(frames)] ^= 255  # one pixel differs per frame, else Pillow merges identical frames
            else:
                vids = (rng.random((frames, px, px)) * 255).astype(np.uint8)
            write_gif(os.path.join(root, 'gifs', fld, f'{i}.gif'), vids)
        ranges.append([-0.1, 0.1, -0.2, 0.0, 50., -40., 5., 1.])  # min_u1,max_u1,min_u2,max_u2,max_vM,min_s22,max_s22,max_E
        stress.append(np.cumsum(rng.random(n_stress)) * rng.uniform(0.5, 2.0))
    np.savetxt(os.path.join(root, 'frame_range_data.csv'), np.array(ranges), delimiter=',')
    np.savetxt(os.path.join(root, 'stress_strain_data.csv'), np.array(stress), delimiter=',')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='slim_model/_smoke/data')
    p.add_argument('--n_train', type=int, default=8)
    p.add_argument('--n_val', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    make_split(os.path.join(a.out, 'training'), a.n_train, rng)
    make_split(os.path.join(a.out, 'validation'), a.n_val, rng)
    print(f'fake data written to {a.out}/training and {a.out}/validation')


if __name__ == '__main__':
    main()
