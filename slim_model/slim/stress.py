"""Stress-strain curve the model itself predicts, read from its generated video (no FE solver).

This follows the authors' pixel estimate in src/utils.py::create_visualization (Lagrangian frame):
  1. topology = pixels whose u_2 is not ~0 in every frame (upper-left quarter, mirrored to the full cell)
  2. sigma_22 channel rescaled to physical units with the training set's global min/max
  3. sigma_22 set to 0 outside the topology
  4. effective stress per frame = -mean(sigma_22) / (1 - strain)
  5. linear shift so the curve passes through 0 at 0 strain (extrapolated from the first two frames)
The only difference: we work on the float tensors (clipped to [0, 1]) instead of re-reading 8-bit gifs.
"""
import numpy as np


def strain_points(num_frames, strain=0.2):
    s = np.linspace(0., strain, num=num_frames)
    s[0] = 0.01 * strain  # first frame is at 1% of the final strain (authors' convention)
    return s


def predicted_stress_curve(video, selected_channels, s22_min, s22_max, zero_u2, strain=0.2, atol=0.02):
    """video: [channels, frames, H, W] in [0, 1] (output of GaussianDiffusion.sample). Returns (curve, topology)."""
    v = np.clip(np.asarray(video, dtype=np.float64), 0., 1.)
    ch = list(selected_channels)
    if 1 not in ch or 3 not in ch:
        raise ValueError('need u_2 (1) and sigma_22 (3) among selected_channels for the Lagrangian estimate')
    u2, s22 = v[ch.index(1)], v[ch.index(3)].copy()
    f, h, w = s22.shape

    quarter = u2[:, :h // 2, :w // 2]
    topo_q = np.logical_not(np.all(np.isclose(quarter, zero_u2, atol=atol), axis=0))
    topo = np.concatenate((topo_q, np.flip(topo_q, axis=0)), axis=0)
    topo = np.concatenate((topo, np.flip(topo, axis=1)), axis=1)

    s22 = s22 * (s22_max - s22_min) + s22_min
    s22[:, ~topo] = 0.

    eps = strain_points(f, strain)
    curve = -s22.reshape(f, -1).mean(axis=1) / (1. - eps)
    shift = curve[0] - ((curve[1] - curve[0]) / (eps[1] - eps[0])) * eps[0]
    return curve - shift, topo


def nrmse(y_true, y_pred):
    """Same definition as src/utils.py::compute_NRMSE."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.sum((y_pred - y_true) ** 2) / np.sum(y_true ** 2)))
