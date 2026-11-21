"""Shared helpers for the slim-model scripts: config, model building, checkpoints, parameter counts."""
import os
import re
from pathlib import Path

import torch
import yaml

from denoising_diffusion_pytorch.video_denoising_diffusion_pytorch import Unet3D, GaussianDiffusion

# Keys the authors' model.yaml does not have. Defaults reproduce their main.py exactly,
# so the released runs/pretrained/model/model.yaml loads unchanged.
DEFAULTS = {
    'unet_dim_mults': [1, 2, 4, 8],
    'image_size': 96,
    'num_frames': 11,
    'train_num_steps': 200000,
    'save_and_sample_every': 10000,
    'null_cond_prob': 0.1,
    'ema_decay': 0.995,
}


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for k, v in DEFAULTS.items():
        cfg.setdefault(k, v)
    cfg['unet_dim_mults'] = [int(m) for m in cfg['unet_dim_mults']]
    return cfg


def build_unet(cfg):
    return Unet3D(
        dim=cfg['unet_dim'],
        dim_mults=tuple(cfg['unet_dim_mults']),
        channels=len(cfg['selected_channels']),
        attn_heads=cfg['unet_attn_heads'],
        attn_dim_head=cfg['unet_attn_dim_head'],
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=cfg['unet_use_sparse_linear_attn'],
        resnet_groups=cfg['unet_resnet_groups'],
        cond_bias=True,
        cond_attention=cfg['unet_cond_attention'],
        cond_attention_tokens=cfg['unet_cond_attention_tokens'],
        cond_att_GRU=cfg['unet_cond_att_GRU'],
        use_temporal_attention_cond=cfg['unet_temporal_att_cond'],
        cond_to_time=cfg['unet_cond_to_time'],
        per_frame_cond=cfg['per_frame_cond'],
        padding_mode=cfg['padding_mode'],
    )


def build_diffusion(cfg, cls=GaussianDiffusion, **extra):
    """Same construction as the authors' main.py, with dim_mults/image size taken from the config."""
    return cls(
        build_unet(cfg),
        image_size=cfg['image_size'],
        channels=len(cfg['selected_channels']),
        num_frames=cfg['num_frames'],
        timesteps=cfg['train_timesteps'],
        loss_type='l1',
        use_dynamic_thres=cfg['use_dynamic_thres'],
        sampling_timesteps=cfg['sampling_timesteps'],
        **extra,
    )


def set_sampling_steps(diffusion, n):
    """n == train_timesteps -> the authors' ancestral sampler; n < train_timesteps -> DDIM (eta = 0)."""
    n = int(n)
    assert 1 <= n <= diffusion.num_timesteps, f'sampling steps must be in [1, {diffusion.num_timesteps}]'
    diffusion.sampling_timesteps = n
    diffusion.is_ddim_sampling = n < diffusion.num_timesteps


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def param_breakdown(diffusion):
    """Parameter count of the denoiser, split by U-Net level."""
    unet = diffusion.denoise_fn
    rows = []
    for name, child in unet.named_children():
        if isinstance(child, torch.nn.ModuleList):
            for i, sub in enumerate(child):
                rows.append((f'{name}[{i}]', count_params(sub)))
        else:
            rows.append((name, count_params(child)))
    own = sum(p.numel() for p in unet.parameters(recurse=False))
    if own:
        rows.append(('unet (direct params)', own))
    return rows


def param_report(diffusion, cfg=None):
    total = count_params(diffusion.denoise_fn)
    lines = []
    if cfg is not None:
        lines.append(f"config: dim={cfg['unet_dim']} dim_mults={tuple(cfg['unet_dim_mults'])}")
    lines.append(f'denoiser parameters: {total:,} ({total / 1e6:.2f}M)')
    for name, n in param_breakdown(diffusion):
        if n:
            lines.append(f'  {name:<24s} {n:>12,}  ({100 * n / total:5.1f}%)')
    return '\n'.join(lines)


def ckpt_path(run_dir, step):
    return Path(run_dir) / 'model' / f'step_{int(step)}' / 'checkpoint.pt'


def latest_step(run_dir):
    steps = []
    for p in (Path(run_dir) / 'model').glob('step_*/checkpoint.pt'):
        m = re.match(r'step_(\d+)$', p.parent.name)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def clean_state_dict(sd):
    """Strip DDP/compile prefixes so a checkpoint from any GPU count loads anywhere."""
    out = {}
    for k, v in sd.items():
        for prefix in ('module.', '_orig_mod.'):
            while k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def torch_load(path):
    # weights_only=False because the checkpoints are our own (or the authors') files
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location='cpu')


def load_weights(diffusion, path, use_ema=True):
    """Load EMA weights (falls back to raw weights). Raises on any key mismatch."""
    obj = torch_load(path)
    key = 'ema' if (use_ema and 'ema' in obj) else 'model'
    diffusion.load_state_dict(clean_state_dict(obj[key]), strict=True)
    return obj.get('steps'), key


def resolve_step(run_dir, step):
    if step in (None, 'latest'):
        s = latest_step(run_dir)
        if s is None:
            raise FileNotFoundError(f'no checkpoints under {run_dir}/model/step_*/checkpoint.pt')
        return s
    return int(step)


def is_multi_process():
    return int(os.environ.get('WORLD_SIZE', '1')) > 1
