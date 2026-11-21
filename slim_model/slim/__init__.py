"""Slim-model training code for this repo: slim configs, a fixed trainer, NRMSE eval.

Importing this package puts the repo root on sys.path, so the scripts in slim_model/ use the repo's own
denoising_diffusion_pytorch/ and src/ (nothing is copied). Every script imports `slim` first.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
