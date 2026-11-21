"""Default locations on Cosmos (same layout as main_modern.py). Override with flags or environment variables:

    SLIM_VAST_DIR    default /cosmos/vast/scratch/dyvo
    SLIM_DATA_ROOT   default $SLIM_VAST_DIR/data/lagrangian            (holds training/ and validation/)
    SLIM_RUNS_DIR    default $SLIM_VAST_DIR/nonlinearmech_runs/slim    (one folder per trained variant)
    SLIM_PRETRAINED  default <repo>/runs/pretrained/pretrained          (authors' 200k-step model, via Git LFS)
"""
import os

from slim import REPO_ROOT

VAST_DIR = os.environ.get('SLIM_VAST_DIR', '/cosmos/vast/scratch/dyvo').rstrip('/')
DATA_ROOT = os.environ.get('SLIM_DATA_ROOT', f'{VAST_DIR}/data/lagrangian')
RUNS_DIR = os.environ.get('SLIM_RUNS_DIR', f'{VAST_DIR}/nonlinearmech_runs/slim')
PRETRAINED_RUN = os.environ.get('SLIM_PRETRAINED', str(REPO_ROOT / 'runs' / 'pretrained' / 'pretrained'))
SLIM_DIR = REPO_ROOT / 'slim_model'
