# Optimization and Simplification of Denoising Diffusion Models used in Inverse Design of Nonlinear Metamaterial

**DSC 180A Capstone, Quarter 1 Project** 

**Team:** Dylan Vo, Rohan Raval

**Mentors:** Prof. Igor Sfiligoi, Prof. Nicholas Boechler



## Overview

Mechanical metamaterials are structures designed so that their geometry determines how they behave under some stress. One of the challenges in working with these materials is finding a geometry that produces a specific nonlinear stress-strain curve. There can be multiple geometries that give similar results, and testing each one through simulations takes a lot of time and computational resources.

In a 2023 paper, Bastek and Kochmann approach this problem using a video denoising diffusion model. The model takes a stress-strain curve as input and generates a 2D unit-cell geometry along with a sequence of frames showing how the structure deforms under compression, like a video or a GIF.

**Quarter 1 goal:** Our main goal this quarter is to reproduce the framework presented in the paper, and additionally train a slimmer model which produces similar quality output. We plan to set up the required environment, prepare the data, run the pretrained model provided by the authors, and make sure the training pipeline can run from start to finish.


## Based On

This repository was forked from **[jhbastek/VideoMetamaterials](https://github.com/jhbastek/VideoMetamaterials)**, the official code for:

> J.-H. Bastek and D. M. Kochmann, "Inverse design of nonlinear mechanical metamaterials via video denoising diffusion models," *Nature Machine Intelligence*, 2023. [doi:10.1038/s42256-023-00762-x](https://www.nature.com/articles/s42256-023-00762-x)

Their implementation builds on Phil Wang's [video diffusion code](https://github.com/lucidrains/imagen-pytorch). The model, training loop and evaluation code (`src/`, `denoising_diffusion_pytorch/`, `abaqus/`) are the original authors' work, used under their MIT license. The GIFs in this repository (`pred_light.gif`, `pred_dark.gif`) are also the authors' results, not ours.

## Our Changes

- **`main.py`:** set `run_name = 'smoke_test_2'` to train a new model instead of loading the pretrained one.
- **`main.py`:** reduced training from 200,000 to 10,000 steps, and sampling from every 10,000 to every 1,000 steps, for a quick test run.
- **`requirements.txt`:** created it to help download the Python packages the code needs.
- **`runs/pretrained/`:** added the authors' pretrained checkpoint (200k steps, via Git LFS).


## Repository Structure

```
.
├── main.py                       # Entry point: train and/or sample 
├── model.yaml                    # Model and training hyperparameters
├── requirements.txt              # Python dependencies
├── src/                          # Data loading, normalization, utilities
├── denoising_diffusion_pytorch/  # 3D U-Net and Gaussian diffusion model
├── eval_abaqus.py, abaqus/       # FEM check of generated designs
├── gif_visualization.py          # Turns raw output GIFs into paper style figures
├── data/
│   ├── target_responses.csv      # Target stress–strain curves (one per row, 11 strain points)
│   └── lagrangian/               # Training/validation data (download separately, see below)
└── runs/
    └── pretrained/               # Authors' pretrained checkpoint (Git LFS), our own runs are stored on Cosmos
```

## Setup

1. **Clone** (requires Git LFS for the pretrained checkpoint):
   ```bash
   git lfs install
   git clone https://github.com/Dylan-Vo05/NonlinearMechQuarter1.git
   cd NonlinearMechQuarter1
   ```
   > **Note:** ZIP downloads (including our Gradescope submission) do not include the pretrained checkpoint. `runs/pretrained/pretrained/model/step_200000/checkpoint.pt` is only a small Git LFS pointer file. To get the full ~620 MB model, clone with Git LFS as shown above, or download `pretrained.zip` from the [ETHZ Research Collection](https://doi.org/10.3929/ethz-b-000629716).

2. **Install dependencies** (Python 3.11, NVIDIA GPU with CUDA):
   ```bash
   pip install -r requirements.txt
   ```

3. **Get the data:** download `lagrangian.zip` from the [ETHZ Research Collection](https://doi.org/10.3929/ethz-b-000629716) and unzip it into your data folder, along with `data/target_responses.csv`.

4. **Set your path:** in `main.py`, change `vast_dir = '/cosmos/vast/scratch/dyvo/'` to your own folder. Outputs go to `<vast_dir>/nonlinearmech_runs/<run_name>/`.

## Running

```bash
accelerate config        # needed the first time, to setup GPUs
accelerate launch main.py
```

- **Default:** trains a new model for 10,000 steps (`run_name = 'smoke_test_2'`). Use a new `run_name` for each run, or the script exits without training.
- **Pretrained model:** set `run_name = 'pretrained'` to generate designs from the authors' checkpoint.



## AI Contribution Statement

None of our work in this repository on in the checkpoint was written or coded the assistance of any AI tools. AI was solely used to help us understand how the code uses GPU and setting it up on Cosmos cluster. All changes to this repo was made on top of the original authors' work. 
