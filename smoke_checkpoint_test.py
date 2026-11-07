
import os
import yaml
import torch

from pathlib import Path
from accelerate import Accelerator

from denoising_diffusion_pytorch import (
    Unet3D,
    GaussianDiffusion,
    Trainer
)



REPO_DIR = Path("/cosmos/nfs/home/dyvo/NonlinearMechQuarter1")

DATA_DIR = Path("/cosmos/vast/scratch/dyvo/data/lagrangian")

TRAIN_DIR = DATA_DIR / "training"
VALIDATION_DIR = DATA_DIR / "validation"

OUTPUT_DIR = Path(
    "/cosmos/vast/scratch/dyvo/nonlinearmech_runs/"
    "lagrangian_smoke_test_3"
)

CONFIG_FILE = REPO_DIR / "model.yaml"

TRAIN_STEPS = 4
BATCH_SIZE = 2



print("\nChecking environment...")

print("PyTorch version:", torch.__version__)
print("ROCm version:", torch.version.hip)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected by PyTorch!")

print("GPU:", torch.cuda.get_device_name(0))

print("GPU smoke test environment check passed.")



print("\nChecking dataset paths...")

for folder in [TRAIN_DIR, VALIDATION_DIR]:

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {folder}"
        )

    if not (folder / "stress_strain_data.csv").is_file():
        raise FileNotFoundError(
            f"Missing stress_strain_data.csv in {folder}"
        )

print("Training directory:", TRAIN_DIR)
print("Validation directory:", VALIDATION_DIR)

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}. "
        "Choose a new output directory before running."
    )

OUTPUT_DIR.mkdir(parents=True)

print("Dataset and output checks passed.")



print("\nLoading model configuration...")

os.chdir(REPO_DIR)

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

print("Selected channels:", config["selected_channels"])
print("Reference frame:", config["reference_frame"])
print("Diffusion timesteps:", config["train_timesteps"])



print("\nInitializing Accelerator...")

accelerator = Accelerator(
    mixed_precision="fp16"
)

print("Device:", accelerator.device)
print("Number of processes:", accelerator.num_processes)



print("\nInitializing U-Net...")

model = Unet3D(
    dim=config["unet_dim"],
    dim_mults=(1, 2, 4, 8),
    channels=len(config["selected_channels"]),
    attn_heads=config["unet_attn_heads"],
    attn_dim_head=config["unet_attn_dim_head"],
    init_dim=None,
    init_kernel_size=7,
    use_sparse_linear_attn=config[
        "unet_use_sparse_linear_attn"
    ],
    resnet_groups=config["unet_resnet_groups"],
    cond_bias=True,
    cond_attention=config["unet_cond_attention"],
    cond_attention_tokens=config[
        "unet_cond_attention_tokens"
    ],
    cond_att_GRU=config["unet_cond_att_GRU"],
    use_temporal_attention_cond=config[
        "unet_temporal_att_cond"
    ],
    cond_to_time=config["unet_cond_to_time"],
    per_frame_cond=config["per_frame_cond"],
    padding_mode=config["padding_mode"],
)

print("U-Net initialized successfully.")



print("\nInitializing diffusion model...")

diffusion = GaussianDiffusion(
    model,
    image_size=96,
    channels=len(config["selected_channels"]),
    num_frames=11,
    timesteps=config["train_timesteps"],
    loss_type="l1",
    use_dynamic_thres=config["use_dynamic_thres"],
    sampling_timesteps=config["sampling_timesteps"],
)

print("Diffusion model initialized successfully.")



print("\nInitializing Trainer...")

trainer = Trainer(
    diffusion,
    folder=str(TRAIN_DIR) + "/",
    validation_folder=str(VALIDATION_DIR) + "/",
    results_folder=str(OUTPUT_DIR),
    selected_channels=config["selected_channels"],

    # Reduced batch size for smoke testing
    train_batch_size=BATCH_SIZE,
    test_batch_size=BATCH_SIZE,

    train_lr=config["learning_rate"],

    # Only five optimizer updates: steps 0 through 4
    train_num_steps=TRAIN_STEPS,

    # Avoid expensive validation sampling
    save_and_sample_every=1000000,

    ema_decay=0.995,

    # Disable external logging
    log=False,

    null_cond_prob=0.1,
    per_frame_cond=config["per_frame_cond"],
    reference_frame=config["reference_frame"],

    run_name="lagrangian_smoke_test_2",
    accelerator=accelerator,
    wandb_username=None
)

print("Trainer initialized successfully.")



print("\nStarting training smoke test...")

trainer.train(
    load_model_step=None,
    num_samples=0,
    num_preds=1
)

print("\nTraining smoke test completed.")



checkpoint = (
    OUTPUT_DIR
    / "model"
    / f"step_{TRAIN_STEPS}"
    / "checkpoint.pt"
)

if not checkpoint.is_file():
    raise RuntimeError(
        f"Training finished but checkpoint is missing: "
        f"{checkpoint}"
    )

print("\n==========================================")
print("SMOKE TEST PASSED")
print("Checkpoint:", checkpoint)
print("==========================================")
