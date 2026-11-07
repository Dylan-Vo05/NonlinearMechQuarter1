
import os
import yaml
import torch
import traceback

from pathlib import Path
from accelerate import Accelerator

from denoising_diffusion_pytorch import (
    Unet3D,
    GaussianDiffusion,
    Trainer
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

REPO_DIR = Path(
    "/cosmos/nfs/home/dyvo/NonlinearMechQuarter1"
)

DATA_DIR = Path(
    "/cosmos/vast/scratch/dyvo/data/lagrangian"
)

TRAIN_DIR = DATA_DIR / "training"
VALIDATION_DIR = DATA_DIR / "validation"

OUTPUT_DIR = Path(
    "/cosmos/vast/scratch/dyvo/nonlinearmech_runs/"
    "lagrangian_smoke_test_4"
)

CONFIG_FILE = REPO_DIR / "model.yaml"

TRAIN_STEPS = 4
BATCH_SIZE = 2

RUN_NAME = "lagrangian_smoke_test_4"


# ==================================================
# 2. MAIN FUNCTION
# ==================================================

def main():

    # --------------------------------------------------
    # 2A. CHECK ENVIRONMENT
    # --------------------------------------------------

    print("\nChecking environment...", flush=True)

    print("PyTorch version:", torch.__version__, flush=True)
    print("ROCm version:", torch.version.hip, flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected by PyTorch!")

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True
    )

    print("GPU environment check passed.", flush=True)


    # --------------------------------------------------
    # 2B. CHECK DATASET PATHS
    # --------------------------------------------------

    print("\nChecking dataset paths...", flush=True)

    for folder in [TRAIN_DIR, VALIDATION_DIR]:

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {folder}"
            )

        required_files = [
            "stress_strain_data.csv",
            "frame_range_data.csv"
        ]

        for filename in required_files:

            if not (folder / filename).is_file():
                raise FileNotFoundError(
                    f"Missing {filename} in {folder}"
                )

    print("Training directory:", TRAIN_DIR, flush=True)
    print("Validation directory:", VALIDATION_DIR, flush=True)


    # --------------------------------------------------
    # 2C. CHECK MODEL CONFIGURATION
    # --------------------------------------------------

    print("\nLoading model configuration...", flush=True)

    os.chdir(REPO_DIR)

    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(CONFIG_FILE)

    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    print("Reference frame:", config["reference_frame"], flush=True)
    print("Selected channels:", config["selected_channels"], flush=True)
    print("Diffusion timesteps:", config["train_timesteps"], flush=True)


    # Check folders actually required by the Dataset class
    required_gif_folders = ["topo", "s_mises", "s_22", "ener"]

    if config["reference_frame"] == "lagrangian":
        required_gif_folders += ["u_1", "u_2"]

    for folder in [TRAIN_DIR, VALIDATION_DIR]:

        for subfolder in required_gif_folders:

            gif_folder = folder / "gifs" / subfolder

            if not gif_folder.is_dir():
                raise FileNotFoundError(
                    f"Missing GIF directory: {gif_folder}"
                )

    print("Dataset structure checks passed.", flush=True)


    # --------------------------------------------------
    # 2D. CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------

    print("\nPreparing output directory...", flush=True)

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True)

    (OUTPUT_DIR / "training").mkdir()
    (OUTPUT_DIR / "model").mkdir()

    print("Output directory:", OUTPUT_DIR, flush=True)


    # --------------------------------------------------
    # 3. INITIALIZE ACCELERATE
    # --------------------------------------------------

    print("\nInitializing Accelerator...", flush=True)

    accelerator = Accelerator(
        mixed_precision="fp16"
    )

    if not hasattr(accelerator, "trackers"):
        accelerator.trackers = []

    print("Device:", accelerator.device, flush=True)
    print(
        "Number of processes:",
        accelerator.num_processes,
        flush=True
    )


    # --------------------------------------------------
    # 4. INITIALIZE U-NET
    # --------------------------------------------------

    print("\nInitializing U-Net...", flush=True)

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

    print("U-Net initialized successfully.", flush=True)


    # --------------------------------------------------
    # 5. INITIALIZE GAUSSIAN DIFFUSION
    # --------------------------------------------------

    print("\nInitializing diffusion model...", flush=True)

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

    print("Diffusion model initialized successfully.", flush=True)


    # --------------------------------------------------
    # 6. INITIALIZE TRAINER
    # --------------------------------------------------

    print("\nInitializing Trainer...", flush=True)

    trainer = Trainer(
        diffusion,
        folder=str(TRAIN_DIR) + "/",
        validation_folder=str(VALIDATION_DIR) + "/",
        results_folder=str(OUTPUT_DIR),
        selected_channels=config["selected_channels"],

        train_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,

        train_lr=config["learning_rate"],
        train_num_steps=TRAIN_STEPS,

        # Prevent periodic validation sampling and Abaqus evaluation
        save_and_sample_every=1000000,

        ema_decay=0.995,

        log=False,

        null_cond_prob=0.1,
        per_frame_cond=config["per_frame_cond"],
        reference_frame=config["reference_frame"],

        run_name=RUN_NAME,
        accelerator=accelerator,
        wandb_username=None
    )

    print("Trainer initialized successfully.", flush=True)


    # --------------------------------------------------
    # 7. TEST VALIDATION DATA LOADING AND FORWARD PASS
    # --------------------------------------------------

    print("\nTesting validation forward pass...", flush=True)

    trainer.model.eval()

    with torch.no_grad():

        val_data, val_cond = next(iter(trainer.dl_test))

        print(
            "Validation data shape:",
            tuple(val_data.shape),
            flush=True
        )

        print(
            "Validation condition shape:",
            tuple(val_cond.shape),
            flush=True
        )

        val_loss = trainer.model(
            x=val_data,
            cond=val_cond,
            null_cond_prob=0.0
        )

        torch.cuda.synchronize()

        if not torch.isfinite(val_loss):
            raise RuntimeError(
                "Validation loss is NaN or infinite!"
            )

    print(
        "Validation loss:",
        val_loss.item(),
        flush=True
    )

    trainer.model.train()

    print("Validation forward pass successful.", flush=True)


    # --------------------------------------------------
    # 8. RUN TRAINING
    # --------------------------------------------------

    print("\nStarting training smoke test...", flush=True)

    trainer.train(
        load_model_step=None,
        num_samples=0,
        num_preds=1
    )

    # Trainer.train() already calls accelerator.end_training().
    # Do not call accelerator.end_training() again here.

    print(
        "\nReturned successfully from trainer.train().",
        flush=True
    )


    # --------------------------------------------------
    # 9. SYNCHRONIZE GPU
    # --------------------------------------------------

    print("\nSynchronizing GPU...", flush=True)

    torch.cuda.synchronize()

    print("GPU synchronization successful.", flush=True)


    # --------------------------------------------------
    # 10. VERIFY CHECKPOINT
    # --------------------------------------------------

    print("\nVerifying checkpoint...", flush=True)

    checkpoint = (
        OUTPUT_DIR
        / "model"
        / f"step_{TRAIN_STEPS}"
        / "checkpoint.pt"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    checkpoint_size = checkpoint.stat().st_size

    if checkpoint_size == 0:
        raise RuntimeError("Checkpoint file is empty!")

    print("Checkpoint:", checkpoint, flush=True)

    print(
        "Checkpoint size:",
        round(checkpoint_size / (1024 * 1024), 2),
        "MB",
        flush=True
    )


    # --------------------------------------------------
    # 11. SUCCESS
    # --------------------------------------------------

    print("\n==========================================", flush=True)
    print("SMOKE TEST PASSED", flush=True)
    print("Training completed successfully.", flush=True)
    print("Validation completed successfully.", flush=True)
    print("Checkpoint saved successfully.", flush=True)
    print("==========================================", flush=True)


# ==================================================
# 12. ERROR HANDLING
# ==================================================

if __name__ == "__main__":

    try:
        main()

    except Exception:

        print("\n========== PYTHON EXCEPTION ==========", flush=True)

        traceback.print_exc()

        print("======================================", flush=True)

        raise
