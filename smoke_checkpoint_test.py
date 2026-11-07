import os
import yaml
import shutil
import datetime
import inspect
from pathlib import Path

import torch
import torch.distributed as dist

from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)

from denoising_diffusion_pytorch import (
    Unet3D,
    GaussianDiffusion,
    Trainer,
)


def main():

    # =========================================================
    # 1. Paths and configuration
    # =========================================================

    run_name = "lagrangian_smoke_test_2"

    scratch = Path("/cosmos/vast/scratch") / os.environ["USER"]

    training_dir = scratch / "data" / "lagrangian" / "training"
    validation_dir = scratch / "data" / "lagrangian" / "validation"

    run_dir = scratch / "nonlinearmech_runs" / run_name

    config_path = Path("model.yaml")

    print("=" * 60)
    print("SMOKE TEST CONFIGURATION")
    print("=" * 60)

    print(f"User:             {os.environ['USER']}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Training data:    {training_dir}")
    print(f"Validation data:  {validation_dir}")
    print(f"Output directory: {run_dir}")
    print(f"Config file:      {config_path}")

    # =========================================================
    # 2. Verify paths
    # =========================================================

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Cannot find configuration file: {config_path.resolve()}"
        )

    if not training_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {training_dir}"
        )

    if not validation_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directory not found: {validation_dir}"
        )

    if run_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {run_dir}\n"
            "Change run_name near the top of this script "
            "before running again."
        )

    # =========================================================
    # 3. Initialize PyTorch distributed environment
    # =========================================================

    # This test uses one process. Supply the environment variables
    # required by PyTorch's env:// rendezvous if they are absent.

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not dist.is_initialized():

        print("\nInitializing Gloo process group...")

        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=datetime.timedelta(seconds=600),
        )

    print("Distributed environment initialized.")
    print(f"Rank: {dist.get_rank()}")
    print(f"World size: {dist.get_world_size()}")

    # =========================================================
    # 4. Initialize Accelerate
    # =========================================================

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True
    )

    process_group_kwargs = InitProcessGroupKwargs(
        timeout=datetime.timedelta(seconds=600)
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[
            ddp_kwargs,
            process_group_kwargs,
        ],
    )

    print("\nAccelerator initialized.")
    print(f"Device: {accelerator.device}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    print(f"Number of processes: {accelerator.num_processes}")

    # =========================================================
    # 5. Verify GPU
    # =========================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot detect a GPU. Check ROCm and "
            "your compute-node allocation."
        )

    print("\nGPU information:")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"ROCm/HIP version: {torch.version.hip}")

    # =========================================================
    # 6. Load model configuration
    # =========================================================

    print("\nLoading model.yaml...")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print("Configuration loaded.")
    print(f"Reference frame: {config['reference_frame']}")
    print(f"Channels: {config['selected_channels']}")
    print(f"Batch size in config: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")

    # =========================================================
    # 7. Create output directories
    # =========================================================

    print("\nCreating output directories...")

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "training").mkdir()
    (run_dir / "model").mkdir()

    shutil.copy2(
        config_path,
        run_dir / "model" / "model.yaml",
    )

    print(f"Created: {run_dir}")

    # =========================================================
    # 8. Initialize U-Net
    # =========================================================

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

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("U-Net initialized.")
    print(f"Number of parameters: {parameter_count:,}")

    # =========================================================
    # 9. Initialize Gaussian diffusion
    # =========================================================

    print("\nInitializing Gaussian diffusion...")

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

    print("Gaussian diffusion initialized.")

    # =========================================================
    # 10. Initialize Trainer
    # =========================================================

    print("\nInitializing Trainer...")

    trainer = Trainer(
        diffusion,
        folder=str(training_dir) + "/",
        validation_folder=str(validation_dir) + "/",
        selected_channels=config["selected_channels"],

        # Training batch size = 1
        train_batch_size=1,

        # IMPORTANT:
        # Trainer internally divides test_batch_size by 2.
        # Therefore, 2 becomes an actual DataLoader batch size of 1.
        test_batch_size=2,

        train_lr=config["learning_rate"],

        # Only two training steps
        train_num_steps=2,

        # Avoid periodic sampling during this short test
        save_and_sample_every=10000,

        ema_decay=0.995,
        log=True,
        null_cond_prob=0.1,
        per_frame_cond=config["per_frame_cond"],
        reference_frame=config["reference_frame"],
        run_name=run_name,
        accelerator=accelerator,
        wandb_username=None,
    )

    print("Trainer initialized.")

    # =========================================================
    # 11. Inspect checkpoint methods
    # =========================================================

    print("\n" + "=" * 60)
    print("CHECKPOINT METHOD INSPECTION")
    print("=" * 60)

    for method_name in ("save", "load", "train"):

        method = getattr(trainer, method_name, None)

        if callable(method):
            print(f"{method_name}{inspect.signature(method)}")
        else:
            print(f"{method_name}: not available")

    # =========================================================
    # 12. Run two training steps
    # =========================================================

    print("\n" + "=" * 60)
    print("STARTING TWO-STEP TRAINING TEST")
    print("=" * 60)

    trainer.train(
        load_model_step=None,
        num_samples=1,
        num_preds=1,
    )

    print("\nTraining call completed.")

    # =========================================================
    # 13. Checkpoint save test
    # =========================================================

    print("\n" + "=" * 60)
    print("CHECKPOINT SAVE TEST")
    print("=" * 60)

    save_method = getattr(trainer, "save", None)

    if callable(save_method):

        save_signature = inspect.signature(save_method)
        print(f"Save method signature: {save_signature}")

        try:
            save_method(0)
            print("trainer.save(0) completed.")
        except Exception as error:
            print(f"Checkpoint save call failed: {error}")

    else:
        print("Trainer does not expose a save() method.")

    # =========================================================
    # 14. Find checkpoint files
    # =========================================================

    print("\nSearching output directory for checkpoints...")

    checkpoint_files = list(run_dir.rglob("*.pt"))

    if checkpoint_files:

        print(f"Found {len(checkpoint_files)} .pt file(s):")

        for checkpoint in checkpoint_files:
            print(f"  {checkpoint}")

    else:
        print("No .pt files found.")

    # =========================================================
    # 15. Checkpoint load test
    # =========================================================

    print("\n" + "=" * 60)
    print("CHECKPOINT LOAD TEST")
    print("=" * 60)

    load_method = getattr(trainer, "load", None)

    if checkpoint_files and callable(load_method):

        print(f"Load method signature: {inspect.signature(load_method)}")

        try:
            load_method(0)
            print("trainer.load(0) completed.")
        except Exception as error:
            print(f"Checkpoint load call failed: {error}")

    elif not checkpoint_files:
        print("Skipping load test: no checkpoint files were found.")

    else:
        print("Skipping load test: Trainer has no load() method.")

    # =========================================================
    # 16. Final status and GPU memory
    # =========================================================

    print("\n" + "=" * 60)
    print("SMOKE TEST FINISHED")
    print("=" * 60)

    print(f"Output directory: {run_dir}")
    print(f"Checkpoint files found: {len(checkpoint_files)}")

    if torch.cuda.is_available():

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory reserved: {reserved:.2f} GB")

    print("\nReview the output above for errors.")


if __name__ == "__main__":
    main()
