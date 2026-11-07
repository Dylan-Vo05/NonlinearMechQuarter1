import yaml
import torch
from pathlib import Path

from denoising_diffusion_pytorch import Unet3D, GaussianDiffusion


def main():

    print("=" * 60)
    print("VideoMetamaterials model initialization test")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Verify GPU availability
    # --------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot detect an available GPU.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("\nGPU:", torch.cuda.get_device_name(device))
    print("HIP version:", torch.version.hip)

    # --------------------------------------------------
    # 2. Load model configuration
    # --------------------------------------------------

    config_path = Path("model.yaml")

    if not config_path.exists():
        raise FileNotFoundError("model.yaml was not found.")

    config = yaml.safe_load(config_path.read_text())

    print("\nConfiguration loaded successfully.")
    print("Reference frame:", config["reference_frame"])
    print("Selected channels:", config["selected_channels"])

    # --------------------------------------------------
    # 3. Initialize the 3D U-Net
    # --------------------------------------------------

    print("\nInitializing Unet3D...")

    model = Unet3D(
        dim=config["unet_dim"],
        dim_mults=(1, 2, 4, 8),
        channels=len(config["selected_channels"]),
        attn_heads=config["unet_attn_heads"],
        attn_dim_head=config["unet_attn_dim_head"],
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=config["unet_use_sparse_linear_attn"],
        resnet_groups=config["unet_resnet_groups"],
        cond_bias=True,
        cond_attention=config["unet_cond_attention"],
        cond_attention_tokens=config["unet_cond_attention_tokens"],
        cond_att_GRU=config["unet_cond_att_GRU"],
        use_temporal_attention_cond=config["unet_temporal_att_cond"],
        cond_to_time=config["unet_cond_to_time"],
        per_frame_cond=config["per_frame_cond"],
        padding_mode=config["padding_mode"],
    )

    print("Unet3D initialized successfully.")

    # Count trainable parameters
    num_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {num_params:,}")

    # --------------------------------------------------
    # 4. Initialize Gaussian Diffusion
    # --------------------------------------------------

    print("\nInitializing GaussianDiffusion...")

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

    print("GaussianDiffusion initialized successfully.")

    # --------------------------------------------------
    # 5. Move model to GPU
    # --------------------------------------------------

    print("\nMoving diffusion model to GPU...")

    diffusion = diffusion.to(device)

    torch.cuda.synchronize(device)

    print("Model successfully moved to:", device)

    # --------------------------------------------------
    # 6. Report GPU memory usage
    # --------------------------------------------------

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)

    print(f"\nGPU memory allocated: {allocated:.2f} GB")
    print(f"GPU memory reserved: {reserved:.2f} GB")

    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
