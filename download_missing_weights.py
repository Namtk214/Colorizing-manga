#!/usr/bin/env python3
"""
Quick fix script to download missing model weights
"""
import os
import logging
from huggingface_hub import hf_hub_download

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_missing_vae_weights():
    """Download the missing VAE weights"""
    vae_dir = "./checkpoints/StableDiffusion/vae"
    
    # Check if the diffusion_pytorch_model.bin exists
    vae_weights_path = os.path.join(vae_dir, "diffusion_pytorch_model.bin")
    
    if os.path.exists(vae_weights_path):
        logger.info("✓ VAE weights already exist")
        return True
    
    logger.info("Downloading missing VAE weights...")
    
    try:
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            filename="vae/diffusion_pytorch_model.bin",
            local_dir="./checkpoints/StableDiffusion",
            local_dir_use_symlinks=False
        )
        logger.info("✓ VAE weights downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download VAE weights: {e}")
        return False

def download_missing_unet_weights():
    """Download the missing UNet weights"""
    unet_dir = "./checkpoints/StableDiffusion/unet"
    
    # Check if the diffusion_pytorch_model.bin exists
    unet_weights_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")
    
    if os.path.exists(unet_weights_path):
        logger.info("✓ UNet weights already exist")
        return True
    
    logger.info("Downloading missing UNet weights...")
    
    try:
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            filename="unet/diffusion_pytorch_model.bin",
            local_dir="./checkpoints/StableDiffusion",
            local_dir_use_symlinks=False
        )
        logger.info("✓ UNet weights downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download UNet weights: {e}")
        return False

def download_missing_text_encoder_weights():
    """Download the missing text encoder weights"""
    text_encoder_dir = "./checkpoints/StableDiffusion/text_encoder"
    
    # Check if the pytorch_model.bin exists
    text_encoder_weights_path = os.path.join(text_encoder_dir, "pytorch_model.bin")
    
    if os.path.exists(text_encoder_weights_path):
        logger.info("✓ Text encoder weights already exist")
        return True
    
    logger.info("Downloading missing text encoder weights...")
    
    try:
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            filename="text_encoder/pytorch_model.bin",
            local_dir="./checkpoints/StableDiffusion",
            local_dir_use_symlinks=False
        )
        logger.info("✓ Text encoder weights downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download text encoder weights: {e}")
        return False

def main():
    """Download all missing weights"""
    logger.info("Downloading missing model weights...")
    
    tasks = [
        ("VAE weights", download_missing_vae_weights),
        ("UNet weights", download_missing_unet_weights),
        ("Text encoder weights", download_missing_text_encoder_weights),
    ]
    
    success_count = 0
    for task_name, task_func in tasks:
        if task_func():
            success_count += 1
    
    if success_count == len(tasks):
        logger.info("🎉 All missing weights downloaded successfully!")
        return True
    else:
        logger.error("❌ Some downloads failed.")
        return False

if __name__ == "__main__":
    main()