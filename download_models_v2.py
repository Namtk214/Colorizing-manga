#!/usr/bin/env python3
"""
Download script for required models in MangaNinja
Creates the exact folder structure as specified in README.md:

-- checkpoints
    |-- StableDiffusion
    |-- models
        |-- clip-vit-large-patch14
        |-- control_v11p_sd15_lineart
        |-- Annotators
            |--sk_model.pth
    |-- MangaNinjia
        |-- denoising_unet.pth
        |-- reference_unet.pth
        |-- point_net.pth
        |-- controlnet.pth
"""
import os
import sys
import logging
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def download_stable_diffusion():
    """Download Stable Diffusion v1.5 model"""
    model_path = "./checkpoints/StableDiffusion"
    
    # Check if all required files exist
    required_files = [
        "vae/diffusion_pytorch_model.bin",
        "unet/diffusion_pytorch_model.bin", 
        "text_encoder/pytorch_model.bin",
        "model_index.json",
        "scheduler/scheduler_config.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(model_path, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if not missing_files:
        logger.info(f"✓ StableDiffusion already complete at {model_path}")
        return True
    
    ensure_dir(model_path)
    logger.info("Downloading Stable Diffusion v1.5...")
    logger.info(f"Missing files: {missing_files}")
    
    try:
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            local_dir=model_path,
            # Don't ignore any patterns to ensure all files are downloaded
        )
        
        # Verify all required files are now present
        still_missing = []
        for file_path in required_files:
            full_path = os.path.join(model_path, file_path)
            if not os.path.exists(full_path):
                still_missing.append(file_path)
        
        if still_missing:
            logger.warning(f"Some files are still missing: {still_missing}")
            # Try to download missing files individually
            for file_path in still_missing:
                try:
                    hf_hub_download(
                        repo_id="runwayml/stable-diffusion-v1-5",
                        filename=file_path,
                        local_dir=model_path,
                        local_dir_use_symlinks=False
                    )
                    logger.info(f"✓ Downloaded {file_path}")
                except Exception as e:
                    logger.error(f"✗ Failed to download {file_path}: {e}")
        
        logger.info("✓ Stable Diffusion downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download Stable Diffusion: {e}")
        return False

def download_clip_model():
    """Download CLIP ViT Large model"""
    model_path = "./checkpoints/models/clip-vit-large-patch14"
    
    if os.path.exists(model_path) and os.listdir(model_path):
        logger.info(f"✓ CLIP model already exists at {model_path}")
        return True
    
    ensure_dir(model_path)
    logger.info("Downloading CLIP ViT-Large-Patch14...")
    
    try:
        snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=model_path
        )
        logger.info("✓ CLIP model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download CLIP model: {e}")
        return False

def download_controlnet():
    """Download ControlNet lineart model"""
    model_path = "./checkpoints/models/control_v11p_sd15_lineart"
    
    if os.path.exists(model_path) and os.listdir(model_path):
        logger.info(f"✓ ControlNet already exists at {model_path}")
        return True
    
    ensure_dir(model_path)
    logger.info("Downloading ControlNet lineart model...")
    
    try:
        snapshot_download(
            repo_id="lllyasviel/control_v11p_sd15_lineart",
            local_dir=model_path
        )
        logger.info("✓ ControlNet downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download ControlNet: {e}")
        return False

def download_annotator_models():
    """Download annotator models from HuggingFace"""
    annotator_path = "./checkpoints/models/Annotators"
    ensure_dir(annotator_path)
    
    models_to_download = [
        "sk_model.pth",
        "sk_model2.pth"
    ]
    
    success_count = 0
    for model_name in models_to_download:
        model_path = os.path.join(annotator_path, model_name)
        if os.path.exists(model_path):
            logger.info(f"✓ {model_name} already exists")
            success_count += 1
            continue
            
        logger.info(f"Downloading {model_name}...")
        try:
            hf_hub_download(
                repo_id="lllyasviel/Annotators",
                filename=model_name,
                local_dir=annotator_path,
                local_dir_use_symlinks=False
            )
            logger.info(f"✓ Downloaded {model_name}")
            success_count += 1
        except Exception as e:
            logger.error(f"✗ Failed to download {model_name}: {e}")
    
    return success_count == len(models_to_download)

def download_manganinja_models():
    """Download MangaNinja custom models"""
    model_path = "./checkpoints/MangaNinjia"
    
    # Check if models already exist
    required_files = [
        "denoising_unet.pth",
        "reference_unet.pth", 
        "point_net.pth",
        "controlnet.pth"
    ]
    
    existing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            existing_files.append(file)
    
    if len(existing_files) == len(required_files):
        logger.info("✓ All MangaNinja models already exist")
        return True
    
    ensure_dir(model_path)
    logger.info("Downloading MangaNinja custom models...")
    
    try:
        snapshot_download(
            repo_id="Johanan0528/MangaNinjia",
            local_dir=model_path
        )
        
        # Verify all required files are present
        missing_files = []
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"Some files are missing: {missing_files}")
            return False
        
        logger.info("✓ MangaNinja models downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download MangaNinja models: {e}")
        logger.info("Please download the models manually from https://huggingface.co/Johanan0528/MangaNinjia")
        return False

def update_config_paths():
    """Update inference.yaml to use local paths"""
    config_path = "./configs/inference.yaml"
    
    try:
        config = OmegaConf.load(config_path)
        
        # Update paths to local directories
        config.model_path.pretrained_model_name_or_path = "./checkpoints/StableDiffusion"
        config.model_path.clip_vision_encoder_path = "./checkpoints/models/clip-vit-large-patch14"
        config.model_path.controlnet_model_name = "./checkpoints/models/control_v11p_sd15_lineart"
        config.model_path.annotator_ckpts_path = "./checkpoints/models/Annotators"
        
        # Save updated config
        OmegaConf.save(config, config_path)
        logger.info("✓ Updated inference.yaml with local paths")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to update config: {e}")
        return False

def verify_directory_structure():
    """Verify the downloaded directory structure matches README requirements"""
    logger.info("Verifying directory structure...")
    
    # Check critical model weight files
    critical_files = {
        "./checkpoints/StableDiffusion/vae/diffusion_pytorch_model.bin": "Stable Diffusion VAE weights",
        "./checkpoints/StableDiffusion/unet/diffusion_pytorch_model.bin": "Stable Diffusion UNet weights",
        "./checkpoints/StableDiffusion/text_encoder/pytorch_model.bin": "Stable Diffusion text encoder weights",
        "./checkpoints/models/Annotators/sk_model.pth": "Annotator model",
        "./checkpoints/MangaNinjia/denoising_unet.pth": "MangaNinjia denoising UNet",
        "./checkpoints/MangaNinjia/reference_unet.pth": "MangaNinjia reference UNet",
        "./checkpoints/MangaNinjia/point_net.pth": "MangaNinjia point network",
        "./checkpoints/MangaNinjia/controlnet.pth": "MangaNinjia ControlNet"
    }
    
    # Check directories with contents
    directories_with_contents = {
        "./checkpoints/StableDiffusion": "Stable Diffusion base model",
        "./checkpoints/models/clip-vit-large-patch14": "CLIP vision model", 
        "./checkpoints/models/control_v11p_sd15_lineart": "ControlNet lineart model"
    }
    
    all_good = True
    
    # Check critical files
    logger.info("Checking critical model files...")
    for file_path, description in critical_files.items():
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) // (1024*1024)  # Size in MB
            logger.info(f"✓ {description}: {file_path} ({file_size} MB)")
        else:
            logger.error(f"✗ {description}: {file_path} (missing)")
            all_good = False
    
    # Check directories
    logger.info("Checking model directories...")
    for dir_path, description in directories_with_contents.items():
        if os.path.isdir(dir_path) and os.listdir(dir_path):
            file_count = len(os.listdir(dir_path))
            logger.info(f"✓ {description}: {dir_path} ({file_count} files)")
        else:
            logger.error(f"✗ {description}: {dir_path} (missing or empty)")
            all_good = False
    
    return all_good

def main():
    """Main download function"""
    logger.info("Starting MangaNinja model download process...")
    logger.info("This will create the directory structure specified in README.md")
    
    # Download all components
    tasks = [
        ("Stable Diffusion", download_stable_diffusion),
        ("CLIP Model", download_clip_model),
        ("ControlNet", download_controlnet),
        ("Annotator Models", download_annotator_models),
        ("MangaNinja Models", download_manganinja_models),
    ]
    
    results = {}
    for task_name, task_func in tasks:
        logger.info(f"\n--- Downloading {task_name} ---")
        results[task_name] = task_func()
    
    # Update config file
    logger.info("\n--- Updating Configuration ---")
    results["Config Update"] = update_config_paths()
    
    # Verify structure
    logger.info("\n--- Verifying Structure ---")
    structure_ok = verify_directory_structure()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*50)
    
    for task_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{task_name}: {status}")
    
    if structure_ok and all(results.values()):
        logger.info("\n🎉 All models downloaded successfully!")
        logger.info("You can now run: python run_gradio.py")
        return True
    else:
        logger.error("\n❌ Some downloads failed. Please check the logs above.")
        logger.info("You may need to download some models manually.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)