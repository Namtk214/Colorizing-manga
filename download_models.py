#!/usr/bin/env python3
"""
Download script for required models in MangaNinjia
"""
import os
import sys
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def download_annotator_models(annotator_path):
    """Download annotator models from HuggingFace"""
    ensure_dir(annotator_path)
    
    models_to_download = [
        "sk_model.pth",
        "sk_model2.pth"
    ]
    
    for model_name in models_to_download:
        model_path = os.path.join(annotator_path, model_name)
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model_name}...")
            try:
                hf_hub_download(
                    repo_id="lllyasviel/Annotators",
                    filename=model_name,
                    local_dir=annotator_path,
                    local_dir_use_symlinks=False
                )
                logger.info(f"✓ Downloaded {model_name}")
            except Exception as e:
                logger.error(f"✗ Failed to download {model_name}: {e}")
        else:
            logger.info(f"✓ {model_name} already exists")

def check_huggingface_models():
    """Check if HuggingFace models are accessible"""
    models_to_check = [
        "runwayml/stable-diffusion-v1-5",
        "openai/clip-vit-large-patch14", 
        "lllyasviel/control_v11p_sd15_lineart"
    ]
    
    for model_id in models_to_check:
        logger.info(f"Checking access to {model_id}...")
        try:
            # Try to download just the config to test access
            from transformers import AutoConfig
            AutoConfig.from_pretrained(model_id)
            logger.info(f"✓ {model_id} is accessible")
        except Exception as e:
            logger.warning(f"⚠ {model_id} may not be fully accessible: {e}")

def download_all_models():
    """Download all required models"""
    logger.info("Starting model download process...")
    
    # Load config
    config_path = "./configs/inference.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    config = OmegaConf.load(config_path)
    
    # Download annotator models
    annotator_path = config.model_path.annotator_ckpts_path
    download_annotator_models(annotator_path)
    
    # Check HuggingFace models
    check_huggingface_models()
    
    logger.info("Model download process completed!")
    return True

def check_manga_models():
    """Check if MangaNinjia custom models exist"""
    config = OmegaConf.load("./configs/inference.yaml")
    
    manga_models = [
        config.model_path.manga_control_model_path,
        config.model_path.manga_reference_model_path,
        config.model_path.manga_main_model_path,
        config.model_path.point_net_path
    ]
    
    all_exist = True
    for model_path in manga_models:
        if os.path.exists(model_path):
            logger.info(f"✓ {model_path} exists")
        else:
            logger.error(f"✗ {model_path} is missing")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    try:
        # Check MangaNinjia models first
        logger.info("Checking MangaNinjia custom models...")
        manga_ok = check_manga_models()
        
        if not manga_ok:
            logger.error("Missing MangaNinjia custom models. Please ensure all .pth files are in checkpoints/MangaNinjia/")
            sys.exit(1)
        
        # Download other required models
        success = download_all_models()
        
        if success:
            logger.info("All models are ready!")
            sys.exit(0)
        else:
            logger.error("Some models could not be downloaded")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)