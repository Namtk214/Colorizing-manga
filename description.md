# MangaNinjia Model Setup Guide

This guide explains how to download and set up the required models for MangaNinjia.

## Prerequisites

- Python 3.10+
- Git
- Internet connection for downloading models

## Installation Methods

### Method 1: Automated Download (Recommended)

We've created an automated download script that handles most model downloads for you.

#### Step 1: Install Dependencies

**Option A: Using compatible requirements file (Recommended):**
```bash
pip install -r requirements_compatible.txt
```

**Option B: Manual installation:**
```bash
pip install torch torchvision diffusers==0.27.2 transformers omegaconf gradio pillow opencv-python numpy==1.26.4 accelerate einops huggingface-hub==0.24.6
```

**Note:** Use `diffusers==0.27.2`, `numpy==1.26.4`, and `huggingface-hub==0.24.6` for compatibility.

#### Step 2: Run the Download Script
```bash
python download_models_v2.py
```

This script will:
- ✅ Create the exact directory structure from README.md
- ✅ Download Stable Diffusion v1.5 to `./checkpoints/StableDiffusion`
- ✅ Download CLIP model to `./checkpoints/models/clip-vit-large-patch14`
- ✅ Download ControlNet to `./checkpoints/models/control_v11p_sd15_lineart`
- ✅ Download Annotator models to `./checkpoints/models/Annotators`
- ✅ Download MangaNinjia models to `./checkpoints/MangaNinjia`
- ✅ Update inference.yaml with local paths
- ✅ Verify complete directory structure

#### Step 3: Run the Application
```bash
# For Gradio web interface
python run_gradio.py

# For command-line inference
python infer.py --help
```

### Method 2: Manual Download (Legacy)

If the automated script fails, you can download models manually:

#### Required Models Structure:
```
-- checkpoints
    |-- StableDiffusion          # Stable Diffusion v1.5 base model
    |-- models
        |-- clip-vit-large-patch14       # CLIP vision model
        |-- control_v11p_sd15_lineart    # ControlNet for line art
        |-- Annotators
            |--sk_model.pth              # Line art detection models
            |--sk_model2.pth
    |-- MangaNinjia              # Custom trained models
        |-- denoising_unet.pth
        |-- reference_unet.pth
        |-- point_net.pth
        |-- controlnet.pth
```

#### Manual Download Steps:

1. **Download all models from HuggingFace**:
   ```bash
   # Create directory structure
   mkdir -p checkpoints/StableDiffusion
   mkdir -p checkpoints/models/clip-vit-large-patch14
   mkdir -p checkpoints/models/control_v11p_sd15_lineart
   mkdir -p checkpoints/models/Annotators
   mkdir -p checkpoints/MangaNinjia
   
   # Download using git lfs (requires git-lfs installed)
   cd checkpoints
   
   git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 StableDiffusion
   git clone https://huggingface.co/openai/clip-vit-large-patch14 models/clip-vit-large-patch14
   git clone https://huggingface.co/lllyasviel/control_v11p_sd15_lineart models/control_v11p_sd15_lineart
   git clone https://huggingface.co/Johanan0528/MangaNinjia MangaNinjia
   
   # Download annotator models
   wget -O models/Annotators/sk_model.pth \
     "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth"
   wget -O models/Annotators/sk_model2.pth \
     "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth"
   
   cd ..
   ```

## Configuration

The model paths are configured in `configs/inference.yaml`:

```yaml
model_path:
  pretrained_model_name_or_path: ./checkpoints/StableDiffusion
  clip_vision_encoder_path: ./checkpoints/models/clip-vit-large-patch14
  controlnet_model_name: ./checkpoints/models/control_v11p_sd15_lineart
  annotator_ckpts_path: ./checkpoints/models/Annotators
  manga_control_model_path: ./checkpoints/MangaNinjia/controlnet.pth
  manga_reference_model_path: ./checkpoints/MangaNinjia/reference_unet.pth
  manga_main_model_path: ./checkpoints/MangaNinjia/denoising_unet.pth
  point_net_path: ./checkpoints/MangaNinjia/point_net.pth
```

## Features

### New Features Added:
- 🔄 **Automatic model downloading** - Models download automatically when missing
- 📁 **Multiple reference image upload** - Upload multiple reference images that get concatenated
- 🖼️ **Resolution optimization** - Changed from 512x512 to 256x256 for faster processing
- ✅ **Model validation** - Automatic checking of model availability at startup

### Usage:
1. Start the application: `python run_gradio.py`
2. Upload multiple reference images (they will be automatically concatenated)
3. Upload your target image (line art or RGB image)
4. Click "Process Images" to resize to 256x256
5. (Optional) Click points for correspondence matching
6. Click "Generate" to create the colorized result

## Troubleshooting

### Common Issues:

1. **ImportError: cannot import name 'cached_download' from 'huggingface_hub'**:
   ```bash
   pip install huggingface-hub==0.24.6
   pip install diffusers==0.27.2
   ```

2. **ImportError with diffusers**:
   ```bash
   pip install diffusers==0.27.2
   ```

3. **NumPy compatibility issues**:
   ```bash
   pip install numpy==1.26.4
   ```

3. **CUDA warnings** (Safe to ignore):
   ```
   UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
   ```
   This is normal on systems without CUDA GPUs.

4. **Missing model weights** (Error: no file named diffusion_pytorch_model.bin found):
   ```bash
   python download_missing_weights.py
   ```

5. **Missing MangaNinjia models**:
   - Ensure all 4 `.pth` files are in `checkpoints/MangaNinjia/`
   - These are custom trained models that must be provided separately

6. **Download failures**:
   - Check internet connection
   - Try running `python download_models_v2.py` again
   - Use manual download method as backup

### Verification Commands:

Check if everything is working:
```bash
# Test core imports
python -c "import torch; from diffusers import DDIMScheduler; print('✓ Dependencies OK')"

# Test model downloads
python download_models.py

# Test application startup
python -c "from run_gradio import *; print('✓ Application imports OK')"
```

## Model Sources

- **Stable Diffusion**: [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **CLIP**: [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- **ControlNet**: [lllyasviel/control_v11p_sd15_lineart](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart)
- **Annotators**: [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators)

## Requirements

For the exact dependency versions, see `environment.yaml` or install via:
```bash
pip install -r requirements.txt
```

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Verify all required files exist
3. Ensure compatible package versions
4. Check Python and system requirements
