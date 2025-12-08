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
```bash
pip install torch torchvision diffusers==0.27.2 transformers omegaconf gradio pillow opencv-python numpy==1.26.4 accelerate einops huggingface-hub
```

**Note:** Use `diffusers==0.27.2` and `numpy==1.26.4` for compatibility.

#### Step 2: Run the Download Script
```bash
python download_models.py
```

This script will:
- ✅ Check if MangaNinjia custom models exist
- ✅ Download annotator models from HuggingFace
- ✅ Verify access to HuggingFace models
- ✅ Display detailed status for each component

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
checkpoints/
├── MangaNinjia/                 # Custom trained models (required)
│   ├── controlnet.pth
│   ├── denoising_unet.pth
│   ├── point_net.pth
│   └── reference_unet.pth
└── models/
    └── Annotators/              # Line art detection models
        ├── sk_model.pth
        └── sk_model2.pth
```

#### Manual Download Steps:

1. **MangaNinjia Custom Models** (Must be provided separately):
   ```bash
   # Ensure these files exist in checkpoints/MangaNinjia/
   ls checkpoints/MangaNinjia/
   # Should show: controlnet.pth, denoising_unet.pth, point_net.pth, reference_unet.pth
   ```

2. **Annotator Models**:
   ```bash
   mkdir -p checkpoints/models/Annotators
   
   # Download from HuggingFace
   wget -O checkpoints/models/Annotators/sk_model.pth \
     "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth"
   
   wget -O checkpoints/models/Annotators/sk_model2.pth \
     "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth"
   ```

3. **HuggingFace Models** (Downloaded automatically when needed):
   - `runwayml/stable-diffusion-v1-5`
   - `openai/clip-vit-large-patch14`
   - `lllyasviel/control_v11p_sd15_lineart`

## Configuration

The model paths are configured in `configs/inference.yaml`:

```yaml
model_path:
  pretrained_model_name_or_path: runwayml/stable-diffusion-v1-5
  clip_vision_encoder_path: openai/clip-vit-large-patch14
  controlnet_model_name: lllyasviel/control_v11p_sd15_lineart
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

1. **ImportError with diffusers**:
   ```bash
   pip install diffusers==0.27.2
   ```

2. **NumPy compatibility issues**:
   ```bash
   pip install numpy==1.26.4
   ```

3. **CUDA warnings** (Safe to ignore):
   ```
   UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
   ```
   This is normal on systems without CUDA GPUs.

4. **Missing MangaNinjia models**:
   - Ensure all 4 `.pth` files are in `checkpoints/MangaNinjia/`
   - These are custom trained models that must be provided separately

5. **Download failures**:
   - Check internet connection
   - Try running `python download_models.py` again
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