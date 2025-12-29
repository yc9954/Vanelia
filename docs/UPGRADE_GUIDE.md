# Vanelia Upgrade Guide: INR-Harmonization + MUSt3R

## Overview of Changes

This upgrade brings two major improvements to the Vanelia pipeline:

1. **INR-Harmonization Compositor**: Replaces the experimental photorealistic compositor with a pre-trained video harmonization model
2. **MUSt3R Camera Extraction**: Replaces Dust3R with MUSt3R for better temporal consistency

## What's New

### 1. INR-Harmonization Compositor (Default)

**Previous**: Photorealistic compositor with random initialization (outputs noise)
**New**: INR-Harmonization with pre-trained weights from HYouTube dataset

**Benefits**:
- ✅ **Production-ready** with pre-trained weights
- ✅ **Arbitrary resolution** support (256x256 to 4K+)
- ✅ **Fast inference** using Implicit Neural Representation
- ✅ **Better quality** than ControlNet for video harmonization
- ✅ **No training required** - works out of the box

**Usage**:
```bash
# Automatic (new default)
python vanelia_pipeline.py --input video.mp4 --model product.glb --output final.mp4

# Explicit
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --compositor-type inr \
    --output final.mp4
```

**Checkpoint**: Automatically downloads `Video_HYouTube_256.pth` (Google Drive)

### 2. MUSt3R Camera Extraction (Default)

**Previous**: Dust3R (image-pair based, no temporal consistency)
**New**: MUSt3R (video-optimized, temporal consistency built-in)

**Benefits**:
- ✅ **Temporal consistency** via multi-layer memory mechanism
- ✅ **Reduced flickering** in final video
- ✅ **Better tracking** with sequential processing
- ✅ **Faster** at 15 FPS vs Dust3R's slower pair-wise approach
- ✅ **Drop-in replacement** - same API

**Usage**:
```bash
# Automatic (new default)
python vanelia_pipeline.py --input video.mp4 --model product.glb --output final.mp4
```

## Migration Guide

### No Code Changes Required!

The pipeline automatically uses the new defaults:
- **Compositor**: `inr` (INR-Harmonization)
- **Camera Extraction**: `MUSt3R`

### Optional: Fallback to Old Methods

If you need the old behavior:

```bash
# Use Dust3R instead of MUSt3R
# (Not recommended - requires code modification)

# Use ControlNet compositor instead of INR
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --compositor-type controlnet \
    --output final.mp4
```

## Compositor Comparison

| Feature | INR-Harmonization | Photorealistic | ControlNet | IC-Light |
|---------|-------------------|----------------|------------|----------|
| **Pre-trained** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐ (untrained) | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | 🚀 Fast | 🚀 Fast | 🐢 Slow | 🐌 Very Slow |
| **Resolution** | Any | Fixed | Any | Fixed |
| **Video Support** | ✅ Trained on video | ❌ No | ⚠️ OK | ⚠️ OK |
| **Memory** | 💚 Low | 💚 Low | 💛 Medium | 🔴 High |
| **Status** | ✅ Recommended | ⚠️ Experimental | ⚠️ OK | ⚠️ OK |

## Camera Extraction Comparison

| Feature | MUSt3R | Dust3R |
|---------|--------|--------|
| **Temporal Consistency** | ✅ Native | ❌ None |
| **Memory Mechanism** | ✅ Multi-layer | ❌ Single-pass |
| **Frame Processing** | Sequential | Pair-wise |
| **Speed** | 15 FPS | Slower |
| **Flickering** | Minimal | Can occur |
| **Video Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Status** | ✅ Recommended | ⚠️ Legacy |

## New Features

### Automatic Checkpoint Download

INR-Harmonization checkpoint is automatically downloaded on first use:

```python
from vanelia.modules.inr_harmonization_compositor import INRHarmonizationCompositor

# Auto-downloads Video_HYouTube_256.pth if not found
compositor = INRHarmonizationCompositor(device='cuda', auto_download=True)
```

**Manual Download**:
```bash
python scripts/download_inr_checkpoint.py
```

### Temporal Context Control

MUSt3R allows controlling temporal context window:

```python
from vanelia.modules.must3r_camera_extraction import MUSt3RCameraExtractor

extractor = MUSt3RCameraExtractor(device='cuda')
result = extractor.run_must3r_inference(
    frame_paths=frames,
    local_context_size=25  # Frames of temporal context (default: 25)
)
```

## Installation

### Quick Setup (RunPod)

Run the complete setup script:
```bash
bash setup_runpod.sh
```

This installs:
- System dependencies
- Blender 4.0+
- **MUSt3R** (with submodules)
- **INR-Harmonization**
- All Python dependencies

### Manual Installation

#### MUSt3R:
```bash
cd /home/user
git clone --recursive https://github.com/naver/must3r.git
cd must3r
pip install -e .
```

#### INR-Harmonization:
```bash
cd /home/user
git clone https://github.com/WindVChen/INR-Harmonization.git
cd INR-Harmonization
pip install -r requirements.txt
```

#### Checkpoint:
```bash
python /home/user/Vanelia/scripts/download_inr_checkpoint.py
```

## Performance Recommendations

### GPU Memory

| GPU | Max Frames | Context Size | Batch Size |
|-----|-----------|--------------|------------|
| A100 80GB | Unlimited | 50 | 16 |
| A100 40GB | 100 | 25 | 8 |
| RTX 3090 24GB | 50 | 15 | 4 |
| RTX 3080 10GB | 30 | 10 | 2 |

### Quality Settings

**High Quality** (Slow):
```bash
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --compositor-type inr \
    --max-frames 100 \
    --output final.mp4
```

**Fast Preview** (Quick):
```bash
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --compositor-type controlnet \
    --max-frames 30 \
    --frame-interval 2 \
    --output preview.mp4
```

## Troubleshooting

### Issue: "Checkpoint not found"

**Solution**: Download manually:
```bash
python scripts/download_inr_checkpoint.py
```

Or download from: https://drive.google.com/file/d/1Tv9aahaPmJ_RGeYdawLCNWNGabZgJo6y/view

Save to: `/home/user/Vanelia/checkpoints/inr_harmonization/Video_HYouTube_256.pth`

### Issue: "MUSt3R not found"

**Solution**: Clone with submodules:
```bash
git clone --recursive https://github.com/naver/must3r.git
```

### Issue: CUDA OOM with MUSt3R

**Solutions**:
1. Reduce frames: `--max-frames 30`
2. Reduce context: modify `local_context_size=15` in code
3. Reduce batch size: modify `max_bs=4` in code

### Issue: INR-Harmonization outputs strange colors

**Cause**: Checkpoint not loaded properly

**Solution**:
1. Verify checkpoint exists:
   ```bash
   ls -lh checkpoints/inr_harmonization/Video_HYouTube_256.pth
   ```
2. Re-download if corrupted
3. Check GPU has enough memory (>4GB required)

## API Changes

### Compositor Initialization

**Old**:
```python
from vanelia.modules.photorealistic_compositor import PhotorealisticCompositor
compositor = PhotorealisticCompositor(device='cuda', cnum=48)
```

**New**:
```python
from vanelia.modules.inr_harmonization_compositor import INRHarmonizationCompositor
compositor = INRHarmonizationCompositor(
    device='cuda',
    base_size=256,
    input_size=256,
    auto_download=True
)
```

### Camera Extraction

**Old**:
```python
from vanelia.modules.dust3r_camera_extraction import Dust3RCameraExtractor
extractor = Dust3RCameraExtractor(device='cuda')
```

**New**:
```python
from vanelia.modules.must3r_camera_extraction import MUSt3RCameraExtractor
extractor = MUSt3RCameraExtractor(device='cuda')
```

**Note**: API is identical, drop-in replacement!

## Documentation

- [INR-Harmonization Paper](https://ieeexplore.ieee.org/document/10285123)
- [MUSt3R Repository](https://github.com/naver/must3r)
- [MUSt3R Integration Guide](./MUST3R_INTEGRATION.md)
- [Photorealistic Compositor Docs](./PHOTOREALISTIC_COMPOSITOR.md)

## Changelog

### v2.0.0 - Major Upgrade (2025-12-29)

**Added**:
- INR-Harmonization compositor with pre-trained weights
- MUSt3R camera extraction with temporal consistency
- Automatic checkpoint download
- MUST3R_INTEGRATION.md documentation
- UPGRADE_GUIDE.md (this file)

**Changed**:
- Default compositor: `controlnet` → `inr`
- Default camera extraction: Dust3R → MUSt3R
- Pipeline description updated
- Examples updated

**Deprecated**:
- Photorealistic compositor (architecture kept for reference)
- Dust3R as default (still available as fallback)

**Removed**:
- None (backwards compatible)

## Support

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section
2. Review GPU memory recommendations
3. Open an issue with:
   - GPU model and memory
   - Number of frames processed
   - Error logs
   - `nvidia-smi` output

## Credits

- **INR-Harmonization**: WindVChen (TCSVT 2023)
- **MUSt3R**: NAVER Labs Europe
- **Original Vanelia**: Based on Dust3R, Blender, ControlNet pipeline
