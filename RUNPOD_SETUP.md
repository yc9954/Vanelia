# Vanelia - RunPod Setup Guide

Complete setup guide for running Vanelia on RunPod NVIDIA A100 (80GB).

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Launch RunPod Instance

**Recommended Pod:**
- GPU: NVIDIA A100 (80GB) or RTX 4090
- Template: PyTorch 2.0+ with CUDA 12.1+
- Disk: 100GB minimum
- Region: Any with A100 availability

**Docker Command (Optional):**
Add this to prevent OOM errors:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Step 2: Clone Repository

```bash
cd /workspace
git clone https://github.com/yc9954/Vanelia.git
cd Vanelia
```

### Step 3: Run Setup Script

```bash
bash setup_runpod.sh
```

This will:
- ✅ Install Blender 4.0.2 to `/workspace/blender`
- ✅ Install Python dependencies
- ✅ Install Dust3R to `/workspace/dust3r`
- ✅ Configure PyTorch memory management
- ✅ Verify all installations

**Setup takes ~5 minutes on A100.**

### Step 4: Run Your First Pipeline

```bash
# Basic usage
python vanelia_pipeline.py \
    --input /workspace/input.mp4 \
    --model /workspace/model.glb \
    --output /workspace/output.mp4
```

---

## 📁 Directory Structure

```
/workspace/
├── Vanelia/              # Main repository
│   ├── vanelia_pipeline.py
│   ├── memory_safe_chunking.py
│   └── vanelia/
│       └── modules/
│           ├── dust3r_camera_extraction.py
│           ├── blender_render.py
│           ├── controlnet_compositor.py
│           └── iclight_compositor.py
├── blender/              # Blender 4.0.2 installation
├── dust3r/               # Dust3R camera tracking
├── input.mp4             # Your input video
├── model.glb             # Your 3D model
└── output.mp4            # Final result
```

---

## 🎯 Usage Examples

### Example 1: Basic Run

```bash
cd /workspace/Vanelia

python vanelia_pipeline.py \
    --input /workspace/my_video.mp4 \
    --model /workspace/chair.glb \
    --output /workspace/result.mp4
```

### Example 2: High Quality + ControlNet-Normal

```bash
python vanelia_pipeline.py \
    --input /workspace/video.mp4 \
    --model /workspace/product.glb \
    --compositor-type controlnet \
    --controlnet-type normal \
    --strength 0.45 \
    --crf 15 \
    --fps 60 \
    --output /workspace/result_hq.mp4
```

### Example 3: Memory-Safe Mode (For Large Videos)

```bash
# Prevents OOM by processing video in 2-second chunks
python memory_safe_chunking.py \
    --input /workspace/large_video.mp4 \
    --model /workspace/object.glb \
    --output /workspace/result.mp4 \
    --chunk-duration 2 \
    --workspace /workspace/temp_chunks
```

### Example 4: Manual Object Placement

```bash
python vanelia_pipeline.py \
    --input /workspace/video.mp4 \
    --model /workspace/desk.glb \
    --no-auto-placement \
    --object-location 0.0 0.0 -2.0 \
    --object-scale 0.5 \
    --output /workspace/result.mp4
```

---

## ⚙️ Key Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--compositor-type` | `controlnet` or `iclight` | `controlnet` | `controlnet` |
| `--controlnet-type` | `depth`, `normal`, `canny` | `depth` | `depth` or `normal` |
| `--strength` | Denoising strength | `0.4` | `0.35-0.5` (ControlNet)<br>`0.2-0.3` (IC-Light) |
| `--auto-placement` | Auto object placement | `True` | `True` |
| `--fps` | Output FPS | `30` | `30` or `60` |
| `--crf` | Quality (lower=better) | `18` | `15-20` |
| `--chunk-duration` | Chunk size (seconds) | `2.0` | `2.0-5.0` |

---

## 🔧 Troubleshooting

### Issue 1: Out of Memory (OOM) Error

**Solution 1: Use Memory-Safe Chunking**
```bash
python memory_safe_chunking.py \
    --input video.mp4 \
    --model model.glb \
    --output output.mp4 \
    --chunk-duration 2
```

**Solution 2: Set PyTorch Memory Config**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**Solution 3: Reduce Batch Size**
Edit `vanelia_pipeline.py` line 387:
```python
batch_size=4  # Change from 8 to 4
```

### Issue 2: Blender Not Found

```bash
# Add Blender to PATH
export PATH="/workspace/blender:$PATH"

# Or use absolute path
/workspace/blender/blender --version
```

### Issue 3: Dust3R Import Error

```bash
# Manually add to Python path
export PYTHONPATH="/workspace/dust3r:$PYTHONPATH"

# Or reinstall
cd /workspace/dust3r
pip install -e .
```

### Issue 4: CUDA Out of Memory

```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce resolution
python vanelia_pipeline.py \
    --input video.mp4 \
    --model model.glb \
    --resolution 1280 720 \
    --output output.mp4
```

---

## 📊 Performance Benchmarks

**NVIDIA A100 80GB:**

| Resolution | FPS | Memory Usage | Processing Speed |
|------------|-----|--------------|------------------|
| 1920x1080 | 30 | ~35GB | 4.5s/frame |
| 1280x720 | 30 | ~20GB | 3.2s/frame |
| 3840x2160 | 30 | ~65GB | 8.1s/frame |

**Example: 100 frames @ 1080p**
- Total time: ~7-8 minutes
- Peak VRAM: ~35GB
- CPU usage: 10-20%

---

## 🎬 Complete Workflow

```bash
# 1. Upload files to RunPod
# Use RunPod web interface to upload:
# - input.mp4 (your video)
# - model.glb (your 3D model)

# 2. SSH into RunPod
# ssh [email protected] -i ~/.ssh/id_ed25519

# 3. Navigate to workspace
cd /workspace/Vanelia

# 4. Run pipeline
python vanelia_pipeline.py \
    --input /workspace/input.mp4 \
    --model /workspace/model.glb \
    --output /workspace/output.mp4

# 5. Download result
# Use RunPod web interface to download output.mp4
```

---

## 🔄 Updating Vanelia

```bash
cd /workspace/Vanelia
git pull origin main
pip install -r requirements.txt --upgrade
```

---

## 💡 Tips & Best Practices

1. **Use Memory-Safe Chunking** for videos > 200 frames
2. **ControlNet-Depth** works best for most cases
3. **ControlNet-Normal** better for detailed surfaces (products, furniture)
4. **Start with low resolution** (720p) for testing, then scale up
5. **Fixed seed** (default: 12345) ensures consistent results
6. **CRF 18** is good balance (15=best quality, 23=smaller file)
7. **Auto-placement** works 90% of the time - manual only if needed

---

## 📝 Example Commands Cheatsheet

```bash
# Quick test (low res, fast)
python vanelia_pipeline.py --input video.mp4 --model model.glb --output test.mp4 --resolution 854 480 --max-frames 30

# Production quality
python vanelia_pipeline.py --input video.mp4 --model model.glb --output final.mp4 --strength 0.4 --crf 15 --fps 60

# Large video (OOM prevention)
python memory_safe_chunking.py --input long_video.mp4 --model model.glb --output final.mp4 --chunk-duration 3

# Debug mode (see intermediate results)
python vanelia_pipeline.py --input video.mp4 --model model.glb --output final.mp4 --workspace /workspace/debug_output
```

---

## 🆘 Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Check GitHub issues: https://github.com/yc9954/Vanelia/issues
3. Verify GPU is available: `nvidia-smi`
4. Check Python version: `python --version` (should be 3.10+)
5. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 📄 License

MIT License - See LICENSE file for details.
