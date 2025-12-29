# Vanelia

**Video Object Insertion Pipeline** - Insert 3D models into videos with photorealistic lighting and temporal consistency.

## Overview

Vanelia is a complete pipeline for inserting 3D models (.glb) into user videos using state-of-the-art computer vision and rendering techniques:

- **Dust3R**: Camera tracking and pose estimation
- **Blender 4.0**: Headless 3D rendering with shadow catcher
- **ControlNet**: Depth-guided compositing with temporal consistency (recommended)
- **IC-Light**: Alternative relighting compositor (fallback)

Designed for **RunPod NVIDIA A100 (80GB)** servers with CUDA 12.1.

---

## Architecture

```
Input Video (.mp4)
    ↓
[Module A] Dust3R Camera Extraction
    → camera_poses.npy (Blender coordinate system)
    → camera_intrinsics.npy
    → background_frames/
    ↓
[Module B] Blender Headless Rendering
    + 3D Model (.glb)
    → render_frames/ (RGBA with transparent background)
    ↓
[Module C] ControlNet Depth Compositing (recommended)
    + background_frames/
    → Depth-guided refinement with temporal consistency
    → final_output.mp4

Alternative: IC-Light Compositing (fallback)
```

---

## Installation

### 1. System Requirements

- Ubuntu 20.04+
- NVIDIA GPU (A100 recommended)
- CUDA 12.1+
- Python 3.10+
- 50GB+ free disk space

### 2. Quick Setup

```bash
cd /home/user/Vanelia

# Install Blender 4.0
chmod +x scripts/install_blender.sh
sudo bash scripts/install_blender.sh

# Setup Python environment
chmod +x scripts/setup_environment.sh
sudo bash scripts/setup_environment.sh

# Activate environment
source /opt/vanelia_env/bin/activate
```

### 3. Manual Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Dust3R
git clone https://github.com/naver/dust3r.git
cd dust3r && pip install -e .

# Install Blender 4.0
wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz
tar -xf blender-4.0.2-linux-x64.tar.xz
sudo mv blender-4.0.2-linux-x64 /opt/blender
sudo ln -s /opt/blender/blender /usr/local/bin/blender
```

---

## Usage

### Complete Pipeline

```bash
python vanelia_pipeline.py \
    --input input_video.mp4 \
    --model brand_asset.glb \
    --output final_video.mp4
```

### Advanced Options

```bash
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --frame-interval 2 \
    --max-frames 100 \
    --strength 0.25 \
    --seed 12345 \
    --latent-blend 0.15 \
    --fps 30 \
    --crf 18
```

### Memory-Safe Video Chunking (OOM Prevention)

For high-resolution videos that may cause OOM (Out Of Memory) errors, use the chunking wrapper:

```bash
python memory_safe_chunking.py \
    --input input.mp4 \
    --model product.glb \
    --output final_result.mp4 \
    --chunk-duration 2 \
    --workspace ./chunk_workspace
```

**How it works:**
1. Splits input video into 2-second (or 60-frame) clips using ffmpeg
2. Processes each clip through Module A → B → C pipeline
3. **Clears GPU memory** after each clip (prevents OOM)
4. Merges all processed clips into final video

**Options:**
- `--chunk-duration 2.0`: Duration of each clip in seconds (default: 2.0)
- `--chunk-frames 60`: Alternative: number of frames per clip (overrides duration)
- `--keep-temp`: Keep temporary files for debugging

**Docker Command (RunPod):**

To prevent PyTorch memory fragmentation, add this to your RunPod Docker Command:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

This limits memory block sizes and helps prevent OOM errors.

### Module-by-Module Execution

#### Module A: Camera Extraction

```bash
python vanelia/modules/dust3r_camera_extraction.py \
    --input video.mp4 \
    --output ./camera_data \
    --interval 1 \
    --device cuda
```

**Key Feature**: Automatic coordinate system conversion from OpenCV to Blender using `T_blender_from_opencv` transformation matrix.

#### Module B: Blender Rendering

```bash
blender --background --python vanelia/modules/blender_render.py -- \
    --glb brand_asset.glb \
    --poses camera_data/camera_poses.npy \
    --intrinsics camera_data/camera_intrinsics.npy \
    --output render_frames/ \
    --resolution 1920 1080
```

**Output**: RGBA images with transparent background and shadow catcher.

#### Module C: ControlNet Compositing (Recommended)

```bash
python vanelia/modules/controlnet_compositor.py \
    --render-dir render_frames/ \
    --bg-dir background_frames/ \
    --output-dir refined_frames/ \
    --video-output final_output.mp4 \
    --controlnet-type depth \
    --strength 0.4 \
    --seed 12345
```

**Key Settings**:
- `--controlnet-type depth`: Depth-guided compositing (also supports: normal, canny)
- `--strength 0.4`: Denoising strength (0.3-0.5 for ControlNet)
- `--seed 12345`: Fixed seed (consistent noise pattern)

**Alternative: IC-Light Compositing (fallback)**

```bash
python vanelia/modules/iclight_compositor.py \
    --render-dir render_frames/ \
    --bg-dir background_frames/ \
    --output-dir refined_frames/ \
    --video-output final_output.mp4 \
    --strength 0.25 \
    --seed 12345 \
    --latent-blend 0.15
```

Note: IC-Light requires special foreground conditioning not available via standard HuggingFace pipelines. Use ControlNet for reliable results.

---

## Technical Details

### Coordinate System Conversion

**Critical**: Dust3R outputs camera poses in **OpenCV coordinates** (Right, Down, Forward), while Blender uses **OpenGL coordinates** (Right, Up, Back).

The conversion matrix in `dust3r_camera_extraction.py:106`:

```python
T_blender_from_opencv = np.array([
    [1,  0,  0, 0],  # X stays the same
    [0,  0,  1, 0],  # Y becomes Z
    [0, -1,  0, 0],  # Z becomes -Y
    [0,  0,  0, 1]
])
```

Without this conversion, rendered objects will appear **upside-down or mirrored**.

### Temporal Consistency

Both compositors prevent flickering through consistent settings:

**ControlNet (Recommended)**:
1. **Fixed Seed**: Same random seed for all frames
2. **Depth Guidance**: Geometric constraints from depth map (Intel/dpt-large)
3. **Moderate Denoising**: 0.3-0.5 strength with ControlNet conditioning
4. **UniPC Scheduler**: Fast, deterministic diffusion steps

**IC-Light (Fallback)**:
1. **Fixed Seed**: Same random seed for all frames
2. **Low Denoising Strength**: 0.2-0.3 (preserves Blender's geometry)
3. **Latent Blending**: 10-20% of previous frame's latent mixed in
4. **DDIM Scheduler**: Deterministic diffusion steps

Note: ControlNet provides better geometric consistency through depth guidance.

### Shadow Catcher

Blender renders objects with realistic shadows on an invisible ground plane:

```python
plane.is_shadow_catcher = True  # Invisible but receives shadows
scene.render.film_transparent = True  # RGBA output
```

### Why ControlNet Instead of IC-Light?

**IC-Light Limitations**:
- Requires special foreground conditioning images (FC mode)
- Not directly available via standard HuggingFace diffusers pipelines
- Complex setup requiring custom preprocessing

**ControlNet Advantages**:
- Works directly with HuggingFace diffusers (plug-and-play)
- Depth guidance provides better geometric consistency
- Multiple control types available (depth, normal, canny)
- Well-documented and actively maintained
- Automatic depth estimation using Intel DPT-Large

**Result**: ControlNet provides more reliable and consistent compositing with simpler implementation.

---

## Directory Structure

```
Vanelia/
├── vanelia/
│   ├── modules/
│   │   ├── dust3r_camera_extraction.py   # Module A
│   │   ├── blender_render.py             # Module B
│   │   ├── controlnet_compositor.py      # Module C (recommended)
│   │   └── iclight_compositor.py         # Module C (fallback)
│   └── __init__.py
├── scripts/
│   ├── install_blender.sh
│   └── setup_environment.sh
├── vanelia_pipeline.py                    # Main orchestrator
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### Objects Appear Upside-Down

**Cause**: Coordinate system mismatch
**Solution**: Verify `T_blender_from_opencv` is applied in `dust3r_camera_extraction.py:106`

### Video Flickers/Jitters

**Cause**: High denoising strength or inconsistent seed
**Solution**: Lower `--strength` to 0.2-0.25, ensure `--seed` is fixed

### Out of Memory (OOM)

**Cause**: Too many frames or high resolution
**Solution**: Use `--max-frames 50` or reduce `--resolution 1280 720`

### Blender Crashes

**Cause**: GPU memory exceeded
**Solution**: Reduce Cycles samples in `blender_render.py:53` or use CPU

---

## Performance

**Tested on NVIDIA A100 (80GB)**:

- Dust3R: ~0.5s/frame
- Blender: ~2-5s/frame (Cycles 64 samples with Optix)
- ControlNet: ~1.8s/frame (20 steps, depth guidance)
- IC-Light: ~1.5s/frame (20 steps, strength 0.25)

**Total (ControlNet)**: ~4.3-7.3s per frame (100 frames ≈ 7-12 minutes)
**Total (IC-Light)**: ~4-7s per frame (100 frames ≈ 7-12 minutes)

---

## Credits

- **Dust3R**: [naver/dust3r](https://github.com/naver/dust3r)
- **ControlNet**: [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
- **IC-Light**: [lllyasviel/IC-Light](https://github.com/lllyasviel/IC-Light)
- **Blender**: [blender.org](https://www.blender.org/)
- **Depth Estimation**: Intel DPT-Large via HuggingFace Transformers

---

## License

MIT License - See LICENSE file for details.

---

## Citation

If you use Vanelia in research, please cite:

```bibtex
@software{vanelia2024,
  title={Vanelia: Video Object Insertion Pipeline},
  author={Vanelia Team},
  year={2024},
  url={https://github.com/your-repo/vanelia}
}
```