# Vanelia

**Video Object Insertion Pipeline** - Insert 3D models into videos with photorealistic lighting and temporal consistency.

## Overview

Vanelia is a complete pipeline for inserting 3D models (.glb) into user videos using state-of-the-art computer vision and rendering techniques:

- **Dust3R**: Camera tracking and pose estimation
- **Blender 4.0**: Headless 3D rendering with shadow catcher
- **IC-Light**: Photorealistic relighting and compositing with temporal consistency

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
[Module C] IC-Light Compositing
    + background_frames/
    → Temporal-consistent refinement
    → final_output.mp4
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

#### Module C: IC-Light Compositing

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

**Anti-Flicker Settings**:
- `--strength 0.25`: Low denoising (preserves geometry)
- `--seed 12345`: Fixed seed (consistent noise pattern)
- `--latent-blend 0.15`: Temporal latent blending

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

IC-Light compositor prevents flickering through:

1. **Fixed Seed**: Same random seed for all frames
2. **Low Denoising Strength**: 0.2-0.3 (preserves Blender's geometry)
3. **Latent Blending**: 10-20% of previous frame's latent mixed in
4. **DDIM Scheduler**: Deterministic diffusion steps

### Shadow Catcher

Blender renders objects with realistic shadows on an invisible ground plane:

```python
plane.is_shadow_catcher = True  # Invisible but receives shadows
scene.render.film_transparent = True  # RGBA output
```

---

## Directory Structure

```
Vanelia/
├── vanelia/
│   ├── modules/
│   │   ├── dust3r_camera_extraction.py   # Module A
│   │   ├── blender_render.py             # Module B
│   │   └── iclight_compositor.py         # Module C
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
- Blender: ~2-5s/frame (Cycles 128 samples)
- IC-Light: ~1.5s/frame (20 steps, strength 0.25)

**Total**: ~4-7s per frame (100 frames ≈ 10-12 minutes)

---

## Credits

- **Dust3R**: [naver/dust3r](https://github.com/naver/dust3r)
- **IC-Light**: [lllyasviel/IC-Light](https://github.com/lllyasviel/IC-Light)
- **Blender**: [blender.org](https://www.blender.org/)

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