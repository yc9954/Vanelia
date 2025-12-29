# MUSt3R Integration

## Overview

Vanelia now uses **MUSt3R** (Multi-view Network for Stereo 3D Reconstruction) instead of Dust3R for camera pose extraction. MUSt3R provides significantly better **temporal consistency** for video sequences, reducing flickering and improving overall quality.

## Key Improvements

### 1. Temporal Consistency

Unlike Dust3R which processes image pairs independently, MUSt3R:
- Uses a **multi-layer memory mechanism** to maintain temporal context
- Processes frames **sequentially** with causal attention
- Applies **feedback mechanisms** after each block for progressive refinement
- Supports up to **300 memory frames** for long-range consistency

### 2. Video-Optimized Architecture

MUSt3R extends Dust3R specifically for video:
- **Symmetric architecture** for multi-view processing
- **Online predictions** of camera pose and 3D structure
- **Local context windows** (default: 25 frames) for efficient processing
- Operates at **15 FPS** with high-quality reconstruction

### 3. Better 3D Reconstruction

- **Multi-view SLAM support** for robust tracking
- **KDTree-based keyframe selection** for optimal frame distribution
- **Improved point cloud quality** through temporal aggregation

## Usage

### Basic Usage (Default)

MUSt3R is now the **default** camera extraction method:

```bash
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4
```

### Fallback to Dust3R

If you need to use Dust3R (not recommended for video):

```python
from vanelia_pipeline import VaneliaPipeline

pipeline = VaneliaPipeline()
result = pipeline.step1_extract_camera_poses(
    video_path="video.mp4",
    use_must3r=False  # Use Dust3R instead
)
```

### Advanced Parameters

```python
from vanelia.modules.must3r_camera_extraction import MUSt3RCameraExtractor

extractor = MUSt3RCameraExtractor(device='cuda')

result = extractor.run_must3r_inference(
    frame_paths=frame_list,
    max_bs=8,                    # Batch size
    init_num_images=8,           # Initial keyframes
    batch_num_views=4,           # Frames per batch
    local_context_size=25        # Temporal context window
)
```

## Comparison: MUSt3R vs Dust3R

| Feature | MUSt3R | Dust3R |
|---------|--------|--------|
| **Temporal Consistency** | ✅ Native support | ❌ No temporal awareness |
| **Processing Mode** | Sequential video | Image pairs |
| **Memory Mechanism** | Multi-layer feedback | Single-pass |
| **Frame-rate** | 15 FPS | Slower |
| **Flickering** | Minimal | Can occur |
| **Video Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Best Use Case** | Video sequences | Static scenes, image collections |

## Installation

MUSt3R is automatically installed via `setup_runpod.sh`:

```bash
# Manual installation
cd /home/user
git clone --recursive https://github.com/naver/must3r.git
cd must3r
pip install -e .
```

## Implementation Details

### Memory Mechanism

MUSt3R maintains temporal consistency through:

1. **Causal Attention**: Each frame only attends to previous frames
2. **Memory Blocks**: Process frames in batches with overlapping context
3. **Feedback Loop**: Updates memory state progressively through sequence

### Coordinate System

Same as Dust3R:
- Input: OpenCV convention (Right, Down, Forward)
- Output: Blender convention (Right, Up, Back)
- Conversion handled automatically

### Output Format

```python
{
    'camera_poses': np.ndarray,      # (N, 4, 4) w2c matrices in Blender coords
    'camera_intrinsics': np.ndarray, # (N, 3, 3) intrinsic matrices
    'ground_plane': dict,            # RANSAC-detected ground plane
    'frame_paths': list,             # Input frame paths
    'output_dir': str                # Output directory
}
```

## Performance Tips

1. **Context Size**: Increase `local_context_size` for longer temporal consistency
   - Default: 25 frames
   - Max recommended: 50 frames (higher memory usage)

2. **Batch Size**: Adjust based on GPU memory
   - A100 80GB: `max_bs=16`
   - A100 40GB: `max_bs=8`
   - RTX 3090: `max_bs=4`

3. **Frame Sampling**: For very long videos, increase `frame_interval`
   - 30 FPS video: `frame_interval=2` → 15 FPS output
   - 60 FPS video: `frame_interval=4` → 15 FPS output

## Troubleshooting

### Issue: "MUSt3R not found"

**Solution**: Ensure MUSt3R is cloned with `--recursive` flag:
```bash
git clone --recursive https://github.com/naver/must3r.git
```

### Issue: CUDA out of memory

**Solutions**:
1. Reduce `max_bs`: `max_bs=4`
2. Reduce `local_context_size`: `local_context_size=15`
3. Limit frames: `max_frames=30`

### Issue: Slow inference

**Solutions**:
1. Enable AMP (automatic mixed precision) - already enabled by default
2. Reduce image size - default 512 is optimal
3. Use fewer keyframes: `init_num_images=6`

## References

- **Paper**: MUSt3R: Multi-view Network for Stereo 3D Reconstruction
- **GitHub**: https://github.com/naver/must3r
- **Related**: MASt3R-SLAM for real-time dense SLAM

## Migration from Dust3R

No code changes needed! The interface is identical:

```python
# Old (Dust3R)
from vanelia.modules.dust3r_camera_extraction import Dust3RCameraExtractor
extractor = Dust3RCameraExtractor(device='cuda')

# New (MUSt3R)
from vanelia.modules.must3r_camera_extraction import MUSt3RCameraExtractor
extractor = MUSt3RCameraExtractor(device='cuda')

# Same API
result = extractor.process_video(video_path, output_dir)
```

The pipeline automatically uses MUSt3R by default.
