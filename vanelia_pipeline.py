#!/usr/bin/env python3
"""
Vanelia - Video Object Insertion Pipeline
Complete end-to-end pipeline for inserting 3D models into videos.

Pipeline:
1. Dust3R: Extract camera poses from video
2. Blender: Render 3D model with camera animation
3. IC-Light: Composite and refine with temporal consistency
4. FFmpeg: Encode final video

Usage:
    python vanelia_pipeline.py \
        --input video.mp4 \
        --model brand_asset.glb \
        --output final_video.mp4
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil
import time
import numpy as np

# Add Dust3R to Python path if it exists
DUST3R_PATH = "/tmp/dust3r"
if os.path.exists(DUST3R_PATH) and DUST3R_PATH not in sys.path:
    sys.path.insert(0, DUST3R_PATH)


class VaneliaPipeline:
    """End-to-end video object insertion pipeline."""

    def __init__(self, workspace: str = "./vanelia_workspace"):
        """
        Initialize pipeline workspace.

        Args:
            workspace: Working directory for intermediate files
        """
        self.workspace = Path(workspace)
        self.workspace.mkdir(exist_ok=True)

        # Define subdirectories
        self.dirs = {
            'input': self.workspace / 'input',
            'camera_data': self.workspace / 'camera_data',
            'background_frames': self.workspace / 'background_frames',
            'render_frames': self.workspace / 'render_frames',
            'refined_frames': self.workspace / 'refined_frames',
            'output': self.workspace / 'output'
        }

        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("VANELIA - Video Object Insertion Pipeline")
        print(f"{'='*70}\n")
        print(f"Workspace: {self.workspace}\n")

    def step1_extract_camera_poses(self, video_path: str,
                                  frame_interval: int = 1,
                                  max_frames: int = None) -> dict:
        """
        Step 1: Extract camera poses using Dust3R.

        Args:
            video_path: Input video path
            frame_interval: Frame sampling interval
            max_frames: Maximum frames to process

        Returns:
            Camera data dictionary
        """
        print(f"\n{'─'*70}")
        print("STEP 1: Camera Pose Extraction (Dust3R)")
        print(f"{'─'*70}\n")

        from vanelia.modules.dust3r_camera_extraction import Dust3RCameraExtractor

        extractor = Dust3RCameraExtractor(device='cuda')

        result = extractor.process_video(
            video_path=video_path,
            output_dir=str(self.dirs['camera_data']),
            frame_interval=frame_interval,
            max_frames=max_frames
        )

        # Move background frames to correct location
        bg_src = self.dirs['camera_data'] / 'background_frames'
        if bg_src.exists():
            for frame in bg_src.glob('*'):
                shutil.copy(frame, self.dirs['background_frames'])

        print(f"\n✓ Step 1 Complete: {len(result['frame_paths'])} camera poses extracted")
        return result

    def step2_render_object(self, glb_path: str,
                          model_scale: float = 1.0,
                          resolution: tuple = (1920, 1080),
                          model_location: tuple = None,
                          plane_normal: np.ndarray = None) -> list:
        """
        Step 2: Render 3D model using Blender.

        Args:
            glb_path: Path to .glb model
            model_scale: Model scale factor
            resolution: Output resolution (width, height)
            model_location: (x, y, z) position in Blender coordinates
            plane_normal: Normal vector of detected plane (for object alignment)

        Returns:
            List of rendered frame paths
        """
        print(f"\n{'─'*70}")
        print("STEP 2: Object Rendering (Blender)")
        print(f"{'─'*70}\n")

        poses_path = self.dirs['camera_data'] / 'camera_poses.npy'
        intrinsics_path = self.dirs['camera_data'] / 'camera_intrinsics.npy'
        metadata_path = self.dirs['camera_data'] / 'camera_metadata.json'

        if not poses_path.exists():
            raise FileNotFoundError(f"Camera poses not found: {poses_path}")

        # Construct Blender command
        blender_script = Path(__file__).parent / 'vanelia' / 'modules' / 'blender_render.py'

        cmd = [
            'blender',
            '--background',
            '--python', str(blender_script),
            '--',
            '--glb', glb_path,
            '--poses', str(poses_path),
            '--intrinsics', str(intrinsics_path),
            '--metadata', str(metadata_path),
            '--output', str(self.dirs['render_frames']),
            '--scale', str(model_scale),
            '--position', str(position[0]), str(position[1]), str(position[2]),
            '--rotation', str(rotation[0]), str(rotation[1]), str(rotation[2]),
            '--resolution', str(resolution[0]), str(resolution[1])
        ]
        
        # Add location if provided
        # Convert to fixed-point format to avoid scientific notation parsing issues
        if model_location is not None:
            loc_str = [f"{float(x):.10f}" for x in model_location]
            cmd.extend(['--location'] + loc_str)
        
        # Add plane normal if provided (for rotation alignment)
        # Convert to fixed-point format to avoid scientific notation parsing issues
        if plane_normal is not None:
            normal_str = [f"{float(x):.10f}" for x in plane_normal]
            cmd.extend(['--plane-normal'] + normal_str)

        if auto_ground:
            cmd.append('--auto-ground')

        print(f"Running Blender...\n")
        result = subprocess.run(cmd, check=True)

        render_frames = sorted(self.dirs['render_frames'].glob('*.png'))
        print(f"\n✓ Step 2 Complete: {len(render_frames)} frames rendered")
        return render_frames

    def step3_composite_and_refine(self,
                                  strength: float = 0.4,
                                  seed: int = 12345,
                                  latent_blend: float = 0.15,
                                  fps: int = 30,
                                  crf: int = 18,
                                  output_path: str = None,
                                  batch_size: int = 4) -> str:
        """
        Step 3: Composite and refine using ControlNet or IC-Light.

        Args:
            strength: Denoising strength (0.3-0.5 for ControlNet, 0.2-0.3 for IC-Light)
            seed: Fixed random seed
            latent_blend: Latent blending ratio (IC-Light only)
            fps: Output video FPS
            crf: Video quality (18=high)
            compositor_type: 'controlnet' (recommended) or 'iclight'
            controlnet_type: 'depth', 'normal', or 'canny' (ControlNet only)
            output_path: Final video output path

        Returns:
            Path to final video
        """
        print(f"\n{'─'*70}")
        print(f"STEP 3: Compositing & Refinement ({compositor_type.upper()})")
        print(f"{'─'*70}\n")

        if compositor_type == "controlnet":
            from vanelia.modules.controlnet_compositor import ControlNetCompositor

            # Initialize ControlNet compositor
            compositor = ControlNetCompositor(
                controlnet_type=controlnet_type,
                device='cuda'
            )

        # Process frames with batch processing
        output_frames = compositor.process_video_sequence(
            render_dir=str(self.dirs['render_frames']),
            background_dir=str(self.dirs['background_frames']),
            output_dir=str(self.dirs['refined_frames']),
            strength=strength,
            seed=seed,
            latent_blend_ratio=latent_blend,
            batch_size=batch_size
        )

        # Encode video
        if output_path is None:
            output_path = str(self.dirs['output'] / 'final_output.mp4')

        final_video = compositor.encode_video_ffmpeg(
            frame_dir=str(self.dirs['refined_frames']),
            output_path=output_path,
            fps=fps,
            crf=crf
        )

        print(f"\n✓ Step 3 Complete: Video saved to {final_video}")
        return final_video

    def run_full_pipeline(self,
                         video_path: str,
                         glb_path: str,
                         output_path: str,
                         frame_interval: int = 1,
                         max_frames: int = None,
                         model_scale: float = None,
                         resolution: tuple = (1920, 1080),
                         compositor_type: str = "controlnet",
                         controlnet_type: str = "depth",
                         strength: float = 0.4,
                         seed: int = 12345,
                         latent_blend: float = 0.15,
                         fps: int = 30,
                         crf: int = 18,
                         skip_step1: bool = False,
                         skip_step2: bool = False,
                         auto_placement: bool = True,
                         manual_location: tuple = None,
                         manual_scale: float = None) -> str:
        """
        Run complete Vanelia pipeline with resume capability.

        Args:
            video_path: Input video
            glb_path: 3D model file
            output_path: Final output video
            frame_interval: Frame sampling
            max_frames: Max frames to process
            model_scale: 3D model scale
            position: Object position (x, y, z)
            rotation: Object rotation in degrees (x, y, z)
            auto_ground: Auto-place on detected ground
            resolution: Output resolution
            compositor_type: Compositor to use ('controlnet' or 'iclight')
            controlnet_type: ControlNet type ('depth', 'normal', 'canny')
            strength: Denoising strength (0.3-0.5 for ControlNet, 0.2-0.3 for IC-Light)
            seed: Random seed (fixed for consistency)
            latent_blend: Temporal latent blending (IC-Light only)
            fps: Output FPS
            crf: Video quality
            skip_step1: Skip camera extraction if already done
            skip_step2: Skip rendering if already done

        Returns:
            Path to final video
        """
        start_time = time.time()

        # Step 1: Camera Extraction (skip if already done)
        if not skip_step1:
            self.step1_extract_camera_poses(
                video_path=video_path,
                frame_interval=frame_interval,
                max_frames=max_frames
            )
        else:
            poses_path = self.dirs['camera_data'] / 'camera_poses.npy'
            if poses_path.exists():
                print(f"\n[Resume] Skipping Step 1: Camera poses already exist")
            else:
                print(f"\n[Resume] WARNING: Step 1 output not found, running anyway...")
                self.step1_extract_camera_poses(
                    video_path=video_path,
                    frame_interval=frame_interval,
                    max_frames=max_frames
                )

        # Analyze object placement (after Step 1, before Step 2)
        from vanelia.modules.object_placement import analyze_placement_from_camera_data
        
        print(f"\n{'─'*70}")
        print("OBJECT PLACEMENT ANALYSIS")
        print(f"{'─'*70}\n")
        
        model_location, determined_scale, plane_normal = analyze_placement_from_camera_data(
            camera_data_dir=self.dirs['camera_data'],
            auto_placement=auto_placement,
            manual_location=manual_location,
            manual_scale=manual_scale if manual_scale is not None else model_scale
        )
        
        # Use determined scale if model_scale was not explicitly provided
        final_scale = model_scale if model_scale is not None else determined_scale
        
        print(f"\n{'='*70}")
        print("OBJECT PLACEMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Location (X, Y, Z): ({model_location[0]:.6f}, {model_location[1]:.6f}, {model_location[2]:.6f})")
        print(f"Scale: {final_scale:.3f}")
        if plane_normal is not None:
            print(f"Plane Normal: ({plane_normal[0]:.6f}, {plane_normal[1]:.6f}, {plane_normal[2]:.6f})")
            print(f"  → Object Z-axis will be aligned with this normal")
        print(f"{'='*70}\n")

        # Step 2: Object Rendering (skip if already done)
        if not skip_step2:
            self.step2_render_object(
                glb_path=glb_path,
                model_scale=final_scale,
                resolution=resolution,
                model_location=model_location,
                plane_normal=plane_normal
            )
        else:
            render_frames = sorted(self.dirs['render_frames'].glob('*.png'))
            bg_frames = sorted(self.dirs['background_frames'].glob('*.jpg')) + \
                       sorted(self.dirs['background_frames'].glob('*.png'))
            
            if render_frames:
                if len(render_frames) != len(bg_frames):
                    print(f"\n[Resume] WARNING: Frame count mismatch detected:")
                    print(f"  - Render frames: {len(render_frames)}")
                    print(f"  - Background frames: {len(bg_frames)}")
                    print(f"  - Re-running Step 2 to generate all {len(bg_frames)} render frames...")
                    # Clear existing render frames to regenerate
                    for frame in render_frames:
                        frame.unlink()
                    self.step2_render_object(
                        glb_path=glb_path,
                        model_scale=final_scale,
                        resolution=resolution,
                        model_location=model_location,
                        plane_normal=plane_normal
                    )
                else:
                    print(f"\n[Resume] Skipping Step 2: {len(render_frames)} frames already rendered")
            else:
                print(f"\n[Resume] WARNING: Step 2 output not found, running anyway...")
                self.step2_render_object(
                    glb_path=glb_path,
                    model_scale=final_scale,
                    resolution=resolution,
                    model_location=model_location,
                    plane_normal=plane_normal
                )

        # Step 3: Compositing & Refinement (always run, can be retried)
        # Use larger batch size for A100 (8 frames at once for faster inference)
        final_video = self.step3_composite_and_refine(
            compositor_type=compositor_type,
            controlnet_type=controlnet_type,
            strength=strength,
            seed=seed,
            latent_blend=latent_blend,
            fps=fps,
            crf=crf,
            output_path=output_path,
            batch_size=8  # A100 80GB can handle 8 frames at once
        )

        elapsed_time = time.time() - start_time

        print(f"\n{'='*70}")
        print("✓ VANELIA PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
        print(f"Final video: {final_video}")
        print(f"{'='*70}\n")

        return final_video


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Vanelia - Video Object Insertion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (ControlNet-Depth, recommended)
  python vanelia_pipeline.py --input video.mp4 --model product.glb --output final.mp4

  # Use IC-Light compositor instead
  python vanelia_pipeline.py --input video.mp4 --model product.glb \\
      --compositor-type iclight --strength 0.25 --output final.mp4

  # Use ControlNet-Normal for better surface detail
  python vanelia_pipeline.py --input video.mp4 --model product.glb \\
      --controlnet-type normal --strength 0.4 --output final.mp4

  # Process every 2nd frame, max 100 frames
  python vanelia_pipeline.py --input video.mp4 --model product.glb \\
      --frame-interval 2 --max-frames 100 --output final.mp4

  # High quality settings
  python vanelia_pipeline.py --input video.mp4 --model product.glb \\
      --strength 0.35 --crf 15 --fps 60 --output final.mp4
        """
    )

    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--model', type=str, required=True,
                       help='3D model file (.glb)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output video path')

    # Pipeline settings
    parser.add_argument('--workspace', type=str, default='./vanelia_workspace',
                       help='Working directory (default: ./vanelia_workspace)')
    parser.add_argument('--frame-interval', type=int, default=1,
                       help='Frame sampling interval (default: 1)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (default: all)')

    # Rendering settings
    parser.add_argument('--model-scale', type=float, default=None,
                       help='3D model scale (default: auto-detect from point cloud)')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080],
                       help='Output resolution (default: 1920 1080)')
    
    # Object placement
    parser.add_argument('--auto-placement', action='store_true', default=True,
                       help='Automatically find optimal object placement (default: True)')
    parser.add_argument('--no-auto-placement', dest='auto_placement', action='store_false',
                       help='Disable automatic placement (use default location)')
    parser.add_argument('--object-location', type=float, nargs=3, default=None,
                       metavar=('X', 'Y', 'Z'),
                       help='Manual object location in Blender coordinates (overrides auto-placement)')
    parser.add_argument('--object-scale', type=float, default=None,
                       help='Manual object scale (overrides auto-placement)')

    # Compositing settings
    parser.add_argument('--compositor-type', type=str, default='controlnet',
                       choices=['controlnet', 'iclight'],
                       help='Compositor to use (default: controlnet, recommended)')
    parser.add_argument('--controlnet-type', type=str, default='depth',
                       choices=['depth', 'normal', 'canny'],
                       help='ControlNet type (default: depth, only used with --compositor-type controlnet)')
    parser.add_argument('--strength', type=float, default=0.4,
                       help='Denoising strength (default: 0.4 for ControlNet, use 0.25 for IC-Light)')
    parser.add_argument('--seed', type=int, default=12345,
                       help='Random seed for consistency (default: 12345)')
    parser.add_argument('--latent-blend', type=float, default=0.15,
                       help='Latent blending ratio for IC-Light (default: 0.15, range: 0.0-0.2)')

    # Video encoding
    parser.add_argument('--fps', type=int, default=30,
                       help='Output FPS (default: 30)')
    parser.add_argument('--crf', type=int, default=18,
                       help='Video quality CRF (default: 18, lower=better)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"ERROR: 3D model not found: {args.model}")
        sys.exit(1)

    # Parameter validation warnings
    if args.compositor_type == 'controlnet' and args.strength < 0.3:
        print(f"⚠ WARNING: Strength {args.strength} is low for ControlNet (recommended: 0.3-0.5)")
        print(f"  Consider using --strength 0.4 for better results")

    if args.compositor_type == 'iclight' and args.strength > 0.35:
        print(f"⚠ WARNING: Strength {args.strength} is high for IC-Light (recommended: 0.2-0.3)")
        print(f"  High strength may cause flickering. Consider using --strength 0.25")

    # Initialize pipeline
    pipeline = VaneliaPipeline(workspace=args.workspace)

    # Run pipeline
    try:
        final_video = pipeline.run_full_pipeline(
            video_path=args.input,
            glb_path=args.model,
            output_path=args.output,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            model_scale=args.model_scale,
            position=tuple(args.position),
            rotation=tuple(args.rotation),
            auto_ground=not args.no_auto_ground,
            resolution=tuple(args.resolution),
            compositor_type=args.compositor_type,
            controlnet_type=args.controlnet_type,
            strength=args.strength,
            seed=args.seed,
            latent_blend=args.latent_blend,
            fps=args.fps,
            crf=args.crf,
            auto_placement=args.auto_placement,
            manual_location=tuple(args.object_location) if args.object_location else None,
            manual_scale=args.object_scale
        )

        print(f"SUCCESS: {final_video}")
        sys.exit(0)

    except Exception as e:
        print(f"\nERROR: Pipeline failed")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
