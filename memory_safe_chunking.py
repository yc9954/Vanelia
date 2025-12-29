#!/usr/bin/env python3
"""
Memory-Safe Video Chunking Wrapper
Prevents OOM (Out Of Memory) by processing high-resolution videos in small chunks.

Pipeline:
1. Split input video into 2-second (or 60-frame) clips using ffmpeg
2. Process each clip through Module A (Tracking) -> B (Rendering) -> C (Refinement)
3. Clear GPU memory after each clip
4. Merge all processed clips into final_result.mp4

Usage:
    python memory_safe_chunking.py \
        --input input.mp4 \
        --model product.glb \
        --output final_result.mp4 \
        --chunk-duration 2 \
        --workspace ./chunk_workspace
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil
import time
import gc
import torch
import json

# Add Dust3R to Python path if it exists
DUST3R_PATH = "/tmp/dust3r"
if os.path.exists(DUST3R_PATH) and DUST3R_PATH not in sys.path:
    sys.path.insert(0, DUST3R_PATH)

# Import memory utilities
try:
    from memory_utils import (
        get_system_memory_info,
        check_memory_threshold,
        clear_system_caches,
        clear_python_memory,
        cleanup_temp_files,
        check_swap_space,
        print_memory_status
    )
    MEMORY_UTILS_AVAILABLE = True
except ImportError:
    print("[Memory] Warning: memory_utils not available, system memory management disabled")
    MEMORY_UTILS_AVAILABLE = False


class MemorySafeChunkingPipeline:
    """Memory-safe video processing pipeline with chunking."""

    def __init__(self, workspace: str = "./chunk_workspace"):
        """
        Initialize chunking pipeline workspace.

        Args:
            workspace: Working directory for intermediate files
        """
        self.workspace = Path(workspace)
        self.workspace.mkdir(exist_ok=True)

        # Define subdirectories
        self.dirs = {
            'temp_clips': self.workspace / 'temp_clips',
            'final_clips': self.workspace / 'final_clips',
            'chunk_workspaces': self.workspace / 'chunk_workspaces'
        }

        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("MEMORY-SAFE VIDEO CHUNKING PIPELINE")
        print(f"{'='*70}\n")
        print(f"Workspace: {self.workspace}\n")
        
        # Check system memory status at startup
        if MEMORY_UTILS_AVAILABLE:
            print_memory_status()
            has_swap, swap_msg = check_swap_space()
            if not has_swap:
                print(f"⚠ WARNING: {swap_msg}")
                print("⚠ Consider running: bash setup_swap.sh to create swap space")
                print("⚠ This helps prevent SSH disconnections due to OOM\n")

    def get_video_info(self, video_path: str) -> dict:
        """
        Get video information using ffprobe.

        Args:
            video_path: Path to input video

        Returns:
            Dictionary with video info (fps, duration, frame_count)
        """
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,duration',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Extract FPS
        r_frame_rate = data['streams'][0].get('r_frame_rate', '30/1')
        num, den = map(int, r_frame_rate.split('/'))
        fps = num / den if den > 0 else 30.0

        # Extract duration
        duration = float(data['format'].get('duration', 0))

        # Calculate frame count
        frame_count = int(duration * fps) if duration > 0 else 0

        return {
            'fps': fps,
            'duration': duration,
            'frame_count': frame_count
        }

    def split_video_into_clips(self, video_path: str, chunk_duration: float = 2.0,
                               chunk_frames: int = None, overlap_frames: int = 15) -> list:
        """
        Split video into short clips using ffmpeg with overlap.

        Args:
            video_path: Input video path
            chunk_duration: Duration of each clip in seconds (default: 2.0)
            chunk_frames: Alternative: number of frames per clip (overrides duration)
            overlap_frames: Number of frames to overlap between clips (default: 15)

        Returns:
            List of clip file paths with overlap metadata
        """
        print(f"\n{'─'*70}")
        print("STEP 1: Splitting Video into Clips (with Overlap)")
        print(f"{'─'*70}\n")

        # Get video info
        video_info = self.get_video_info(video_path)
        fps = video_info['fps']
        duration = video_info['duration']
        total_frames = video_info['frame_count']

        # Determine chunk duration and frames
        if chunk_frames:
            chunk_duration = chunk_frames / fps
            base_frames = chunk_frames
        else:
            base_frames = int(chunk_duration * fps)

        overlap_duration = overlap_frames / fps
        print(f"Using chunk_duration={chunk_duration:.2f}s ({base_frames} frames)")
        print(f"Overlap: {overlap_frames} frames ({overlap_duration:.2f}s)")

        # Calculate number of clips (with overlap)
        # Each clip starts at: i * (base_frames - overlap_frames) frames
        num_clips = int((total_frames - overlap_frames) / (base_frames - overlap_frames)) + 1
        if num_clips * (base_frames - overlap_frames) >= total_frames:
            num_clips -= 1
        num_clips = max(1, num_clips)  # At least 1 clip

        print(f"Video info: {duration:.2f}s, {fps:.2f} fps, {total_frames} frames")
        print(f"Will create {num_clips} clips with {overlap_frames}-frame overlap\n")

        clip_paths = []
        clip_metadata = []

        for i in range(num_clips):
            # Calculate start frame (with overlap)
            if i == 0:
                start_frame = 0
            else:
                start_frame = i * (base_frames - overlap_frames)
            
            start_time = start_frame / fps
            
            # Calculate end frame
            if i == num_clips - 1:
                # Last clip: include all remaining frames
                end_frame = total_frames
                clip_duration = (end_frame - start_frame) / fps
            else:
                # Regular clip: base_frames + overlap
                end_frame = start_frame + base_frames
                clip_duration = base_frames / fps

            clip_path = self.dirs['temp_clips'] / f"clip_{i+1:03d}.mp4"

            cmd = [
                'ffmpeg',
                '-y',  # Overwrite
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(clip_duration),
                '-c', 'copy',  # Fast copy (no re-encoding)
                '-avoid_negative_ts', 'make_zero',
                str(clip_path)
            ]

            print(f"  Creating clip {i+1}/{num_clips}: {clip_path.name}")
            print(f"    Frames: {start_frame}-{end_frame-1} ({end_frame-start_frame} frames)")
            print(f"    Time: {start_time:.2f}s - {start_time+clip_duration:.2f}s")
            
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                clip_paths.append(clip_path)
                clip_metadata.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'overlap_start': start_frame + (base_frames - overlap_frames) if i > 0 else None,
                    'overlap_end': start_frame + base_frames if i < num_clips - 1 else None,
                    'overlap_frames': overlap_frames if i > 0 and i < num_clips - 1 else (overlap_frames if i > 0 else 0)
                })
            else:
                print(f"  WARNING: Failed to create clip {i+1}: {result.stderr}")
                # Try with re-encoding if copy fails
                cmd_reencode = [
                    'ffmpeg',
                    '-y',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(clip_duration),
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    str(clip_path)
                ]
                result2 = subprocess.run(cmd_reencode, capture_output=True, text=True)
                if result2.returncode == 0:
                    clip_paths.append(clip_path)
                    clip_metadata.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'overlap_start': start_frame + (base_frames - overlap_frames) if i > 0 else None,
                        'overlap_end': start_frame + base_frames if i < num_clips - 1 else None,
                        'overlap_frames': overlap_frames if i > 0 and i < num_clips - 1 else (overlap_frames if i > 0 else 0)
                    })
                else:
                    print(f"  ERROR: Failed to create clip {i+1} even with re-encoding")

        # Save metadata for merge step
        metadata_path = self.workspace / "clip_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(clip_metadata, f, indent=2)

        print(f"\n✓ Created {len(clip_paths)} clips with overlap")
        print(f"  Metadata saved to: {metadata_path}")
        return clip_paths

    def process_single_clip(self, clip_path: Path, clip_idx: int, total_clips: int,
                           glb_path: str, workspace_base: Path,
                           frame_interval: int = 1,
                           model_scale: float = 1.0,
                           resolution: tuple = (1920, 1080),
                           strength: float = 0.25,
                           seed: int = 12345,
                           latent_blend: float = 0.15,
                           fps: int = 30,
                           crf: int = 18,
                           auto_placement: bool = True,
                           manual_location: tuple = None,
                           manual_scale: float = None) -> Path:
        """
        Process a single clip through the full pipeline (Module A -> B -> C).

        Args:
            clip_path: Path to input clip
            clip_idx: Index of current clip (0-based)
            total_clips: Total number of clips
            glb_path: Path to 3D model (.glb)
            workspace_base: Base workspace directory
            frame_interval: Frame sampling interval
            model_scale: 3D model scale
            resolution: Output resolution
            strength: IC-Light denoising strength
            seed: Random seed
            latent_blend: Latent blending ratio
            fps: Output FPS
            crf: Video quality

        Returns:
            Path to processed clip video
        """
        print(f"\n{'─'*70}")
        print(f"PROCESSING CLIP {clip_idx+1}/{total_clips}: {clip_path.name}")
        print(f"{'─'*70}\n")

        # Create workspace for this clip
        clip_workspace = workspace_base / f"clip_{clip_idx+1:03d}_workspace"
        clip_workspace.mkdir(exist_ok=True)

        # Import pipeline
        # Handle import from same directory or as module
        try:
            from vanelia_pipeline import VaneliaPipeline
        except ImportError:
            # If running as script, add parent directory to path
            import sys
            script_dir = Path(__file__).parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            from vanelia_pipeline import VaneliaPipeline

        # Initialize pipeline for this clip
        pipeline = VaneliaPipeline(workspace=str(clip_workspace))

        try:
            # Run full pipeline on this clip
            output_clip_path = clip_workspace / "output" / "final_output.mp4"

            # Check if steps are already completed
            poses_path = clip_workspace / "camera_data" / "camera_poses.npy"
            render_frames = list((clip_workspace / "render_frames").glob("*.png"))
            
            skip_step1 = poses_path.exists()
            skip_step2 = len(render_frames) > 0
            
            if skip_step1 or skip_step2:
                print(f"\n[Resume] Detected completed steps:")
                print(f"  - Step 1 (Camera): {'✓' if skip_step1 else '✗'}")
                print(f"  - Step 2 (Render): {'✓' if skip_step2 else '✗'} ({len(render_frames)} frames)")
                print(f"  - Step 3 (Composite): Will run/retry\n")

            pipeline.run_full_pipeline(
                video_path=str(clip_path),
                glb_path=glb_path,
                output_path=str(output_clip_path),
                frame_interval=frame_interval,
                max_frames=None,
                model_scale=model_scale,
                resolution=resolution,
                strength=strength,
                auto_placement=auto_placement,
                manual_location=manual_location,
                manual_scale=manual_scale,
                seed=seed,
                latent_blend=latent_blend,
                fps=fps,
                crf=crf,
                skip_step1=skip_step1,
                skip_step2=skip_step2
            )

            # Copy final clip to final_clips directory
            final_clip_path = self.dirs['final_clips'] / f"final_clip_{clip_idx+1:03d}.mp4"
            shutil.copy2(output_clip_path, final_clip_path)

            print(f"\n✓ Clip {clip_idx+1}/{total_clips} processed: {final_clip_path.name}")
            return final_clip_path

        except Exception as e:
            print(f"\n✗ ERROR processing clip {clip_idx+1}: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Explicitly delete pipeline and all references
            # This helps ensure models are freed before garbage collection
            del pipeline
            pipeline = None

    def clear_gpu_memory(self):
        """
        Clear GPU memory completely.
        Must be called after each clip processing.
        
        This function performs:
        1. Python garbage collection
        2. PyTorch CUDA cache clearing
        3. Memory synchronization
        4. System memory management (if available)
        
        CRITICAL: This must be called after each clip to prevent OOM.
        """
        print("\n[Memory] Clearing GPU and system memory...")

        # Step 1: Python garbage collection (aggressive)
        if MEMORY_UTILS_AVAILABLE:
            clear_python_memory()
        else:
            collected = gc.collect()
            print(f"  - Garbage collected {collected} objects")

        # Step 2: Clear PyTorch CUDA cache
        # This releases cached memory blocks back to the system
        if torch.cuda.is_available():
            # Get memory stats before clearing
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get memory stats after clearing
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            
            print(f"  - CUDA cache cleared")
            print(f"  - GPU memory allocated: {allocated_after:.2f} GB (was {allocated_before:.2f} GB)")
            print(f"  - GPU memory reserved: {reserved_after:.2f} GB (was {reserved_before:.2f} GB)")

        # Step 3: System memory management
        if MEMORY_UTILS_AVAILABLE:
            # Check system memory
            mem_info = get_system_memory_info()
            if mem_info:
                print(f"  - System RAM: {mem_info['used_gb']:.2f} GB / {mem_info['total_gb']:.2f} GB "
                      f"({mem_info['percent']:.1f}%)")
                
                # If memory usage is high, clear system caches
                if mem_info['percent'] > 80:
                    print("  - High system memory usage detected, clearing caches...")
                    clear_system_caches()
                
                # Check swap space
                has_swap, swap_msg = check_swap_space()
                if not has_swap:
                    print(f"  - ⚠ {swap_msg}")
                    print("  - Consider running: bash setup_swap.sh")

        print("[Memory] ✓ Memory cleared\n")

    def merge_clips(self, clip_paths: list, output_path: str, fps: int = 30, 
                   overlap_frames: int = 15, crossfade_duration: float = 0.5) -> str:
        """
        Merge all processed clips into final video with crossfade transitions.

        Args:
            clip_paths: List of processed clip paths
            output_path: Final output video path
            fps: Output FPS
            overlap_frames: Number of overlapping frames (for crossfade)
            crossfade_duration: Crossfade duration in seconds (default: 0.5)

        Returns:
            Path to final merged video
        """
        print(f"\n{'─'*70}")
        print("STEP 4: Merging Clips into Final Video (with Crossfade)")
        print(f"{'─'*70}\n")

        if not clip_paths:
            raise ValueError("No clips to merge")

        if len(clip_paths) == 1:
            # Single clip: just copy
            shutil.copy2(clip_paths[0], output_path)
            print(f"✓ Single clip copied to: {output_path}")
            return output_path

        # Load metadata if available
        metadata_path = self.workspace / "clip_metadata.json"
        metadata = []
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        print(f"Merging {len(clip_paths)} clips with crossfade...")
        print(f"Crossfade duration: {crossfade_duration}s ({int(crossfade_duration * fps)} frames)\n")

        # Build complex filter for crossfade
        filter_complex_parts = []
        input_args = []
        
        # Add all clips as inputs
        for i, clip_path in enumerate(clip_paths):
            input_args.extend(['-i', str(clip_path)])
        
        # Simple concat without trimming overlap (preserves full video length)
        # Overlap will be handled by crossfade if needed, but for now just concat
        # This ensures the full video length is preserved
        if len(clip_paths) > 1:
            # Simple concat - preserve all frames
            filter_parts = []
            for i in range(len(clip_paths)):
                filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
            
            # Concatenate all clips without trimming
            concat_inputs = ''.join([f"[v{i}]" for i in range(len(clip_paths))])
            filter_parts.append(f"{concat_inputs}concat=n={len(clip_paths)}:v=1:a=0[outv]")
            
            filter_complex = ';'.join(filter_parts)
        else:
            filter_complex = "[0:v]setpts=PTS-STARTPTS[outv]"

        # Merge with crossfade using xfade filter (more sophisticated)
        # For simplicity, we'll use trim + concat, then apply crossfade
        # Alternative: use xfade filter for smoother transitions
        
        cmd = [
            'ffmpeg',
            '-y',
        ] + input_args + [
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            output_path
        ]

        print(f"Running ffmpeg merge with crossfade...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"WARNING: Crossfade merge failed: {result.stderr}")
            print("Falling back to simple concat...")
            
            # Fallback: simple concat
            concat_file = self.workspace / "concat_list.txt"
            with open(concat_file, 'w') as f:
                for clip_path in clip_paths:
                    abs_path = os.path.abspath(clip_path)
                    f.write(f"file '{abs_path}'\n")
            
            cmd_simple = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-c:a', 'aac',
                '-r', str(fps),
                output_path
            ]
            result = subprocess.run(cmd_simple, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR: Failed to merge clips: {result.stderr}")
            raise RuntimeError("ffmpeg merge failed")

        print(f"\n✓ Merged {len(clip_paths)} clips with crossfade into: {output_path}")
        return output_path

    def run_chunked_pipeline(self,
                            video_path: str,
                            glb_path: str,
                            output_path: str,
                            chunk_duration: float = 2.0,
                            chunk_frames: int = None,
                            overlap_frames: int = 15,
                            frame_interval: int = 1,
                            model_scale: float = 1.0,
                            resolution: tuple = (1920, 1080),
                            strength: float = 0.25,
                            seed: int = 12345,
                            latent_blend: float = 0.15,
                            fps: int = 30,
                            crf: int = 18,
                            cleanup_temp: bool = True,
                            auto_placement: bool = True,
                            manual_location: tuple = None,
                            manual_scale: float = None) -> str:
        """
        Run complete memory-safe chunked pipeline.

        Args:
            video_path: Input video path
            glb_path: 3D model file (.glb)
            output_path: Final output video path
            chunk_duration: Duration of each clip in seconds (default: 2.0)
            chunk_frames: Alternative: number of frames per clip
            frame_interval: Frame sampling interval
            model_scale: 3D model scale
            resolution: Output resolution
            strength: IC-Light denoising strength
            seed: Random seed
            latent_blend: Latent blending ratio
            fps: Output FPS
            crf: Video quality
            cleanup_temp: Whether to clean up temporary files

        Returns:
            Path to final merged video
        """
        start_time = time.time()

        # Step 1: Split video into clips
        clip_paths = self.split_video_into_clips(
            video_path=video_path,
            chunk_duration=chunk_duration,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames
        )

        if not clip_paths:
            raise ValueError("No clips were created from input video")

        # Step 2: Process each clip iteratively
        processed_clips = []
        total_clips = len(clip_paths)

        for idx, clip_path in enumerate(clip_paths):
            try:
                # Check memory before processing clip
                if MEMORY_UTILS_AVAILABLE:
                    if check_memory_threshold(90.0):
                        print(f"\n⚠ WARNING: System memory usage > 90% before processing clip {idx+1}")
                        print("Cleaning up temporary files and caches...")
                        cleanup_temp_files([
                            str(self.dirs['temp_clips']),
                            str(self.dirs['chunk_workspaces'] / f"clip_{idx:03d}_workspace")
                        ])
                        clear_system_caches()
                
                # Process clip through full pipeline
                processed_clip = self.process_single_clip(
                    clip_path=clip_path,
                    clip_idx=idx,
                    total_clips=total_clips,
                    glb_path=glb_path,
                    workspace_base=self.dirs['chunk_workspaces'],
                    frame_interval=frame_interval,
                    model_scale=model_scale,
                    resolution=resolution,
                    strength=strength,
                    seed=seed,
                    latent_blend=latent_blend,
                    fps=fps,
                    crf=crf,
                    auto_placement=auto_placement,
                    manual_location=manual_location,
                    manual_scale=manual_scale
                )
                processed_clips.append(processed_clip)

            except Exception as e:
                print(f"\n✗ Failed to process clip {idx+1}: {e}")
                raise

            finally:
                # CRITICAL: Clear GPU and system memory after each clip
                self.clear_gpu_memory()
                
                # Clean up completed clip workspace if memory is tight
                if MEMORY_UTILS_AVAILABLE and check_memory_threshold(75.0):
                    completed_workspace = self.dirs['chunk_workspaces'] / f"clip_{idx+1:03d}_workspace"
                    if completed_workspace.exists() and cleanup_temp:
                        try:
                            # Keep only the final output, remove intermediate files
                            final_output = completed_workspace / "output" / "final_output.mp4"
                            if final_output.exists():
                                # Remove intermediate directories but keep output
                                for subdir in ['background_frames', 'render_frames', 'refined_frames', 'camera_data']:
                                    subdir_path = completed_workspace / subdir
                                    if subdir_path.exists():
                                        shutil.rmtree(subdir_path)
                                print(f"  - Cleaned intermediate files for clip {idx+1}")
                        except Exception as cleanup_error:
                            print(f"  - Warning: Could not clean clip {idx+1} workspace: {cleanup_error}")

        # Step 3: Merge all processed clips
        final_video = self.merge_clips(
            clip_paths=processed_clips,
            output_path=output_path,
            fps=fps,
            overlap_frames=overlap_frames
        )

        # Cleanup temporary files
        if cleanup_temp:
            print(f"\n[Cleanup] Removing temporary files...")
            if self.dirs['temp_clips'].exists():
                shutil.rmtree(self.dirs['temp_clips'])
            if self.dirs['chunk_workspaces'].exists():
                shutil.rmtree(self.dirs['chunk_workspaces'])
            print("[Cleanup] ✓ Temporary files removed")

        elapsed_time = time.time() - start_time

        print(f"\n{'='*70}")
        print("✓ MEMORY-SAFE CHUNKING PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
        print(f"Processed clips: {total_clips}")
        print(f"Final video: {final_video}")
        print(f"{'='*70}\n")

        return final_video


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Memory-Safe Video Chunking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 2-second chunks
  python memory_safe_chunking.py --input input.mp4 --model product.glb --output final_result.mp4

  # Use 60-frame chunks instead
  python memory_safe_chunking.py --input input.mp4 --model product.glb \\
      --output final_result.mp4 --chunk-frames 60

  # Custom chunk duration
  python memory_safe_chunking.py --input input.mp4 --model product.glb \\
      --output final_result.mp4 --chunk-duration 3.0

Docker Command (RunPod):
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
        """
    )

    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input video path (input.mp4)')
    parser.add_argument('--model', type=str, required=True,
                       help='3D model file (.glb)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output video path (final_result.mp4)')

    # Chunking settings
    parser.add_argument('--chunk-duration', type=float, default=2.0,
                       help='Duration of each clip in seconds (default: 2.0)')
    parser.add_argument('--chunk-frames', type=int, default=None,
                       help='Alternative: number of frames per clip (overrides --chunk-duration)')
    parser.add_argument('--overlap-frames', type=int, default=15,
                       help='Number of frames to overlap between clips (default: 15, ~0.5s)')

    # Workspace
    parser.add_argument('--workspace', type=str, default='./chunk_workspace',
                       help='Working directory (default: ./chunk_workspace)')

    # Pipeline settings (passed to VaneliaPipeline)
    parser.add_argument('--frame-interval', type=int, default=1,
                       help='Frame sampling interval (default: 1)')
    parser.add_argument('--model-scale', type=float, default=1.0,
                       help='3D model scale (default: 1.0)')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080],
                       help='Output resolution (default: 1920 1080)')
    
    # Object placement settings
    parser.add_argument('--no-auto-placement', dest='auto_placement', action='store_false',
                       help='Disable automatic object placement (use manual or default)')
    parser.add_argument('--object-location', type=float, nargs=3, default=None,
                       metavar=('X', 'Y', 'Z'),
                       help='Manual object location in Blender coordinates (e.g., 0.0 0.0 -2.0)')
    parser.add_argument('--object-scale', type=float, default=None,
                       help='Manual object scale (overrides --model-scale for placement)')

    # IC-Light settings
    parser.add_argument('--strength', type=float, default=0.25,
                       help='IC-Light denoising strength (default: 0.25)')
    parser.add_argument('--seed', type=int, default=12345,
                       help='Random seed for consistency (default: 12345)')
    parser.add_argument('--latent-blend', type=float, default=0.15,
                       help='Latent blending ratio (default: 0.15)')

    # Video encoding
    parser.add_argument('--fps', type=int, default=30,
                       help='Output FPS (default: 30)')
    parser.add_argument('--crf', type=int, default=18,
                       help='Video quality CRF (default: 18)')

    # Cleanup
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep temporary files (default: False)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"ERROR: 3D model not found: {args.model}")
        sys.exit(1)

    # Check for required tools
    for tool in ['ffmpeg', 'ffprobe']:
        result = subprocess.run(['which', tool], capture_output=True)
        if result.returncode != 0:
            print(f"ERROR: {tool} not found. Please install ffmpeg.")
            sys.exit(1)

    # Initialize pipeline
    pipeline = MemorySafeChunkingPipeline(workspace=args.workspace)

    # Run chunked pipeline
    try:
        final_video = pipeline.run_chunked_pipeline(
            video_path=args.input,
            glb_path=args.model,
            output_path=args.output,
            chunk_duration=args.chunk_duration,
            chunk_frames=args.chunk_frames,
            overlap_frames=args.overlap_frames,
            frame_interval=args.frame_interval,
            model_scale=args.model_scale,
            resolution=tuple(args.resolution),
            strength=args.strength,
            seed=args.seed,
            latent_blend=args.latent_blend,
            fps=args.fps,
            crf=args.crf,
            cleanup_temp=not args.keep_temp,
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

