"""
Module A: Dust3R Camera Pose Extraction
Extracts camera poses and intrinsics from video using Dust3R.
Converts OpenCV coordinate system to Blender coordinate system.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, List, Tuple
import sys

# Add Dust3R to path (clone from: https://github.com/naver/dust3r)
# Assuming dust3r is installed in the environment or path
try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
except ImportError:
    print("ERROR: Dust3R not found. Please install from https://github.com/naver/dust3r")
    sys.exit(1)


class Dust3RCameraExtractor:
    """Extract camera poses from video using Dust3R model."""

    def __init__(self, model_path: str = None, device: str = "cuda"):
        """
        Initialize Dust3R model.

        Args:
            model_path: Path to Dust3R checkpoint (if None, uses default)
            device: Device to run inference on
        """
        self.device = device

        # Load Dust3R model
        print(f"[Dust3R] Loading model on {device}...")
        if model_path is None:
            # Use default model (will download if not exists)
            model_path = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
        print("[Dust3R] Model loaded successfully")

    def extract_frames(self, video_path: str, output_dir: str,
                      frame_interval: int = 1, max_frames: int = None) -> List[str]:
        """
        Extract frames from video.

        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            frame_interval: Extract every Nth frame (1 = all frames)
            max_frames: Maximum number of frames to extract

        Returns:
            List of frame paths
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_idx = 0
        extracted_count = 0

        print(f"[Extract] Extracting frames from {video_path}...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                extracted_count += 1

                if max_frames and extracted_count >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        print(f"[Extract] Extracted {len(frame_paths)} frames")
        return frame_paths

    def opencv_to_blender_matrix(self, opencv_matrix: np.ndarray) -> np.ndarray:
        """
        Convert OpenCV camera matrix to Blender coordinate system.

        OpenCV: Right (+X), Down (+Y), Forward (+Z)
        Blender: Right (+X), Up (+Z), Back (-Y)

        Transformation:
        X_blender = X_opencv
        Y_blender = -Z_opencv
        Z_blender = Y_opencv

        Args:
            opencv_matrix: 4x4 camera extrinsic matrix in OpenCV format

        Returns:
            4x4 camera matrix in Blender format
        """
        # Coordinate system conversion matrix
        # This matrix transforms from OpenCV to Blender coordinates
        T_blender_from_opencv = np.array([
            [1,  0,  0, 0],  # X stays the same
            [0,  0,  1, 0],  # Y becomes Z
            [0, -1,  0, 0],  # Z becomes -Y
            [0,  0,  0, 1]
        ], dtype=np.float32)

        # Apply transformation: C_blender = T * C_opencv
        blender_matrix = T_blender_from_opencv @ opencv_matrix

        return blender_matrix

    def run_dust3r_inference(self, frame_paths: List[str],
                            batch_size: int = 4) -> Dict:
        """
        Run Dust3R inference on frames.

        Args:
            frame_paths: List of frame image paths
            batch_size: Batch size for inference

        Returns:
            Dictionary containing camera poses and point cloud
        """
        print(f"[Dust3R] Running inference on {len(frame_paths)} frames...")

        # Load images
        images = load_images(frame_paths, size=512)

        # Create pairs for reconstruction
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

        # Run inference
        output = inference(pairs, self.model, self.device, batch_size=batch_size)

        # Global alignment to get camera poses
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)

        print(f"[Dust3R] Global alignment loss: {loss:.4f}")

        # Extract camera parameters
        cameras = scene.get_im_poses()  # Camera poses (world to camera)
        intrinsics = scene.get_intrinsics()  # Camera intrinsics
        pts3d = scene.get_pts3d()  # 3D points

        return {
            'cameras': cameras,
            'intrinsics': intrinsics,
            'pts3d': pts3d,
            'frame_paths': frame_paths
        }

    def process_video(self, video_path: str, output_dir: str,
                     frame_interval: int = 1, max_frames: int = None) -> Dict:
        """
        Complete pipeline: extract frames → run Dust3R → convert coordinates.

        Args:
            video_path: Path to input video
            output_dir: Directory for output files
            frame_interval: Frame sampling interval
            max_frames: Maximum frames to process

        Returns:
            Dictionary with camera data in Blender coordinates
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract background frames
        bg_dir = output_path / "background_frames"
        frame_paths = self.extract_frames(video_path, str(bg_dir),
                                         frame_interval, max_frames)

        # Run Dust3R
        result = self.run_dust3r_inference(frame_paths)

        # Convert camera poses to Blender coordinate system
        cameras_opencv = result['cameras'].cpu().numpy()
        intrinsics = result['intrinsics'].cpu().numpy()

        cameras_blender = []
        for cam_matrix in cameras_opencv:
            # Dust3R returns camera-to-world matrix, convert to 4x4 if needed
            if cam_matrix.shape == (3, 4):
                cam_matrix_4x4 = np.vstack([cam_matrix, [0, 0, 0, 1]])
            else:
                cam_matrix_4x4 = cam_matrix

            # Convert to Blender coordinates
            blender_cam = self.opencv_to_blender_matrix(cam_matrix_4x4)
            cameras_blender.append(blender_cam)

        cameras_blender = np.array(cameras_blender)

        # Save results
        poses_path = output_path / "camera_poses.npy"
        intrinsics_path = output_path / "camera_intrinsics.npy"
        metadata_path = output_path / "camera_metadata.json"

        np.save(poses_path, cameras_blender)
        np.save(intrinsics_path, intrinsics)

        # Save metadata
        metadata = {
            'num_frames': len(frame_paths),
            'video_path': video_path,
            'frame_interval': frame_interval,
            'camera_poses_shape': cameras_blender.shape,
            'intrinsics_shape': intrinsics.shape,
            'coordinate_system': 'Blender (Right, Up, Back)',
            'frame_paths': frame_paths
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[DONE] Camera extraction complete!")
        print(f"  - Poses saved to: {poses_path}")
        print(f"  - Intrinsics saved to: {intrinsics_path}")
        print(f"  - Metadata saved to: {metadata_path}")
        print(f"  - Coordinate system: Blender (with T_blender_from_opencv conversion)")

        return {
            'cameras_blender': cameras_blender,
            'intrinsics': intrinsics,
            'frame_paths': frame_paths,
            'metadata': metadata
        }


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract camera poses from video using Dust3R")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--interval", type=int, default=1, help="Frame sampling interval")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--model", type=str, default=None, help="Dust3R model path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Initialize extractor
    extractor = Dust3RCameraExtractor(model_path=args.model, device=args.device)

    # Process video
    result = extractor.process_video(
        video_path=args.input,
        output_dir=args.output,
        frame_interval=args.interval,
        max_frames=args.max_frames
    )

    print(f"\n✓ Extracted {len(result['frame_paths'])} camera poses")


if __name__ == "__main__":
    main()
