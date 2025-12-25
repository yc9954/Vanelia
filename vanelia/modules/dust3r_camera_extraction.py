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

        try:
            self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
            print("[Dust3R] ✓ Model loaded successfully")
        except Exception as e:
            print(f"[Dust3R] ⚠ Failed to load model from {model_path}")
            print(f"[Dust3R] Error: {str(e)[:100]}")
            print("[Dust3R] Retrying with trust_remote_code=True...")
            try:
                self.model = AsymmetricCroCo3DStereo.from_pretrained(
                    model_path,
                    trust_remote_code=True
                ).to(device)
                print("[Dust3R] ✓ Model loaded (alternative method)")
            except Exception as e2:
                print(f"[Dust3R] ✗ Failed to load model: {e2}")
                raise RuntimeError(f"Could not load Dust3R model. Please check installation.") from e2

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

    def opencv_to_blender_matrix(self, opencv_c2w: np.ndarray) -> np.ndarray:
        """
        Convert Dust3R camera-to-world matrix to Blender world-to-camera.

        Dust3R outputs: Camera-to-World (c2w) in OpenCV convention
        Blender needs: World-to-Camera (w2c) in OpenGL convention

        OpenCV: Right (+X), Down (+Y), Forward (+Z)
        Blender: Right (+X), Up (+Z), Back (-Y)

        Steps:
        1. Convert OpenCV c2w to Blender c2w (coordinate change)
        2. Invert to get Blender w2c (what Blender needs)

        Args:
            opencv_c2w: 4x4 camera-to-world matrix from Dust3R (OpenCV)

        Returns:
            4x4 world-to-camera matrix in Blender format
        """
        # Step 1: OpenCV → Blender coordinate conversion
        # This changes the axes while keeping c2w form
        T_blender_from_opencv = np.array([
            [1,  0,  0, 0],  # X stays the same
            [0,  0,  1, 0],  # Y becomes Z
            [0, -1,  0, 0],  # Z becomes -Y
            [0,  0,  0, 1]
        ], dtype=np.float32)

        # Convert to Blender coordinate system (still c2w)
        blender_c2w = T_blender_from_opencv @ opencv_c2w @ np.linalg.inv(T_blender_from_opencv)

        # Step 2: Invert to get world-to-camera (what Blender camera needs)
        blender_w2c = np.linalg.inv(blender_c2w)

        return blender_w2c

    def detect_ground_plane(self, point_cloud: np.ndarray,
                           ransac_threshold: float = 0.05,
                           max_iterations: int = 1000) -> Dict:
        """
        Detect ground plane from point cloud using RANSAC.

        Plane equation: Ax + By + Cz + D = 0

        Args:
            point_cloud: Nx3 array of 3D points
            ransac_threshold: Distance threshold for inliers
            max_iterations: RANSAC iterations

        Returns:
            Dictionary with plane parameters {A, B, C, D, inliers, normal}
        """
        if len(point_cloud) < 3:
            return None

        print(f"[RANSAC] Detecting ground plane from {len(point_cloud)} points...")

        best_plane = None
        best_inliers = 0

        for _ in range(max_iterations):
            # Randomly sample 3 points
            idx = np.random.choice(len(point_cloud), 3, replace=False)
            p1, p2, p3 = point_cloud[idx]

            # Compute plane normal using cross product
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal_len = np.linalg.norm(normal)

            if normal_len < 1e-6:
                continue

            normal = normal / normal_len  # Normalize

            # Plane equation: Ax + By + Cz + D = 0
            # where (A, B, C) is the normal and D = -dot(normal, point)
            A, B, C = normal
            D = -np.dot(normal, p1)

            # Compute distances from all points to plane
            distances = np.abs(
                A * point_cloud[:, 0] +
                B * point_cloud[:, 1] +
                C * point_cloud[:, 2] + D
            )

            # Count inliers
            inliers = np.sum(distances < ransac_threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = {
                    'A': float(A),
                    'B': float(B),
                    'C': float(C),
                    'D': float(D),
                    'normal': normal.tolist(),
                    'inliers': int(inliers),
                    'inlier_ratio': float(inliers / len(point_cloud))
                }

        if best_plane:
            print(f"[RANSAC] ✓ Ground plane found: {best_inliers}/{len(point_cloud)} inliers "
                  f"({best_plane['inlier_ratio']*100:.1f}%)")
            print(f"[RANSAC] Plane equation: {best_plane['A']:.3f}x + {best_plane['B']:.3f}y + "
                  f"{best_plane['C']:.3f}z + {best_plane['D']:.3f} = 0")
        else:
            print("[RANSAC] WARNING: No ground plane detected")

        return best_plane

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
        pts3d = result['pts3d']  # Keep for point cloud

        cameras_blender = []
        for cam_matrix in cameras_opencv:
            # Dust3R returns camera-to-world matrix, convert to 4x4 if needed
            if cam_matrix.shape == (3, 4):
                cam_matrix_4x4 = np.vstack([cam_matrix, [0, 0, 0, 1]])
            else:
                cam_matrix_4x4 = cam_matrix

            # Convert to Blender coordinates (now properly inverted)
            blender_cam = self.opencv_to_blender_matrix(cam_matrix_4x4)
            cameras_blender.append(blender_cam)

        cameras_blender = np.array(cameras_blender)

        # Extract and save point cloud
        print("[Dust3R] Extracting point cloud...")
        if isinstance(pts3d, torch.Tensor):
            pts3d_np = pts3d.cpu().numpy()
        else:
            pts3d_np = pts3d

        # Flatten point cloud from all views
        points_list = []
        for i, pts in enumerate(pts3d_np):
            if len(pts.shape) == 3:  # (H, W, 3)
                pts_flat = pts.reshape(-1, 3)
                # Remove invalid points (NaN, Inf)
                valid_mask = np.all(np.isfinite(pts_flat), axis=1)
                points_list.append(pts_flat[valid_mask])

        if points_list:
            point_cloud = np.vstack(points_list)
            print(f"[Dust3R] Point cloud size: {len(point_cloud)} points")
        else:
            point_cloud = np.array([])
            print("[Dust3R] WARNING: No valid point cloud extracted")

        # Detect ground plane using RANSAC
        ground_plane = self.detect_ground_plane(point_cloud) if len(point_cloud) > 100 else None

        # Save results
        poses_path = output_path / "camera_poses.npy"
        intrinsics_path = output_path / "camera_intrinsics.npy"
        pointcloud_path = output_path / "point_cloud.npy"
        metadata_path = output_path / "camera_metadata.json"

        np.save(poses_path, cameras_blender)
        np.save(intrinsics_path, intrinsics)
        np.save(pointcloud_path, point_cloud)

        # Save metadata
        metadata = {
            'num_frames': len(frame_paths),
            'video_path': video_path,
            'frame_interval': frame_interval,
            'camera_poses_shape': cameras_blender.shape,
            'intrinsics_shape': intrinsics.shape,
            'point_cloud_size': int(len(point_cloud)),
            'coordinate_system': 'Blender (Right, Up, Back)',
            'coordinate_conversion': 'Camera-to-World → World-to-Camera (inverted)',
            'ground_plane': ground_plane,
            'frame_paths': frame_paths
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[DONE] Camera extraction complete!")
        print(f"  - Poses saved to: {poses_path}")
        print(f"  - Intrinsics saved to: {intrinsics_path}")
        print(f"  - Point cloud saved to: {pointcloud_path}")
        print(f"  - Metadata saved to: {metadata_path}")
        print(f"  - Coordinate system: Blender W2C (properly inverted from Dust3R C2W)")

        return {
            'cameras_blender': cameras_blender,
            'intrinsics': intrinsics,
            'point_cloud': point_cloud,
            'ground_plane': ground_plane,
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
