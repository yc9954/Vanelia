"""
Module A: MUSt3R Camera Pose Extraction
Extracts camera poses and intrinsics from video using MUSt3R.
MUSt3R provides better temporal consistency than Dust3R for video sequences.
Converts OpenCV coordinate system to Blender coordinate system.

Based on MUSt3R: Multi-view Network for Stereo 3D Reconstruction
Repository: https://github.com/naver/must3r
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, List, Tuple
import sys

# Add MUSt3R to path - try common installation paths
must3r_paths = [
    "/workspace/must3r",
    "/home/user/must3r",
    os.path.join(os.path.dirname(__file__), "../../../must3r"),
    os.path.join(os.path.dirname(__file__), "../../must3r"),
]

for must3r_path in must3r_paths:
    if os.path.exists(must3r_path) and os.path.isdir(must3r_path):
        if must3r_path not in sys.path:
            sys.path.insert(0, must3r_path)
            print(f"[MUSt3R] Added to path: {must3r_path}")
        break

# Import MUSt3R
try:
    from must3r.demo.inference import must3r_inference_video
    from must3r.model import get_pointmaps_activation
    import must3r.tools.path_to_dust3r  # This adds dust3r to path
    from dust3r.datasets import ImgNorm
except ImportError as e:
    print(f"ERROR: MUSt3R not found: {e}")
    print("Tried paths:", must3r_paths)
    print("\nPlease install MUSt3R:")
    print("  cd /home/user")
    print("  git clone --recursive https://github.com/naver/must3r.git")
    print("  cd must3r")
    print("  pip install -e .")
    sys.exit(1)


class MUSt3RCameraExtractor:
    """Extract camera poses from video using MUSt3R model with temporal consistency."""

    def __init__(self, model_name: str = None, device: str = "cuda"):
        """
        Initialize MUSt3R model.

        Args:
            model_name: MUSt3R model name (default: uses recommended checkpoint)
            device: Device to run inference on
        """
        self.device = device

        # Load MUSt3R model
        print(f"[MUSt3R] Loading model on {device}...")
        if model_name is None:
            # Use default MUSt3R model
            model_name = "naver/MUSt3R_ViTLarge_BaseDecoder_512_dpt"

        try:
            # Import model loading
            from must3r.model import AsymmetricMASt3R
            from dust3r.model import AsymmetricCroCo3DStereo

            # Try loading MUSt3R
            try:
                self.model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
                print(f"[MUSt3R] ✓ Model loaded: {model_name}")
            except:
                # Fallback to simpler loading
                print(f"[MUSt3R] Trying alternative loading method...")
                from huggingface_hub import hf_hub_download

                # Download checkpoint
                checkpoint = hf_hub_download(
                    repo_id="naver/MUSt3R_ViTLarge_BaseDecoder_512_dpt",
                    filename="checkpoint-best.pth"
                )

                # Load with torch
                self.model = torch.load(checkpoint, map_location=device)
                print(f"[MUSt3R] ✓ Model loaded from checkpoint")

            # Split into encoder and decoder for must3r_inference_video
            if hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
                self.encoder = self.model.encoder
                self.decoder = self.model.decoder
            else:
                # Model is already tuple or single model
                if isinstance(self.model, tuple):
                    self.encoder, self.decoder = self.model
                else:
                    self.encoder = self.decoder = self.model

        except Exception as e:
            print(f"[MUSt3R] ✗ Failed to load model: {e}")
            raise RuntimeError(f"Could not load MUSt3R model. Please check installation.") from e

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
        Convert MUSt3R camera-to-world matrix to Blender world-to-camera.

        MUSt3R outputs: Camera-to-World (c2w) in OpenCV convention
        Blender needs: World-to-Camera (w2c) in OpenGL convention

        OpenCV: Right (+X), Down (+Y), Forward (+Z)
        Blender: Right (+X), Up (+Z), Back (-Y)

        Args:
            opencv_c2w: 4x4 camera-to-world matrix from MUSt3R (OpenCV)

        Returns:
            4x4 world-to-camera matrix in Blender format
        """
        # OpenCV → Blender coordinate conversion
        T_blender_from_opencv = np.array([
            [1,  0,  0, 0],  # X stays the same
            [0,  0,  1, 0],  # Y becomes Z
            [0, -1,  0, 0],  # Z becomes -Y
            [0,  0,  0, 1]
        ], dtype=np.float32)

        # Convert to Blender coordinate system (still c2w)
        blender_c2w = T_blender_from_opencv @ opencv_c2w @ np.linalg.inv(T_blender_from_opencv)

        # Invert to get world-to-camera (what Blender camera needs)
        blender_w2c = np.linalg.inv(blender_c2w)

        return blender_w2c

    def detect_ground_plane(self, point_cloud: np.ndarray,
                           ransac_threshold: float = 0.05,
                           max_iterations: int = 1000) -> Dict:
        """
        Detect ground plane from point cloud using RANSAC.

        Args:
            point_cloud: Nx3 array of 3D points
            ransac_threshold: Distance threshold for inliers
            max_iterations: RANSAC iterations

        Returns:
            Dictionary with plane parameters
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

            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal_len = np.linalg.norm(normal)

            if normal_len < 1e-6:
                continue

            normal = normal / normal_len

            # Compute D (plane offset)
            D = -np.dot(normal, p1)

            # Count inliers
            distances = np.abs(np.dot(point_cloud, normal) + D)
            inlier_mask = distances < ransac_threshold
            num_inliers = np.sum(inlier_mask)

            if num_inliers > best_inliers:
                best_inliers = num_inliers
                A, B, C = normal
                best_plane = {
                    'A': float(A),
                    'B': float(B),
                    'C': float(C),
                    'D': float(D),
                    'normal': normal,
                    'inliers': num_inliers,
                    'inlier_ratio': num_inliers / len(point_cloud)
                }

        if best_plane:
            print(f"[RANSAC] ✓ Ground plane found: {best_inliers}/{len(point_cloud)} inliers "
                  f"({best_plane['inlier_ratio']*100:.1f}%)")
        else:
            print("[RANSAC] WARNING: No ground plane detected")

        return best_plane

    def run_must3r_inference(self, frame_paths: List[str],
                            max_bs: int = 8,
                            init_num_images: int = 8,
                            batch_num_views: int = 4,
                            local_context_size: int = 25) -> Dict:
        """
        Run MUSt3R inference on video frames with temporal consistency.

        Args:
            frame_paths: List of frame image paths (in temporal order)
            max_bs: Maximum batch size for inference
            init_num_images: Number of initial keyframes
            batch_num_views: Batch size for processing additional frames
            local_context_size: Size of local temporal context window

        Returns:
            Dictionary containing camera poses and point cloud
        """
        print(f"[MUSt3R] Running video inference on {len(frame_paths)} frames...")
        print(f"[MUSt3R] Temporal context size: {local_context_size}")

        # Run MUSt3R video inference with temporal consistency
        scene = must3r_inference_video(
            model=(self.encoder, self.decoder),
            device=self.device,
            image_size=512,
            amp=True,  # Automatic mixed precision
            filelist=frame_paths,
            max_bs=max_bs,
            init_num_images=init_num_images,
            batch_num_views=batch_num_views,
            num_refinements_iterations=10,
            local_context_size=local_context_size,
            verbose=True
        )

        print(f"[MUSt3R] ✓ Video inference complete")

        # Extract camera poses (c2w format)
        cameras = torch.stack(scene.cams2world).cpu().numpy()  # (N, 4, 4)

        # Extract intrinsics (focal lengths)
        focals = np.array(scene.focals)  # (N,)

        # Create intrinsics matrix for each frame
        intrinsics = []
        for i, focal in enumerate(focals):
            # Assume principal point at image center
            H, W = scene.true_shape[i].cpu().numpy()
            cx, cy = W / 2, H / 2

            K = np.array([
                [focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            intrinsics.append(K)

        intrinsics = np.array(intrinsics)

        # Extract 3D points from all frames
        all_pts3d = []
        for i, x_out_frame in enumerate(scene.x_out):
            if 'pts3d' in x_out_frame:
                pts = x_out_frame['pts3d'].cpu().numpy()  # (H, W, 3)
                # Flatten and filter valid points
                pts_flat = pts.reshape(-1, 3)
                valid_mask = ~np.isnan(pts_flat).any(axis=1)
                all_pts3d.append(pts_flat[valid_mask])

        # Combine all points
        if all_pts3d:
            pts3d = np.vstack(all_pts3d)
        else:
            pts3d = None

        return {
            'cameras': cameras,
            'intrinsics': intrinsics,
            'pts3d': pts3d,
            'frame_paths': frame_paths,
            'scene': scene  # Keep full scene for potential future use
        }

    def process_video(self, video_path: str, output_dir: str,
                     frame_interval: int = 1, max_frames: int = None) -> Dict:
        """
        Complete pipeline: extract frames → run MUSt3R → convert coordinates.

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

        # Run MUSt3R with temporal consistency
        result = self.run_must3r_inference(frame_paths)

        # Convert camera poses to Blender coordinate system
        cameras_opencv = result['cameras']
        intrinsics = result['intrinsics']

        cameras_blender = []
        for cam_matrix in cameras_opencv:
            # MUSt3R returns 4x4 c2w matrices
            blender_cam = self.opencv_to_blender_matrix(cam_matrix)
            cameras_blender.append(blender_cam)

        cameras_blender = np.array(cameras_blender)

        # Save point cloud (if available)
        pts3d_path = output_path / "point_cloud.npy"
        if result['pts3d'] is not None:
            point_cloud = result['pts3d']
            np.save(str(pts3d_path), point_cloud)
            print(f"[Save] Point cloud: {pts3d_path} ({len(point_cloud)} points)")

            # Detect ground plane
            ground_plane = self.detect_ground_plane(point_cloud)
        else:
            ground_plane = None
            print("[Warning] No point cloud available")

        # Save camera data
        poses_path = output_path / "camera_poses.npy"
        np.save(str(poses_path), cameras_blender)
        print(f"[Save] Camera poses: {poses_path}")

        intrinsics_path = output_path / "camera_intrinsics.npy"
        np.save(str(intrinsics_path), intrinsics)
        print(f"[Save] Camera intrinsics: {intrinsics_path}")

        # Save metadata
        metadata = {
            'num_frames': len(frame_paths),
            'frame_paths': [str(p) for p in frame_paths],
            'ground_plane': ground_plane,
            'temporal_consistency': True,  # MUSt3R provides this
            'method': 'MUSt3R'
        }

        metadata_path = output_path / "camera_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[Save] Metadata: {metadata_path}")

        return {
            'camera_poses': cameras_blender,
            'camera_intrinsics': intrinsics,
            'ground_plane': ground_plane,
            'frame_paths': frame_paths,
            'output_dir': str(output_path)
        }


if __name__ == "__main__":
    # Test the extractor
    print("Testing MUSt3R Camera Extractor...")

    try:
        extractor = MUSt3RCameraExtractor(device='cuda')
        print("[SUCCESS] Extractor initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
