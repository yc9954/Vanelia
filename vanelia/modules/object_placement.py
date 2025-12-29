"""
Object Placement Analyzer
Analyzes Dust3R point cloud to find optimal object placement and scale.
Uses RANSAC plane detection for intelligent surface alignment.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import json


class ObjectPlacementAnalyzer:
    """Analyze 3D point cloud to determine optimal object placement."""
    
    def __init__(self, point_cloud_path: Optional[Path] = None):
        """
        Initialize analyzer.
        
        Args:
            point_cloud_path: Path to saved point cloud (.npy file)
        """
        self.point_cloud_path = point_cloud_path
        self.points_3d = None
        
        if point_cloud_path and point_cloud_path.exists():
            self.load_point_cloud(point_cloud_path)
    
    def load_point_cloud(self, path: Path):
        """Load point cloud from file."""
        print(f"[Object Placement] Loading point cloud from: {path}")
        self.points_3d = np.load(path)
        
        # Handle different point cloud shapes
        if self.points_3d.ndim == 1:
            # 1D array - try to reshape to (N, 3)
            if len(self.points_3d) % 3 == 0:
                self.points_3d = self.points_3d.reshape(-1, 3)
            else:
                print(f"[Object Placement] WARNING: Point cloud shape {self.points_3d.shape} not recognized")
        elif self.points_3d.ndim > 2:
            # Flatten to (N, 3) if possible
            if self.points_3d.shape[-1] == 3:
                self.points_3d = self.points_3d.reshape(-1, 3)
        
        num_points = len(self.points_3d) if self.points_3d.ndim > 0 else 0
        print(f"[Object Placement] Loaded point cloud: shape {self.points_3d.shape}, ~{num_points} points")
    
    def detect_plane_ransac(self, 
                            points: np.ndarray,
                            max_iterations: int = 1000,
                            distance_threshold: float = 0.1,
                            min_inliers: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[int]]]:
        """
        Detect the largest plane in point cloud using RANSAC algorithm.
        
        Args:
            points: Point cloud array (N, 3)
            max_iterations: Maximum RANSAC iterations
            distance_threshold: Distance threshold for inliers (meters)
            min_inliers: Minimum number of inliers to consider a valid plane
        
        Returns:
            Tuple of (plane_center, plane_normal, inlier_indices)
            Returns (None, None, None) if no plane found
        """
        if len(points) < 3:
            return None, None, None
        
        print(f"[Plane Detection] Running RANSAC on {len(points)} points...")
        
        best_plane = None
        best_inliers = []
        best_inlier_count = 0
        
        # RANSAC iterations
        for iteration in range(max_iterations):
            # Randomly sample 3 points
            sample_indices = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[sample_indices]
            
            # Calculate plane normal using cross product
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            # Skip if points are collinear
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            
            normal = normal / norm  # Normalize
            
            # Calculate plane equation: ax + by + cz + d = 0
            # where (a, b, c) = normal, d = -normal · p1
            d = -np.dot(normal, p1)
            
            # Find inliers (points close to plane)
            distances = np.abs(np.dot(points, normal) + d)
            inliers = np.where(distances < distance_threshold)[0]
            
            if len(inliers) > best_inlier_count:
                best_inlier_count = len(inliers)
                best_inliers = inliers.tolist()
                # Recalculate plane from all inliers for better accuracy
                inlier_points = points[inliers]
                centroid = inlier_points.mean(axis=0)
                
                # Use SVD for more robust normal calculation
                centered = inlier_points - centroid
                if len(centered) >= 3:
                    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                    refined_normal = Vt[2]  # Last row is normal
                    if np.dot(refined_normal, normal) < 0:
                        refined_normal = -refined_normal  # Ensure consistent orientation
                    best_plane = (centroid, refined_normal)
        
        if best_inlier_count < min_inliers:
            print(f"[Plane Detection] WARNING: Only {best_inlier_count} inliers found (min: {min_inliers})")
            return None, None, None
        
        plane_center, plane_normal = best_plane
        print(f"[Plane Detection] ✓ Found plane with {best_inlier_count} inliers")
        print(f"[Plane Detection]   Center: {plane_center}")
        print(f"[Plane Detection]   Normal: {plane_normal}")
        
        return plane_center, plane_normal, best_inliers
    
    def raycast_to_point_cloud(self,
                               pixel_coords: Tuple[float, float],
                               camera_pose: np.ndarray,
                               camera_intrinsics: np.ndarray,
                               resolution: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Raycast from 2D pixel coordinates to 3D point cloud.
        
        Args:
            pixel_coords: (u, v) pixel coordinates (0-1 normalized or pixel values)
            camera_pose: 4x4 camera pose matrix (Blender coordinates)
            camera_intrinsics: Camera intrinsics matrix (3x3)
            resolution: Image resolution (width, height)
        
        Returns:
            3D point where ray intersects point cloud, or None
        """
        if self.points_3d is None or len(self.points_3d) == 0:
            return None
        
        # Normalize pixel coordinates if needed
        u, v = pixel_coords
        if u <= 1.0 and v <= 1.0:
            u = u * resolution[0]
            v = v * resolution[1]
        
        # Convert pixel to camera ray
        # Camera ray direction in camera space
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        # Normalized camera coordinates
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        z_cam = 1.0
        
        # Ray direction in camera space
        ray_dir_cam = np.array([x_cam, y_cam, z_cam])
        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)
        
        # Transform to world space
        cam_rotation = camera_pose[:3, :3]
        cam_position = camera_pose[:3, 3]
        ray_dir_world = cam_rotation @ ray_dir_cam
        
        # Find intersection with point cloud
        # Simple approach: find closest point along ray
        min_dist = float('inf')
        closest_point = None
        
        for point in self.points_3d:
            if len(point) < 3:
                continue
            point_3d = point[:3]
            
            # Vector from camera to point
            to_point = point_3d - cam_position
            
            # Project onto ray
            proj_length = np.dot(to_point, ray_dir_world)
            if proj_length < 0:
                continue  # Point is behind camera
            
            # Closest point on ray
            closest_on_ray = cam_position + proj_length * ray_dir_world
            
            # Distance from point to ray
            dist = np.linalg.norm(point_3d - closest_on_ray)
            
            if dist < min_dist:
                min_dist = dist
                closest_point = point_3d
        
        if closest_point is not None and min_dist < 0.5:  # 50cm threshold
            print(f"[Raycast] Hit point: {closest_point} (distance: {min_dist:.3f}m)")
            return closest_point
        
        return None
    
    def analyze_empty_space(self, 
                           camera_poses: np.ndarray,
                           search_radius: float = 2.0,
                           min_distance_from_camera: float = 0.5,
                           max_distance_from_camera: float = 5.0,
                           use_plane_detection: bool = True) -> Tuple[Tuple[float, float, float], float, Optional[np.ndarray]]:
        """
        Find optimal placement location by analyzing point cloud.
        
        Strategy:
        1. Use RANSAC to detect the largest plane (e.g., floor/table)
        2. Place object at plane center
        3. Align object Z-axis with plane normal
        4. Fallback to density-based approach if no plane found
        
        Args:
            camera_poses: Array of 4x4 camera pose matrices (Blender coordinates)
            search_radius: Radius to search for empty space (meters)
            min_distance_from_camera: Minimum distance from any camera (meters)
            max_distance_from_camera: Maximum distance from any camera (meters)
            use_plane_detection: If True, use RANSAC plane detection
        
        Returns:
            Tuple of (optimal_location, optimal_scale, plane_normal)
            plane_normal is None if no plane detected
        """
        if self.points_3d is None or len(self.points_3d) == 0:
            print("[Object Placement] WARNING: No point cloud available. Using default placement.")
            return (0.0, 0.0, -2.0), 1.0, None
        
        print(f"[Object Placement] Analyzing {len(self.points_3d)} points for optimal placement...")
        
        # Extract camera positions from poses (4th column, first 3 elements)
        camera_positions = []
        for pose in camera_poses:
            if pose.shape == (4, 4):
                cam_pos = pose[:3, 3]  # Translation component
                camera_positions.append(cam_pos)
        
        camera_positions = np.array(camera_positions)
        print(f"[Object Placement] Found {len(camera_positions)} camera positions")
        
        # Convert point cloud to Blender coordinates if needed
        # (Assuming points are already in Blender coordinates)
        points = self.points_3d
        
        # Try RANSAC plane detection first (preferred method)
        plane_normal = None
        if use_plane_detection:
            # Filter points within reasonable distance from cameras
            if len(camera_positions) > 0:
                valid_points = []
                for point in points:
                    if len(point.shape) == 0:
                        point = np.array([point])
                    if len(point) >= 3:
                        point_3d = point[:3]
                        distances = np.linalg.norm(camera_positions - point_3d, axis=1)
                        min_dist = np.min(distances)
                        if min_distance_from_camera <= min_dist <= max_distance_from_camera:
                            valid_points.append(point_3d)
                
                if len(valid_points) > 0:
                    valid_points = np.array(valid_points)
                    plane_center, plane_normal, inliers = self.detect_plane_ransac(valid_points)
                    
                    if plane_center is not None and plane_normal is not None:
                        # Use plane center as placement location
                        optimal_location = plane_center
                        
                        # Estimate scale based on camera distance
                        if len(camera_positions) > 0:
                            distances = np.linalg.norm(camera_positions - optimal_location, axis=1)
                            avg_distance = np.mean(distances)
                            optimal_scale = np.clip(avg_distance / 3.0, 0.3, 2.0)
                        else:
                            optimal_scale = 1.0
                        
                        print(f"[Object Placement] ✓ Using plane-based placement")
                        print(f"[Object Placement]   Location: {optimal_location}")
                        print(f"[Object Placement]   Normal: {plane_normal}")
                        print(f"[Object Placement]   Scale: {optimal_scale:.2f}")
                        
                        return tuple(optimal_location), optimal_scale, plane_normal
        
        # Fallback to density-based approach if plane detection failed
        print(f"[Object Placement] Falling back to density-based placement...")
        
        # Filter points: only consider points in reasonable range
        # Calculate distance from each point to nearest camera
        if len(camera_positions) > 0:
            # Find points within reasonable distance from cameras
            valid_points = []
            for point in points:
                if len(point.shape) == 0:  # Single point
                    point = np.array([point])
                if len(point) >= 3:
                    point_3d = point[:3]
                    # Distance to nearest camera
                    distances = np.linalg.norm(camera_positions - point_3d, axis=1)
                    min_dist = np.min(distances)
                    
                    if min_distance_from_camera <= min_dist <= max_distance_from_camera:
                        valid_points.append(point_3d)
            
            if len(valid_points) == 0:
                print("[Object Placement] WARNING: No valid points found. Using default placement.")
                return (0.0, 0.0, -2.0), 1.0, None
            
            valid_points = np.array(valid_points)
        else:
            valid_points = points[:, :3] if len(points.shape) > 1 else points
        
        # Find empty space: grid-based approach
        # Create a 3D grid and count points in each cell
        bounds_min = valid_points.min(axis=0) - search_radius
        bounds_max = valid_points.max(axis=0) + search_radius
        
        grid_resolution = 0.5  # 50cm grid cells
        grid_size = ((bounds_max - bounds_min) / grid_resolution).astype(int) + 1
        
        # Count points in each grid cell
        grid_indices = ((valid_points - bounds_min) / grid_resolution).astype(int)
        grid_indices = np.clip(grid_indices, 0, grid_size - 1)
        
        # Create density grid
        density_grid = np.zeros(tuple(grid_size))
        for idx in grid_indices:
            density_grid[tuple(idx)] += 1
        
        # Find cells with low density (empty space)
        # Prefer locations in front of cameras (negative Z in Blender)
        empty_candidates = []
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                for k in range(grid_size[2]):
                    if density_grid[i, j, k] < 5:  # Low density threshold
                        cell_center = bounds_min + np.array([i, j, k]) * grid_resolution
                        
                        # Check distance from cameras
                        if len(camera_positions) > 0:
                            distances = np.linalg.norm(camera_positions - cell_center, axis=1)
                            min_dist = np.min(distances)
                            
                            if min_distance_from_camera <= min_dist <= max_distance_from_camera:
                                # Prefer negative Z (in front of cameras in Blender)
                                score = -density_grid[i, j, k] - cell_center[2] * 0.1
                                empty_candidates.append((cell_center, score, min_dist))
        
        if len(empty_candidates) == 0:
            print("[Object Placement] WARNING: No empty space found. Using camera-relative placement.")
            # Fallback: place object in front of average camera position
            if len(camera_positions) > 0:
                avg_cam_pos = camera_positions.mean(axis=0)
                # Place 2 meters in front (negative Z direction)
                optimal_location = avg_cam_pos + np.array([0, 0, -2.0])
            else:
                optimal_location = np.array([0.0, 0.0, -2.0])
        else:
            # Sort by score (lowest density, most negative Z)
            empty_candidates.sort(key=lambda x: x[1])
            optimal_location = empty_candidates[0][0]
            print(f"[Object Placement] Found optimal location: {optimal_location}")
        
        # Estimate optimal scale based on point cloud extent and camera distances
        if len(camera_positions) > 0:
            # Average distance from cameras
            distances = np.linalg.norm(camera_positions - optimal_location, axis=1)
            avg_distance = np.mean(distances)
            
            # Scale based on distance: closer = smaller, farther = larger
            # Typical object size: 0.5-2.0 meters at 2-5 meter distance
            optimal_scale = np.clip(avg_distance / 3.0, 0.3, 2.0)
        else:
            optimal_scale = 1.0
        
        print(f"[Object Placement] Optimal location: {optimal_location}")
        print(f"[Object Placement] Optimal scale: {optimal_scale:.2f}")
        
        return tuple(optimal_location), optimal_scale, None  # No plane normal for density-based
    
    def estimate_scale_from_point_cloud(self, 
                                        object_bbox_size: Tuple[float, float, float],
                                        target_screen_size_ratio: float = 0.2) -> float:
        """
        Estimate object scale based on desired screen coverage.
        
        Args:
            object_bbox_size: Bounding box size of the object (width, height, depth)
            target_screen_size_ratio: Desired object size as ratio of screen (0.0-1.0)
        
        Returns:
            Estimated scale factor
        """
        if self.points_3d is None or len(self.points_3d) == 0:
            return 1.0
        
        # Estimate scene scale from point cloud
        points = self.points_3d
        if len(points.shape) > 1 and points.shape[1] >= 3:
            scene_extent = np.ptp(points[:, :3], axis=0)
            scene_scale = np.mean(scene_extent)
        else:
            scene_scale = 5.0  # Default
        
        # Calculate scale to achieve target screen coverage
        object_diagonal = np.linalg.norm(object_bbox_size)
        scale = (scene_scale * target_screen_size_ratio) / object_diagonal
        
        return np.clip(scale, 0.1, 5.0)


def analyze_placement_from_camera_data(camera_data_dir: Path,
                                       auto_placement: bool = True,
                                       manual_location: Optional[Tuple[float, float, float]] = None,
                                       manual_scale: Optional[float] = None,
                                       use_plane_detection: bool = True) -> Tuple[Tuple[float, float, float], float, Optional[np.ndarray]]:
    """
    Analyze camera data to determine object placement.
    
    Args:
        camera_data_dir: Directory containing camera_poses.npy and point_cloud.npy
        auto_placement: If True, automatically find optimal placement
        manual_location: Manual location override (x, y, z)
        manual_scale: Manual scale override
        use_plane_detection: If True, use RANSAC plane detection
    
    Returns:
        Tuple of (location, scale, plane_normal)
        plane_normal is None if no plane detected or manual placement
    """
    camera_data_dir = Path(camera_data_dir)
    
    # Load camera poses
    poses_path = camera_data_dir / 'camera_poses.npy'
    if not poses_path.exists():
        print(f"[Object Placement] WARNING: Camera poses not found: {poses_path}")
        return (0.0, 0.0, -2.0), 1.0
    
    camera_poses = np.load(poses_path)
    
    # Load point cloud if available
    point_cloud_path = camera_data_dir / 'point_cloud.npy'
    analyzer = None
    if point_cloud_path.exists():
        analyzer = ObjectPlacementAnalyzer(point_cloud_path)
    else:
        print(f"[Object Placement] Point cloud not found: {point_cloud_path}")
        print(f"[Object Placement] Using camera-based placement only")
    
    # Use manual overrides if provided
    plane_normal = None
    if manual_location is not None:
        location = manual_location
        print(f"[Object Placement] Using manual location: {location}")
    elif auto_placement and analyzer:
        location, scale, plane_normal = analyzer.analyze_empty_space(
            camera_poses, 
            use_plane_detection=use_plane_detection
        )
    else:
        # Fallback: place in front of average camera
        if len(camera_poses) > 0:
            camera_positions = np.array([pose[:3, 3] for pose in camera_poses])
            avg_cam_pos = camera_positions.mean(axis=0)
            location = tuple(avg_cam_pos + np.array([0, 0, -2.0]))
        else:
            location = (0.0, 0.0, -2.0)
        print(f"[Object Placement] Using fallback location: {location}")
    
    # Determine scale
    if manual_scale is not None:
        scale = manual_scale
        print(f"[Object Placement] Using manual scale: {scale}")
    elif auto_placement and analyzer and manual_location is None:
        # Scale already determined by analyze_empty_space
        pass
    else:
        scale = 1.0
        print(f"[Object Placement] Using default scale: {scale}")
    
    return location, scale, plane_normal

