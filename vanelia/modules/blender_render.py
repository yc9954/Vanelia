"""
Module B: Blender Headless Rendering Script
Renders 3D models with camera animation and shadow catcher.
Outputs RGBA images with transparent background.

Usage:
    blender --background --python blender_render.py -- \
        --glb brand_asset.glb \
        --poses camera_poses.npy \
        --intrinsics camera_intrinsics.npy \
        --output render_frames/
"""

import bpy
import numpy as np
import sys
import os
from pathlib import Path
from mathutils import Matrix, Vector
import json


class BlenderRenderer:
    """Headless Blender renderer for video object insertion."""

    def __init__(self):
        """Initialize Blender scene."""
        self.clear_scene()
        self.setup_render_settings()

    def clear_scene(self):
        """Remove all default objects from scene."""
        print("[Blender] Clearing default scene...")

        # Delete all objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Delete all materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)

        # Delete all lights
        for light in bpy.data.lights:
            bpy.data.lights.remove(light)

    def setup_render_settings(self, resolution_x: int = 1920, resolution_y: int = 1080):
        """
        Configure render settings for transparent RGBA output.

        Args:
            resolution_x: Output width
            resolution_y: Output height
        """
        print(f"[Blender] Setting up render: {resolution_x}x{resolution_y}")

        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 128  # High quality
        scene.render.resolution_x = resolution_x
        scene.render.resolution_y = resolution_y
        scene.render.resolution_percentage = 100

        # CRITICAL: Transparent background (RGBA)
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'

        # Enable denoising for cleaner output
        scene.cycles.use_denoising = True
        scene.view_layers[0].cycles.use_denoising = True

        print("[Blender] ✓ Transparent RGBA rendering enabled")

    def load_glb_model(self, glb_path: str, scale: float = 1.0,
                      location: tuple = (0, 0, 0)) -> bpy.types.Object:
        """
        Load .glb model into scene.

        Args:
            glb_path: Path to .glb file
            scale: Uniform scale factor
            location: (x, y, z) position

        Returns:
            Loaded object
        """
        print(f"[Blender] Loading GLB: {glb_path}")

        # Import GLB
        bpy.ops.import_scene.gltf(filepath=glb_path)

        # Get imported object (latest object)
        imported_obj = bpy.context.selected_objects[0]
        imported_obj.scale = (scale, scale, scale)
        imported_obj.location = location

        print(f"[Blender] ✓ Loaded: {imported_obj.name}")
        return imported_obj

    def create_shadow_catcher(self, size: float = 10.0) -> bpy.types.Object:
        """
        Create invisible shadow catcher plane.

        Args:
            size: Plane size

        Returns:
            Shadow catcher object
        """
        print("[Blender] Creating shadow catcher...")

        # Create plane
        bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
        plane = bpy.context.active_object
        plane.name = "ShadowCatcher"

        # Create shadow catcher material
        mat = bpy.data.materials.new(name="ShadowCatcherMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        # Setup shader nodes
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_shader = nodes.new(type='ShaderNodeBsdfDiffuse')

        # Link nodes
        links = mat.node_tree.links
        links.new(node_shader.outputs['BSDF'], node_output.inputs['Surface'])

        # Assign material
        plane.data.materials.append(mat)

        # Enable shadow catcher in Cycles
        plane.is_shadow_catcher = True

        print("[Blender] ✓ Shadow catcher created")
        return plane

    def setup_camera(self, name: str = "Camera") -> bpy.types.Object:
        """
        Create camera object.

        Args:
            name: Camera name

        Returns:
            Camera object
        """
        print("[Blender] Creating camera...")

        camera_data = bpy.data.cameras.new(name=name)
        camera_obj = bpy.data.objects.new(name, camera_data)
        bpy.context.scene.collection.objects.link(camera_obj)
        bpy.context.scene.camera = camera_obj

        return camera_obj

    def set_camera_intrinsics(self, camera_obj: bpy.types.Object,
                             intrinsics: np.ndarray, resolution: tuple):
        """
        Apply camera intrinsic matrix.

        Args:
            camera_obj: Camera object
            intrinsics: 3x3 intrinsic matrix [fx, fy, cx, cy]
            resolution: (width, height)
        """
        width, height = resolution

        if intrinsics.shape == (3, 3):
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
        else:
            # Assume [fx, fy, cx, cy] format
            fx, fy, cx, cy = intrinsics[:4]

        # Calculate sensor width (assuming 35mm equivalent)
        sensor_width = 36.0  # mm
        focal_length = (fx * sensor_width) / width

        camera_obj.data.lens = focal_length
        camera_obj.data.sensor_width = sensor_width

        # Shift for principal point offset
        camera_obj.data.shift_x = (cx - width / 2) / width
        camera_obj.data.shift_y = (cy - height / 2) / height

        print(f"[Blender] ✓ Camera focal length: {focal_length:.2f}mm")

    def apply_camera_poses(self, camera_obj: bpy.types.Object,
                          poses: np.ndarray, start_frame: int = 1):
        """
        Apply camera pose animation using keyframes.

        Args:
            camera_obj: Camera object
            poses: Array of 4x4 camera matrices (Blender coordinate system)
            start_frame: Starting frame number
        """
        print(f"[Blender] Applying {len(poses)} camera poses...")

        scene = bpy.context.scene
        scene.frame_start = start_frame
        scene.frame_end = start_frame + len(poses) - 1

        for idx, pose_matrix in enumerate(poses):
            frame_num = start_frame + idx

            # Convert numpy array to Blender Matrix
            # Dust3R outputs camera-to-world, Blender uses world-to-camera
            # So we need to invert the matrix
            cam_matrix = Matrix(pose_matrix.tolist())

            # Blender camera looks down -Z axis, apply correction if needed
            # The pose from Dust3R should already be in correct Blender space
            # after opencv_to_blender_matrix conversion
            camera_obj.matrix_world = cam_matrix

            # Insert keyframes
            camera_obj.keyframe_insert(data_path="location", frame=frame_num)
            camera_obj.keyframe_insert(data_path="rotation_euler", frame=frame_num)

        print(f"[Blender] ✓ Animation: frames {start_frame}-{scene.frame_end}")

    def setup_lighting(self, energy: float = 5.0):
        """
        Create basic lighting setup.

        Args:
            energy: Light intensity
        """
        print("[Blender] Setting up lighting...")

        # Key light
        bpy.ops.object.light_add(type='AREA', location=(5, -5, 5))
        key_light = bpy.context.active_object
        key_light.data.energy = energy
        key_light.data.size = 5

        # Fill light
        bpy.ops.object.light_add(type='AREA', location=(-5, -5, 3))
        fill_light = bpy.context.active_object
        fill_light.data.energy = energy * 0.5
        fill_light.data.size = 5

        print("[Blender] ✓ Lighting setup complete")

    def render_animation(self, output_dir: str, file_prefix: str = "frame_"):
        """
        Render all frames.

        Args:
            output_dir: Output directory path
            file_prefix: Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)

        scene = bpy.context.scene
        num_frames = scene.frame_end - scene.frame_start + 1

        print(f"\n[Blender] Starting render: {num_frames} frames...")
        print(f"[Blender] Output: {output_dir}")

        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)

            # Set output path
            output_path = os.path.join(output_dir, f"{file_prefix}{frame:06d}.png")
            scene.render.filepath = output_path

            # Render
            bpy.ops.render.render(write_still=True)

            progress = ((frame - scene.frame_start + 1) / num_frames) * 100
            print(f"  [{progress:5.1f}%] Frame {frame}/{scene.frame_end} → {output_path}")

        print(f"\n[Blender] ✓ Render complete: {num_frames} frames")


def parse_args():
    """Parse command-line arguments (after --)."""
    try:
        argv = sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        argv = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=str, required=True, help="Path to .glb model")
    parser.add_argument("--poses", type=str, required=True, help="Path to camera_poses.npy")
    parser.add_argument("--intrinsics", type=str, required=True, help="Path to camera_intrinsics.npy")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--scale", type=float, default=1.0, help="Model scale")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080],
                       help="Resolution (width height)")

    return parser.parse_args(argv)


def main():
    """Main rendering pipeline."""
    args = parse_args()

    print("\n" + "="*60)
    print("VANELIA - Blender Headless Renderer")
    print("="*60 + "\n")

    # Load camera data
    print(f"[Load] Camera poses: {args.poses}")
    poses = np.load(args.poses)
    print(f"[Load] Intrinsics: {args.intrinsics}")
    intrinsics = np.load(args.intrinsics)
    print(f"[Load] ✓ Loaded {len(poses)} camera poses\n")

    # Initialize renderer
    renderer = BlenderRenderer()
    renderer.setup_render_settings(args.resolution[0], args.resolution[1])

    # Load 3D model
    model_obj = renderer.load_glb_model(args.glb, scale=args.scale)

    # Setup shadow catcher
    renderer.create_shadow_catcher(size=20.0)

    # Setup lighting
    renderer.setup_lighting(energy=10.0)

    # Setup camera
    camera = renderer.setup_camera()
    renderer.set_camera_intrinsics(camera, intrinsics[0], tuple(args.resolution))
    renderer.apply_camera_poses(camera, poses)

    # Render
    renderer.render_animation(args.output)

    print("\n" + "="*60)
    print("RENDER COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
