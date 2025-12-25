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
                      location: tuple = (0, 0, 0),
                      rotation: tuple = (0, 0, 0)) -> bpy.types.Object:
        """
        Load .glb model into scene with proper material handling.

        Args:
            glb_path: Path to .glb file
            scale: Uniform scale factor
            location: (x, y, z) position
            rotation: (x, y, z) rotation in radians

        Returns:
            Loaded object (parent of hierarchy if multiple)
        """
        print(f"[Blender] Loading GLB: {glb_path}")

        # Import GLB with all options enabled
        bpy.ops.import_scene.gltf(
            filepath=glb_path,
            import_shading='FLAT',  # Import materials as-is
            merge_vertices=False
        )

        # Get all imported objects
        imported_objects = bpy.context.selected_objects

        if not imported_objects:
            raise RuntimeError(f"Failed to import {glb_path}")

        # Find the root object (parent or first mesh)
        root_obj = None
        for obj in imported_objects:
            if obj.parent is None:
                root_obj = obj
                break

        if root_obj is None:
            root_obj = imported_objects[0]

        # Apply transformations
        root_obj.scale = (scale, scale, scale)
        root_obj.location = location
        root_obj.rotation_euler = rotation

        # Ensure materials use Cycles properly
        for obj in imported_objects:
            if obj.type == 'MESH' and obj.data.materials:
                for mat in obj.data.materials:
                    if mat:
                        mat.use_nodes = True
                        # Ensure proper viewport and render visibility
                        mat.blend_method = 'OPAQUE'

        print(f"[Blender] ✓ Loaded: {root_obj.name} ({len(imported_objects)} objects)")
        return root_obj

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
            # NOTE: Matrix is already world-to-camera (inverted in dust3r_camera_extraction.py)
            cam_matrix = Matrix(pose_matrix.tolist())

            # Blender camera looks down -Z axis, apply correction if needed
            # The pose from Dust3R should already be in correct Blender space
            # after opencv_to_blender_matrix conversion
            camera_obj.matrix_world = cam_matrix

            # Insert keyframes
            camera_obj.keyframe_insert(data_path="location", frame=frame_num)
            camera_obj.keyframe_insert(data_path="rotation_euler", frame=frame_num)

        print(f"[Blender] ✓ Animation: frames {start_frame}-{scene.frame_end}")

    def setup_lighting(self, use_world_hdri: bool = True,
                      hdri_strength: float = 1.0,
                      sun_energy: float = 5.0):
        """
        Create realistic lighting setup with environment lighting.

        Args:
            use_world_hdri: Use procedural sky/environment lighting
            hdri_strength: Environment lighting strength
            sun_energy: Sun light intensity
        """
        print("[Blender] Setting up lighting...")

        # Setup world environment lighting (replaces HDRI for simplicity)
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # Clear existing nodes
        nodes.clear()

        # Background shader with sky texture
        node_background = nodes.new(type='ShaderNodeBackground')
        node_output = nodes.new(type='ShaderNodeOutputWorld')

        if use_world_hdri:
            # Add Sky Texture for outdoor/realistic lighting
            node_sky = nodes.new(type='ShaderNodeTexSky')
            node_sky.sky_type = 'NISHITA'  # Physically-based sky
            node_sky.sun_elevation = np.radians(45)  # 45-degree sun
            node_sky.sun_rotation = 0
            node_sky.ground_albedo = 0.3

            links.new(node_sky.outputs['Color'], node_background.inputs['Color'])

        node_background.inputs['Strength'].default_value = hdri_strength

        links.new(node_background.outputs['Background'], node_output.inputs['Surface'])

        # Add sun light for shadows
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        sun = bpy.context.active_object
        sun.data.energy = sun_energy
        sun.rotation_euler = (np.radians(45), 0, np.radians(45))

        print(f"[Blender] ✓ Lighting: World environment + Sun (strength={hdri_strength})")


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
    parser.add_argument("--metadata", type=str, default=None, help="Path to camera_metadata.json")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--scale", type=float, default=1.0, help="Model scale")
    parser.add_argument("--position", type=float, nargs=3, default=[0, 0, 0],
                       help="Object position (x y z)")
    parser.add_argument("--rotation", type=float, nargs=3, default=[0, 0, 0],
                       help="Object rotation in degrees (x y z)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080],
                       help="Resolution (width height)")
    parser.add_argument("--auto-ground", action='store_true',
                       help="Automatically place object on detected ground plane")

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

    # Load metadata for ground plane info
    ground_plane = None
    if args.metadata:
        import json
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
            ground_plane = metadata.get('ground_plane')
            if ground_plane:
                print(f"[Load] Ground plane: {ground_plane['A']:.3f}x + {ground_plane['B']:.3f}y + "
                      f"{ground_plane['C']:.3f}z + {ground_plane['D']:.3f} = 0")

    # Determine object placement
    obj_position = tuple(args.position)
    obj_rotation = tuple(np.radians(args.rotation))  # Convert degrees to radians

    if args.auto_ground and ground_plane:
        # Place object on ground plane at (x, y) = (0, 0)
        # Solve for z: z = -(A*x + B*y + D) / C
        A, B, C, D = ground_plane['A'], ground_plane['B'], ground_plane['C'], ground_plane['D']
        x, y = obj_position[0], obj_position[1]
        if abs(C) > 1e-6:
            z_ground = -(A * x + B * y + D) / C
            obj_position = (x, y, z_ground)
            print(f"[Auto-Ground] Placing object at ({x:.2f}, {y:.2f}, {z_ground:.2f})")
        else:
            print("[Auto-Ground] WARNING: Ground plane is vertical, using manual position")

    # Initialize renderer
    renderer = BlenderRenderer()
    renderer.setup_render_settings(args.resolution[0], args.resolution[1])

    # Load 3D model with position and rotation
    model_obj = renderer.load_glb_model(
        args.glb,
        scale=args.scale,
        location=obj_position,
        rotation=obj_rotation
    )

    # Setup shadow catcher (aligned with ground plane if available)
    if ground_plane:
        # TODO: Rotate shadow catcher to match ground plane normal
        renderer.create_shadow_catcher(size=20.0)
    else:
        renderer.create_shadow_catcher(size=20.0)

    # Setup realistic lighting
    renderer.setup_lighting(use_world_hdri=True, hdri_strength=1.0, sun_energy=5.0)

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
