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

        # GPU Optimization
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'  # Try CUDA first

        # Enable all available GPUs
        for device in prefs.devices:
            if device.type in {'CUDA', 'OPTIX', 'HIP'}:
                device.use = True

        scene.cycles.device = 'GPU'
        scene.cycles.samples = 64  # Increased samples for better quality
        scene.render.resolution_x = resolution_x
        scene.render.resolution_y = resolution_y
        scene.render.resolution_percentage = 100

        # CRITICAL: Transparent background (RGBA)
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'

        # Enable denoising for cleaner output (compensates for lower samples)
        scene.cycles.use_denoising = True
        scene.view_layers[0].cycles.use_denoising = True
        
        # Increase light bounces for better lighting
        scene.cycles.max_bounces = 8
        scene.cycles.diffuse_bounces = 4
        scene.cycles.glossy_bounces = 4
        scene.cycles.transparent_max_bounces = 8

        # Tile size for GPU rendering (deprecated in Blender 4.0+)
        try:
            if hasattr(scene.render, 'tile_x'):
                scene.render.tile_x = 256
                scene.render.tile_y = 256
        except AttributeError:
            # Blender 4.0+ uses automatic tiling
            pass

        print(f"[Blender] ✓ Transparent RGBA rendering enabled ({resolution_x}x{resolution_y})")

    def load_glb_model(self, glb_path: str, scale: float = 1.0,
                      location: tuple = (0, 0, 0),
                      plane_normal: np.ndarray = None) -> bpy.types.Object:
        """
        Load GLB model into scene.

        Args:
            glb_path: Path to .glb file
            scale: Uniform scale factor
            location: (x, y, z) position
            plane_normal: Normal vector of plane to align with (aligns object Z-axis)

        Returns:
            Loaded object (parent of hierarchy if multiple)
        """
        print(f"[Blender] Loading GLB: {glb_path}")

        # Import GLB with PBR material preservation
        bpy.ops.import_scene.gltf(
            filepath=glb_path,
            import_shading='NORMALS',  # ✅ Preserve PBR nodes
            merge_vertices=False,
            bone_heuristic='TEMPERANCE'
        )

        # Get imported objects
        imported_objects = bpy.context.selected_objects
        if not imported_objects:
            raise RuntimeError(f"Failed to import {glb_path}: No objects were imported")

        # Get imported object (latest object)
        imported_obj = imported_objects[0]
        imported_obj.scale = (scale, scale, scale)
        imported_obj.location = location
        
        # Print object placement info for debugging
        print(f"\n[Blender] Object Placement:")
        print(f"  - Location: ({location[0]:.6f}, {location[1]:.6f}, {location[2]:.6f})")
        print(f"  - Scale: {scale:.3f}")
        bbox = imported_obj.bound_box
        bbox_size = (
            max(v[0] for v in bbox) - min(v[0] for v in bbox),
            max(v[1] for v in bbox) - min(v[1] for v in bbox),
            max(v[2] for v in bbox) - min(v[2] for v in bbox)
        )
        print(f"  - Bounding Box Size: ({bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f})")
        print(f"  - Scaled Size: ({bbox_size[0]*scale:.3f}, {bbox_size[1]*scale:.3f}, {bbox_size[2]*scale:.3f})")

        # Align object Z-axis with plane normal if provided
        if plane_normal is not None:
            # Convert numpy array to Blender Vector
            target_normal = Vector(plane_normal.tolist() if hasattr(plane_normal, 'tolist') else plane_normal)
            target_normal.normalize()
            
            # Object's default up vector is Z-axis (0, 0, 1) in Blender
            default_up = Vector((0, 0, 1))
            
            # Calculate rotation to align default_up with target_normal
            # Use rotation_difference if available, otherwise use manual calculation
            try:
                # Calculate rotation quaternion
                rotation_quat = default_up.rotation_difference(target_normal)
                imported_obj.rotation_mode = 'QUATERNION'
                imported_obj.rotation_quaternion = rotation_quat
            except:
                # Fallback: manual calculation using cross product
                # Find rotation axis and angle
                axis = default_up.cross(target_normal)
                if axis.length > 1e-6:
                    axis.normalize()
                    angle = default_up.angle(target_normal)
                    imported_obj.rotation_mode = 'AXIS_ANGLE'
                    imported_obj.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
                else:
                    # Vectors are parallel, no rotation needed
                    pass
            
            print(f"[Blender] ✓ Aligned object Z-axis with plane normal: {target_normal}")

        # Validate and fix materials
        mat_fixed_count = 0
        for obj in imported_objects:
            if obj.type == 'MESH' and obj.data.materials:
                for slot in obj.material_slots:
                    mat = slot.material
                    if mat and mat.use_nodes:
                        nodes = mat.node_tree.nodes
                        bsdf = nodes.get('Principled BSDF')

                        if bsdf:
                            # Fix black materials (Base Color too dark)
                            base_color = bsdf.inputs['Base Color'].default_value
                            if base_color[0] < 0.05 and base_color[1] < 0.05 and base_color[2] < 0.05:
                                bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
                                mat_fixed_count += 1

                        # Ensure proper rendering
                        mat.blend_method = 'OPAQUE'
                        mat.shadow_method = 'OPAQUE'

        if mat_fixed_count > 0:
            print(f"[Blender] ⚠ Fixed {mat_fixed_count} black materials")

        print(f"[Blender] ✓ Loaded: {imported_obj.name} ({len(imported_objects)} objects)")
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
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Calculate focal length in mm (assuming sensor width of 36mm)
        sensor_width = 36.0
        focal_length = (fx / width) * sensor_width

        camera_obj.data.sensor_width = sensor_width
        camera_obj.data.lens = focal_length

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

        # Print first camera pose for debugging
        if len(poses) > 0:
            first_pose = poses[0]
            cam_pos = first_pose[:3, 3]
            print(f"[Blender] First camera position: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})")

        for idx, pose_matrix in enumerate(poses):
            frame_num = start_frame + idx

            # Convert numpy array to Blender Matrix
            # NOTE: Matrix is already world-to-camera (inverted in dust3r_camera_extraction.py)
            cam_matrix = Matrix(pose_matrix.tolist())

            # Blender camera looks down -Z axis, apply correction if needed
            # The pose from Dust3R should already be in correct Blender space
            # after opencv_to_blender_matrix conversion
            camera_obj.matrix_world = cam_matrix
            
            # Debug: Print camera info for first frame
            if idx == 0:
                cam_pos = cam_matrix.translation
                cam_rot = cam_matrix.to_euler()
                print(f"[Blender] Frame {frame_num} camera:")
                print(f"  Position: ({cam_pos.x:.3f}, {cam_pos.y:.3f}, {cam_pos.z:.3f})")
                print(f"  Rotation: ({cam_rot.x:.3f}, {cam_rot.y:.3f}, {cam_rot.z:.3f})")

            # Insert keyframes
            camera_obj.keyframe_insert(data_path="location", frame=frame_num)
            camera_obj.keyframe_insert(data_path="rotation_euler", frame=frame_num)

        print(f"[Blender] ✓ Animation: frames {start_frame}-{scene.frame_end}")

    def setup_lighting(self, energy: float = 8.0):
        """
        Create neutral lighting setup for IC-Light relighting.
        
        IC-Light (ControlNet Inpaint) will relight the object to match the background,
        so we use moderate, neutral lighting that provides enough detail for relighting.
        
        Args:
            energy: Base light intensity (moderate for relighting compatibility)
        """
        print("[Blender] Setting up neutral lighting (IC-Light will relight to match background)...")

        # Main key light (front-right, above) - moderate intensity
        bpy.ops.object.light_add(type='AREA', location=(3, -3, 4))
        key_light = bpy.context.active_object
        key_light.data.energy = energy
        key_light.data.size = 8
        key_light.name = "KeyLight"

        # Fill light (front-left, above) - softer fill
        bpy.ops.object.light_add(type='AREA', location=(-3, -3, 4))
        fill_light = bpy.context.active_object
        fill_light.data.energy = energy * 0.6
        fill_light.data.size = 8
        fill_light.name = "FillLight"

        # Top light (directly above) - ambient fill for visibility
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 6))
        top_light = bpy.context.active_object
        top_light.data.energy = energy * 0.3
        top_light.data.size = 10
        top_light.name = "TopLight"

        print(f"[Blender] ✓ Neutral lighting setup complete (3 lights, base energy: {energy})")
        print(f"[Blender]   → IC-Light will relight object to match background video")

    def check_object_visibility(self, camera_obj: bpy.types.Object, obj: bpy.types.Object, frame: int = 1):
        """
        Check if object is visible from camera at given frame.
        
        Args:
            camera_obj: Camera object
            obj: Object to check
            frame: Frame number
        """
        scene = bpy.context.scene
        scene.frame_set(frame)
        
        # Get camera and object positions
        cam_pos = Vector(camera_obj.matrix_world.translation)
        obj_pos = Vector(obj.location)
        
        # Get camera direction (looks down -Z in Blender)
        cam_matrix = camera_obj.matrix_world
        cam_forward = Vector((0, 0, -1))
        cam_forward.rotate(cam_matrix.to_3x3())
        
        # Vector from camera to object
        to_obj = obj_pos - cam_pos
        distance = to_obj.length
        
        # Check if object is in front of camera
        dot_product = to_obj.normalized().dot(cam_forward.normalized())
        in_front = dot_product > 0
        
        # Get object bounding box
        bbox_corners = [Vector(corner) for corner in obj.bound_box]
        bbox_center = sum(bbox_corners, Vector()) / len(bbox_corners)
        bbox_size = max((max(c[i] for c in bbox_corners) - min(c[i] for c in bbox_corners)) for i in range(3))
        
        print(f"\n[Blender] Visibility Check (Frame {frame}):")
        print(f"  - Camera position: ({cam_pos.x:.3f}, {cam_pos.y:.3f}, {cam_pos.z:.3f})")
        print(f"  - Object position: ({obj_pos.x:.3f}, {obj_pos.y:.3f}, {obj_pos.z:.3f})")
        print(f"  - Distance: {distance:.3f}")
        print(f"  - In front of camera: {in_front}")
        print(f"  - Object bbox center: ({bbox_center.x:.3f}, {bbox_center.y:.3f}, {bbox_center.z:.3f})")
        print(f"  - Object bbox size: {bbox_size:.3f}")
        
        if not in_front:
            print(f"  ⚠ WARNING: Object is behind camera!")
        if distance > 20:
            print(f"  ⚠ WARNING: Object is very far from camera ({distance:.1f} units)")
        if bbox_size * obj.scale[0] < 0.01:
            print(f"  ⚠ WARNING: Object is very small (scaled size: {bbox_size * obj.scale[0]:.3f})")

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
            frame_filename = f"{file_prefix}{frame:06d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            scene.render.filepath = frame_path

            # Render
            bpy.ops.render.render(write_still=True)

            if (frame - scene.frame_start) % 10 == 0:
                print(f"[Blender] Rendered frame {frame}/{scene.frame_end}")

        print(f"\n[Blender] ✓ Render complete: {num_frames} frames")


def parse_args():
    """Parse command-line arguments."""
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    import argparse
    parser = argparse.ArgumentParser(description='Blender headless renderer')
    parser.add_argument('--glb', type=str, required=True, help='GLB model path')
    parser.add_argument('--poses', type=str, required=True, help='Camera poses .npy file')
    parser.add_argument('--intrinsics', type=str, required=True, help='Camera intrinsics .npy file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--scale', type=float, default=1.0, help='Model scale')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080], help='Resolution (width height)')
    parser.add_argument('--location', type=float, nargs=3, default=None, help='Object location (x y z)')
    parser.add_argument('--position', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Object position (x y z)')
    parser.add_argument('--rotation', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Object rotation in degrees (x y z)')
    parser.add_argument('--auto-ground', action='store_true', help='Auto-place object on ground plane')
    parser.add_argument('--metadata', type=str, default=None, help='Path to camera metadata JSON')
    parser.add_argument('--plane-normal', type=float, nargs=3, default=None, help='Plane normal for alignment (x y z)')

    return parser.parse_args(argv)


def main():
    """Main rendering function."""
    args = parse_args()

    # Handle location vs position for backward compatibility
    if args.location is not None:
        obj_position = tuple(args.location)
    else:
        obj_position = tuple(args.position)

    plane_normal = np.array(args.plane_normal) if args.plane_normal else None

    # Load metadata for ground plane info and background frames
    ground_plane = None
    bg_image = None
    if args.metadata:
        import json
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
            ground_plane = metadata.get('ground_plane')
            if ground_plane:
                print(f"[Load] Ground plane: {ground_plane['A']:.3f}x + {ground_plane['B']:.3f}y + "
                      f"{ground_plane['C']:.3f}z + {ground_plane['D']:.3f} = 0")

            # Get first background frame for lighting estimation
            frame_paths = metadata.get('frame_paths', [])
            if frame_paths:
                bg_image = frame_paths[0]

    # Apply rotation (convert degrees to radians)
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
    renderer.setup_render_settings(resolution_x=args.resolution[0], resolution_y=args.resolution[1])

    # Load model
    obj = renderer.load_glb_model(
        glb_path=args.glb,
        scale=args.scale,
        location=obj_position,
        plane_normal=plane_normal
    )

    # Create shadow catcher
    renderer.create_shadow_catcher(size=20.0)

    # Setup camera
    camera = renderer.setup_camera()

    # Load camera data
    poses = np.load(args.poses)
    intrinsics = np.load(args.intrinsics)

    # Apply intrinsics (use first frame's intrinsics)
    if len(intrinsics.shape) == 3:
        intrinsics = intrinsics[0]
    renderer.set_camera_intrinsics(camera, intrinsics, tuple(args.resolution))

    # Apply poses
    renderer.apply_camera_poses(camera, poses)

    # Setup lighting
    renderer.setup_lighting(energy=8.0)

    # Check visibility before rendering
    renderer.check_object_visibility(camera, obj, frame=1)

    # Render
    renderer.render_animation(args.output)

    print("\n[Blender] ✓ Rendering complete!")


if __name__ == '__main__':
    main()
