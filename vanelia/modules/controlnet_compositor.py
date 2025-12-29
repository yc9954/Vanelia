"""
Module C: ControlNet Depth-based Compositor (IC-Light 대체)
Depth-guided compositing with ControlNet for better control and temporal consistency.

ControlNet은 허깅페이스에서 바로 사용 가능하며 더 안정적입니다.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import subprocess
import sys

# Diffusers imports
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image


class ControlNetCompositor:
    """
    ControlNet-based video compositor for temporal consistency.
    Uses depth/normal maps for better geometry preservation.
    """

    def __init__(self,
                 controlnet_type: str = "depth",
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda",
                 torch_dtype=torch.float16):
        """
        Initialize ControlNet pipeline.

        Args:
            controlnet_type: Type of ControlNet ('depth', 'normal', 'canny')
            model_id: Base Stable Diffusion model
            device: Device to run on
            torch_dtype: Torch dtype for inference
        """
        self.device = device
        self.dtype = torch_dtype
        self.controlnet_type = controlnet_type

        print(f"[ControlNet] Loading {controlnet_type} ControlNet...")

        # Load ControlNet model
        controlnet_models = {
            'depth': "lllyasviel/control_v11f1p_sd15_depth",
            'normal': "lllyasviel/control_v11p_sd15_normalbae",
            'canny': "lllyasviel/control_v11p_sd15_canny"
        }

        controlnet_path = controlnet_models.get(controlnet_type, controlnet_models['depth'])

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch_dtype
        ).to(device)

        # Load SD pipeline with ControlNet
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None
        ).to(device)

        # Use faster scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Enable optimizations
        self.pipe.enable_attention_slicing()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[ControlNet] ✓ xformers enabled")
        except Exception:
            print("[ControlNet] ⚠ xformers not available")

        print(f"[ControlNet] ✓ Pipeline loaded ({controlnet_type})\n")

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB image using MiDaS.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Depth map (H, W) normalized to 0-255
        """
        try:
            from transformers import pipeline as hf_pipeline

            # Lazy load depth estimator
            if not hasattr(self, 'depth_estimator'):
                print("[Depth] Loading MiDaS depth estimator...")
                self.depth_estimator = hf_pipeline(
                    "depth-estimation",
                    model="Intel/dpt-large",
                    device=0 if self.device == "cuda" else -1
                )
                print("[Depth] ✓ Depth estimator loaded")

            # Estimate depth
            depth_result = self.depth_estimator(Image.fromarray(image))
            depth_map = np.array(depth_result['depth'])

            # Normalize to 0-255
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            depth_map = (depth_map * 255).astype(np.uint8)

            return depth_map

        except Exception as e:
            print(f"[Depth] Failed to estimate depth: {e}")
            # Fallback: return dummy depth (all same depth)
            return np.ones_like(image[:, :, 0], dtype=np.uint8) * 128

    def alpha_blend(self, foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
        """
        Alpha blend foreground (RGBA) with background (RGB).

        Args:
            foreground: RGBA image (H, W, 4)
            background: RGB image (H, W, 3)

        Returns:
            Blended RGB image
        """
        # Resize if needed
        if foreground.shape[:2] != background.shape[:2]:
            foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

        # Extract alpha channel
        if foreground.shape[2] == 4:
            alpha = foreground[:, :, 3:4] / 255.0
            fg_rgb = foreground[:, :, :3]
        else:
            alpha = np.ones((foreground.shape[0], foreground.shape[1], 1))
            fg_rgb = foreground

        # Blend
        blended = (fg_rgb * alpha + background * (1 - alpha)).astype(np.uint8)
        return blended

    def process_frame(self,
                     foreground_path: str,
                     background_path: str,
                     strength: float = 0.4,
                     guidance_scale: float = 7.5,
                     seed: int = 12345,
                     prompt: str = "high quality photo, professional lighting, sharp focus, 8k",
                     negative_prompt: str = "blurry, low quality, distorted, artifacts",
                     num_inference_steps: int = 20,
                     controlnet_conditioning_scale: float = 0.8) -> Image.Image:
        """
        Process single frame with ControlNet.

        Args:
            foreground_path: Path to rendered RGBA frame
            background_path: Path to background frame
            strength: Denoising strength (0.3-0.5 for balance)
            guidance_scale: CFG scale
            seed: Random seed (FIXED for consistency)
            prompt: Positive prompt
            negative_prompt: Negative prompt
            num_inference_steps: Diffusion steps
            controlnet_conditioning_scale: ControlNet strength (0.5-1.0)

        Returns:
            Refined image
        """
        # Load images
        fg_img = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
        bg_img = cv2.imread(background_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

        # 1st pass: Alpha blend
        composite = self.alpha_blend(fg_img, bg_img)
        composite_pil = Image.fromarray(composite)

        # Generate control signal (depth map)
        if self.controlnet_type == 'depth':
            control_image = self.estimate_depth(composite)
            control_image_pil = Image.fromarray(control_image).convert("RGB")
        elif self.controlnet_type == 'canny':
            # Canny edge detection
            gray = cv2.cvtColor(composite, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            control_image_pil = Image.fromarray(edges).convert("RGB")
        else:
            # Fallback to composite itself
            control_image_pil = composite_pil

        # Setup generator with FIXED seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run ControlNet refinement
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=composite_pil,
                control_image=control_image_pil,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator
            )

            result_img = output.images[0]

        return result_img

    def process_video_sequence(self,
                              render_dir: str,
                              background_dir: str,
                              output_dir: str,
                              strength: float = 0.4,
                              seed: int = 12345,
                              prompt: str = "photorealistic, high quality, natural lighting, 4k",
                              negative_prompt: str = "blurry, low quality, cartoon") -> List[str]:
        """
        Process complete video sequence with temporal consistency.

        Args:
            render_dir: Directory with Blender renders (RGBA)
            background_dir: Directory with background frames
            output_dir: Output directory for refined frames
            strength: Denoising strength (0.3-0.5 recommended)
            seed: Fixed seed for all frames
            prompt: Positive prompt
            negative_prompt: Negative prompt

        Returns:
            List of output frame paths
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get frame lists
        render_frames = sorted(Path(render_dir).glob("*.png"))
        bg_frames = sorted(Path(background_dir).glob("*.jpg")) + \
                   sorted(Path(background_dir).glob("*.png"))

        assert len(render_frames) == len(bg_frames), \
            f"Frame count mismatch: {len(render_frames)} renders vs {len(bg_frames)} backgrounds"

        num_frames = len(render_frames)
        print(f"\n[ControlNet] Processing {num_frames} frames...")
        print(f"[ControlNet] Settings:")
        print(f"  - Type: {self.controlnet_type}")
        print(f"  - Strength: {strength}")
        print(f"  - Seed: {seed} (FIXED)")
        print()

        output_paths = []

        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            frame_iter = tqdm(enumerate(zip(render_frames, bg_frames)),
                            total=num_frames,
                            desc="ControlNet Processing",
                            unit="frame")
        except ImportError:
            frame_iter = enumerate(zip(render_frames, bg_frames))

        for idx, (render_path, bg_path) in frame_iter:
            # Process frame
            output_img = self.process_frame(
                foreground_path=str(render_path),
                background_path=str(bg_path),
                strength=strength,
                seed=seed,  # SAME seed for all frames
                prompt=prompt,
                negative_prompt=negative_prompt
            )

            # Save output
            output_path = os.path.join(output_dir, f"refined_{idx:06d}.png")
            output_img.save(output_path)
            output_paths.append(output_path)

            # Clear CUDA cache periodically
            if idx % 10 == 0:
                torch.cuda.empty_cache()

            # Progress (fallback for no tqdm)
            if 'tqdm' not in sys.modules:
                progress = ((idx + 1) / num_frames) * 100
                print(f"  [{progress:5.1f}%] Frame {idx+1}/{num_frames} → {output_path}")

        print(f"\n[ControlNet] ✓ Refined {num_frames} frames")
        return output_paths

    @staticmethod
    def encode_video_ffmpeg(frame_dir: str,
                          output_path: str,
                          fps: int = 30,
                          crf: int = 18,
                          pattern: str = "refined_%06d.png") -> str:
        """
        Encode frames to MP4 using ffmpeg.

        Args:
            frame_dir: Directory containing frames
            output_path: Output video path
            fps: Frames per second
            crf: Constant Rate Factor (18 = high quality)
            pattern: Frame filename pattern

        Returns:
            Output video path
        """
        print(f"\n[ffmpeg] Encoding video...")
        print(f"  - Input: {frame_dir}/{pattern}")
        print(f"  - Output: {output_path}")
        print(f"  - FPS: {fps}, CRF: {crf}")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frame_dir, pattern),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"[ffmpeg] ✓ Video encoded: {output_path}")
        else:
            print(f"[ffmpeg] ERROR: {result.stderr}")
            raise RuntimeError("ffmpeg encoding failed")

        return output_path


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="ControlNet Depth-based Compositor")
    parser.add_argument("--render-dir", type=str, required=True,
                       help="Directory with Blender renders (RGBA)")
    parser.add_argument("--bg-dir", type=str, required=True,
                       help="Directory with background frames")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for refined frames")
    parser.add_argument("--video-output", type=str, default="final_output.mp4",
                       help="Final video output path")
    parser.add_argument("--controlnet-type", type=str, default="depth",
                       choices=['depth', 'normal', 'canny'],
                       help="ControlNet type (default: depth)")
    parser.add_argument("--strength", type=float, default=0.4,
                       help="Denoising strength (0.3-0.5 recommended)")
    parser.add_argument("--seed", type=int, default=12345,
                       help="Fixed random seed for consistency")
    parser.add_argument("--fps", type=int, default=30,
                       help="Output video FPS")
    parser.add_argument("--crf", type=int, default=18,
                       help="Video quality (18=high, 23=default)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Initialize compositor
    compositor = ControlNetCompositor(
        controlnet_type=args.controlnet_type,
        device=args.device
    )

    # Process video sequence
    output_frames = compositor.process_video_sequence(
        render_dir=args.render_dir,
        background_dir=args.bg_dir,
        output_dir=args.output_dir,
        strength=args.strength,
        seed=args.seed
    )

    # Encode to video
    compositor.encode_video_ffmpeg(
        frame_dir=args.output_dir,
        output_path=args.video_output,
        fps=args.fps,
        crf=args.crf
    )

    print("\n" + "="*60)
    print("✓ VANELIA COMPOSITING COMPLETE")
    print("="*60)
    print(f"Final video: {args.video_output}")
    print(f"Total frames: {len(output_frames)}")
    print()


if __name__ == "__main__":
    main()
