"""
Module C: IC-Light Temporal-Consistent Compositor
Composites rendered objects with background using IC-Light for relighting.
Ensures temporal consistency with fixed seed and low denoising strength.

IC-Light Repository: https://github.com/lllyasviel/IC-Light
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import subprocess

# Diffusers imports
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from diffusers.utils import load_image


class ICLightCompositor:
    """
    Temporal-consistent video compositor using IC-Light.
    Prevents flickering through fixed seeds and low denoising strength.
    """

    def __init__(self,
                 model_id: str = "lllyasviel/ic-light-sd15-fc",
                 device: str = "cuda",
                 torch_dtype=torch.float16):
        """
        Initialize IC-Light pipeline.

        Args:
            model_id: HuggingFace model ID for IC-Light
            device: Device to run on
            torch_dtype: Torch dtype for inference
        """
        self.device = device
        self.dtype = torch_dtype

        print(f"[IC-Light] Loading model: {model_id}")

        # Load IC-Light pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None
        ).to(device)

        # Use DDIM scheduler for better temporal consistency
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Enable optimizations
        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                print("[IC-Light] xformers not available, skipping")

        print("[IC-Light] ✓ Model loaded successfully\n")

    def alpha_blend(self, foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
        """
        Alpha blend foreground (RGBA) with background (RGB).

        Args:
            foreground: RGBA image (H, W, 4)
            background: RGB image (H, W, 3)

        Returns:
            Blended RGB image
        """
        # Ensure same size
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
                     strength: float = 0.25,
                     guidance_scale: float = 7.5,
                     seed: int = 12345,
                     prompt: str = "high quality, photorealistic, cinematic lighting, 4k, natural shadows",
                     negative_prompt: str = "flickering, jittering, distorted, cartoon, blurry, artifacts",
                     num_inference_steps: int = 20,
                     previous_latent: Optional[torch.Tensor] = None,
                     latent_blend_ratio: float = 0.0) -> Tuple[Image.Image, torch.Tensor]:
        """
        Process single frame with IC-Light.

        Args:
            foreground_path: Path to rendered RGBA frame
            background_path: Path to background frame
            strength: Denoising strength (0.2-0.3 for consistency)
            guidance_scale: CFG scale
            seed: Random seed (MUST be fixed for consistency)
            prompt: Positive prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of diffusion steps
            previous_latent: Previous frame's latent for blending
            latent_blend_ratio: Ratio to blend previous latent (0.0-0.2)

        Returns:
            Tuple of (output image, current latent)
        """
        # Load images
        fg_img = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
        bg_img = cv2.imread(background_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

        # 1st pass: Alpha blend
        composite = self.alpha_blend(fg_img, bg_img)
        composite_pil = Image.fromarray(composite)

        # Setup generator with FIXED seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run IC-Light refinement
        with torch.no_grad():
            # Encode input
            latents = self.pipe.vae.encode(
                self.pipe.image_processor.preprocess(composite_pil).to(
                    device=self.device, dtype=self.dtype
                )
            ).latent_dist.sample(generator=generator)
            latents = latents * self.pipe.vae.config.scaling_factor

            # Optional: Blend with previous frame's latent
            if previous_latent is not None and latent_blend_ratio > 0:
                latents = (1 - latent_blend_ratio) * latents + latent_blend_ratio * previous_latent

            # Denoise with LOW strength
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=composite_pil,
                strength=strength,  # CRITICAL: Keep low (0.2-0.3)
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil"
            )

            result_img = output.images[0]

        return result_img, latents

    def process_video_sequence(self,
                              render_dir: str,
                              background_dir: str,
                              output_dir: str,
                              strength: float = 0.25,
                              seed: int = 12345,
                              latent_blend_ratio: float = 0.15,
                              prompt: str = "high quality, photorealistic, cinematic lighting, 4k, blend with background",
                              negative_prompt: str = "flickering, jittering, distorted, cartoon") -> List[str]:
        """
        Process complete video sequence with temporal consistency.

        Args:
            render_dir: Directory with Blender renders (RGBA)
            background_dir: Directory with background frames
            output_dir: Output directory for refined frames
            strength: Denoising strength (LOW for consistency)
            seed: Fixed seed for all frames
            latent_blend_ratio: Blend ratio with previous latent (0.1-0.2)
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
        print(f"\n[IC-Light] Processing {num_frames} frames...")
        print(f"[IC-Light] Settings:")
        print(f"  - Strength: {strength} (LOW for consistency)")
        print(f"  - Seed: {seed} (FIXED)")
        print(f"  - Latent blend: {latent_blend_ratio}")
        print()

        output_paths = []
        previous_latent = None

        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            frame_iter = tqdm(enumerate(zip(render_frames, bg_frames)),
                            total=num_frames,
                            desc="IC-Light Processing",
                            unit="frame")
        except ImportError:
            frame_iter = enumerate(zip(render_frames, bg_frames))

        for idx, (render_path, bg_path) in frame_iter:
            # Process frame
            output_img, current_latent = self.process_frame(
                foreground_path=str(render_path),
                background_path=str(bg_path),
                strength=strength,
                seed=seed,  # SAME seed for all frames
                previous_latent=previous_latent,
                latent_blend_ratio=latent_blend_ratio,
                prompt=prompt,
                negative_prompt=negative_prompt
            )

            # Save output
            output_path = os.path.join(output_dir, f"refined_{idx:06d}.png")
            output_img.save(output_path)
            output_paths.append(output_path)

            # Memory optimization: Move latent to CPU
            if previous_latent is not None:
                del previous_latent
            previous_latent = current_latent.cpu()

            # Clear CUDA cache periodically
            if idx % 10 == 0:
                torch.cuda.empty_cache()

            # Progress (fallback for no tqdm)
            if 'tqdm' not in sys.modules:
                progress = ((idx + 1) / num_frames) * 100
                print(f"  [{progress:5.1f}%] Frame {idx+1}/{num_frames} → {output_path}")

        print(f"\n[IC-Light] ✓ Refined {num_frames} frames")
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
            "-y",  # Overwrite
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

    parser = argparse.ArgumentParser(description="IC-Light Temporal-Consistent Compositor")
    parser.add_argument("--render-dir", type=str, required=True,
                       help="Directory with Blender renders (RGBA)")
    parser.add_argument("--bg-dir", type=str, required=True,
                       help="Directory with background frames")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for refined frames")
    parser.add_argument("--video-output", type=str, default="final_output.mp4",
                       help="Final video output path")
    parser.add_argument("--strength", type=float, default=0.25,
                       help="Denoising strength (0.2-0.3 recommended)")
    parser.add_argument("--seed", type=int, default=12345,
                       help="Fixed random seed for consistency")
    parser.add_argument("--latent-blend", type=float, default=0.15,
                       help="Latent blending ratio (0.0-0.2)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Output video FPS")
    parser.add_argument("--crf", type=int, default=18,
                       help="Video quality (18=high, 23=default)")
    parser.add_argument("--model", type=str, default="lllyasviel/ic-light-sd15-fc",
                       help="IC-Light model ID")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Initialize compositor
    compositor = ICLightCompositor(model_id=args.model, device=args.device)

    # Process video sequence
    output_frames = compositor.process_video_sequence(
        render_dir=args.render_dir,
        background_dir=args.bg_dir,
        output_dir=args.output_dir,
        strength=args.strength,
        seed=args.seed,
        latent_blend_ratio=args.latent_blend
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
