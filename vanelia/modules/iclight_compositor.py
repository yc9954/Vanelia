"""
Module C: ControlNet Inpaint Temporal-Consistent Compositor
Composites rendered objects with background using ControlNet Inpaint for relighting.
Ensures temporal consistency with fixed seed and low denoising strength.

Uses ControlNet Inpaint instead of IC-Light for better stability and compatibility.
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import subprocess

# Diffusers imports
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image


class ICLightCompositor:
    """
    Temporal-consistent video compositor using ControlNet Inpaint.
    Prevents flickering through fixed seeds and low denoising strength.
    """

    def __init__(self,
                 device: str = "cuda",
                 torch_dtype=torch.float16):
        """
        Initialize ControlNet Inpaint pipeline.

        Args:
            device: Device to run on
            torch_dtype: Torch dtype for inference
        """
        self.device = device
        self.dtype = torch_dtype

        print("[ControlNet Inpaint] Loading models...")

        # Load ControlNet Inpaint model (most powerful and verified model)
        try:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint",
                torch_dtype=torch_dtype
            )
            print("[ControlNet Inpaint] ✓ ControlNet loaded")

            # Load pipeline with ControlNet
            self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                safety_checker=None
            ).to(device)
            print("[ControlNet Inpaint] ✓ Pipeline loaded")
        except Exception as e:
            print(f"[ControlNet Inpaint] WARNING: Failed to load ControlNet: {e}")
            print("[ControlNet Inpaint] Falling back to standard Stable Diffusion Inpaint")
            # Fallback to standard Inpaint (without ControlNet)
            from diffusers import StableDiffusionInpaintPipeline
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch_dtype,
                safety_checker=None
            ).to(device)
            self.use_controlnet = False
            print("[ControlNet Inpaint] ✓ Fallback pipeline loaded")
        else:
            self.use_controlnet = True

        # Use DPM++ 2M Karras scheduler (faster than DDIM, same quality)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            use_karras_sigmas=True
        )
        print("[ControlNet Inpaint] ✓ Using DPM++ 2M Karras scheduler (faster)")

        # Enable optimizations for speed
        self.pipe.enable_attention_slicing(1)  # Slice attention for memory efficiency
        
        # Check if xformers is available
        xformers_enabled = False
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                xformers_enabled = True
                print("[ControlNet Inpaint] ✓ xformers enabled")
            except Exception:
                print("[ControlNet Inpaint] xformers not available, skipping")
        
        # VAE optimizations
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
            print("[ControlNet Inpaint] ✓ VAE slicing enabled")
        if hasattr(self.pipe, 'enable_vae_tiling'):
            self.pipe.enable_vae_tiling()
            print("[ControlNet Inpaint] ✓ VAE tiling enabled")
        
        # Compile model for faster inference (PyTorch 2.0+)
        # NOTE: torch.compile is incompatible with xformers flash attention
        # Only compile if xformers is not enabled
        if not xformers_enabled:
            try:
                if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
                    # Suppress dynamo errors for compatibility
                    import torch._dynamo
                    torch._dynamo.config.suppress_errors = True
                    
                    self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                    if hasattr(self.pipe, 'controlnet') and self.pipe.controlnet is not None:
                        self.pipe.controlnet = torch.compile(self.pipe.controlnet, mode="reduce-overhead")
                    print("[ControlNet Inpaint] ✓ Model compiled with torch.compile")
            except Exception as e:
                print(f"[ControlNet Inpaint] torch.compile not available: {e}")
        else:
            # When xformers is enabled, suppress dynamo errors to avoid conflicts
            try:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                print("[ControlNet Inpaint] ✓ Dynamo errors suppressed (xformers compatibility)")
            except Exception:
                pass

        print("[ControlNet Inpaint] ✓ Model loaded successfully\n")

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

    def create_mask_from_alpha(self, foreground: np.ndarray) -> np.ndarray:
        """
        Create mask from RGBA foreground's alpha channel.
        Only masks the actual object, not shadow catcher areas.

        Args:
            foreground: RGBA image (H, W, 4)

        Returns:
            Binary mask (H, W) where object is white (255) and background is black (0)
        """
        if foreground.shape[2] == 4:
            # Extract alpha channel and RGB channels
            alpha = foreground[:, :, 3]
            rgb = foreground[:, :, :3]
            
            # Create mask: object should have high alpha AND visible RGB content
            # Shadow catcher has shadows but low alpha, so we use a higher threshold
            # Only mask pixels with significant alpha (object) and visible content
            alpha_threshold = 50  # Higher threshold to exclude shadow catcher artifacts
            mask = (alpha > alpha_threshold).astype(np.uint8) * 255
            
            # Additional check: ensure RGB content exists (not just transparent)
            # This helps exclude shadow-only areas
            rgb_brightness = np.mean(rgb, axis=2)
            has_content = rgb_brightness > 10  # Minimum brightness threshold
            mask = mask & (has_content.astype(np.uint8) * 255)
        else:
            # No alpha channel, create minimal mask (just center area as fallback)
            h, w = foreground.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            # Create a small center mask as fallback
            center_h, center_w = h // 2, w // 2
            mask_size = min(h, w) // 4
            mask[center_h-mask_size:center_h+mask_size, center_w-mask_size:center_w+mask_size] = 255

        return mask

    def process_frame(self,
                     foreground_path: str,
                     background_path: str,
                     strength: float = 0.3,
                     guidance_scale: float = 7.5,
                     seed: int = 12345,
                     prompt: str = "high quality, photorealistic, cinematic lighting, 4k, natural shadows, blend seamlessly with background",
                     negative_prompt: str = "flickering, jittering, distorted, cartoon, blurry, artifacts, unnatural lighting",
                     num_inference_steps: int = 12,
                     previous_latent: Optional[torch.Tensor] = None,
                     latent_blend_ratio: float = 0.0) -> Tuple[Image.Image, torch.Tensor]:
        """
        Process single frame with ControlNet Inpaint.

        Args:
            foreground_path: Path to rendered RGBA frame
            background_path: Path to background frame
            strength: Denoising strength (0.3 recommended for ControlNet)
            guidance_scale: CFG scale
            seed: Random seed (MUST be fixed for consistency)
            prompt: Positive prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of diffusion steps (12 recommended for DPM++, same quality as 20 with DDIM)
            previous_latent: Previous frame's latent for blending (not used in ControlNet)
            latent_blend_ratio: Ratio to blend previous latent (not used in ControlNet)

        Returns:
            Tuple of (output image, current latent)
        """
        # Load images
        fg_img = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
        bg_img = cv2.imread(background_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

        # Ensure both images have the same size
        # Use foreground size as reference (Blender render resolution)
        target_height, target_width = fg_img.shape[:2]
        if bg_img.shape[:2] != (target_height, target_width):
            bg_img = cv2.resize(bg_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            print(f"[ControlNet Inpaint] Resized background from {bg_img.shape[:2]} to ({target_height}, {target_width})")

        # 1st pass: Alpha blend to create initial composite
        composite = self.alpha_blend(fg_img, bg_img)
        composite_pil = Image.fromarray(composite)

        # For low strength (0.3 or less), skip ControlNet and just return alpha blend
        # This avoids mask issues and preserves Blender's quality
        if strength <= 0.1:
            print(f"[ControlNet Inpaint] Strength {strength} is very low, using alpha blend only")
            result_img = composite_pil
            latents = torch.zeros((1, 4, result_img.height // 8, result_img.width // 8),
                                 device=self.device, dtype=self.dtype)
            return result_img, latents

        # Create mask from alpha channel (object area = white, background = black)
        mask = self.create_mask_from_alpha(fg_img)
        mask_pil = Image.fromarray(mask).convert("RGB")
        
        # Ensure mask and composite have exactly the same size
        if composite_pil.size != mask_pil.size:
            mask_pil = mask_pil.resize(composite_pil.size, Image.LANCZOS)
            print(f"[ControlNet Inpaint] Resized mask to match composite: {composite_pil.size}")

        # Check if mask is too large (covers most of image - indicates problem)
        mask_array = np.array(mask_pil.convert("L"))
        mask_coverage = np.sum(mask_array > 128) / (mask_array.shape[0] * mask_array.shape[1])
        
        if mask_coverage > 0.8:
            print(f"[ControlNet Inpaint] WARNING: Mask covers {mask_coverage*100:.1f}% of image, using alpha blend only")
            result_img = composite_pil
            latents = torch.zeros((1, 4, result_img.height // 8, result_img.width // 8),
                                 device=self.device, dtype=self.dtype)
            return result_img, latents

        # Setup generator with FIXED seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run ControlNet Inpaint refinement
        with torch.no_grad():
            if self.use_controlnet:
                # ControlNet Inpaint: use composite as control_image
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=composite_pil,
                    mask_image=mask_pil,
                    control_image=composite_pil,  # Use composite as control for consistency
                    strength=strength,  # 0.3 recommended to preserve Blender geometry
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type="pil"
                )
            else:
                # Standard Inpaint (fallback)
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=composite_pil,
                    mask_image=mask_pil,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type="pil"
                )

            result_img = output.images[0]

            # Get latent for temporal consistency (if needed)
            # Note: ControlNet doesn't expose latents easily, so we create a dummy one
            latents = torch.zeros((1, 4, result_img.height // 8, result_img.width // 8),
                                 device=self.device, dtype=self.dtype)

        return result_img, latents

    def process_batch(self,
                    foreground_paths: List[str],
                    background_paths: List[str],
                    strength: float = 0.3,
                    seed: int = 12345,
                    prompt: str = "high quality, photorealistic, cinematic lighting, 4k, natural shadows, blend seamlessly with background",
                    negative_prompt: str = "flickering, jittering, distorted, cartoon, blurry, artifacts, unnatural lighting",
                    num_inference_steps: int = 12) -> List[Image.Image]:
        """
        Process multiple frames in batch for faster inference.
        
        Args:
            foreground_paths: List of rendered RGBA frame paths
            background_paths: List of background frame paths
            strength: Denoising strength
            seed: Random seed
            prompt: Positive prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of diffusion steps
        
        Returns:
            List of output images
        """
        batch_size = len(foreground_paths)
        assert len(foreground_paths) == len(background_paths), "Batch size mismatch"
        
        # Load all images
        composites = []
        masks = []
        valid_indices = []
        
        for fg_path, bg_path in zip(foreground_paths, background_paths):
            fg_img = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
            bg_img = cv2.imread(bg_path)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            
            # Resize to match
            target_height, target_width = fg_img.shape[:2]
            if bg_img.shape[:2] != (target_height, target_width):
                bg_img = cv2.resize(bg_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Alpha blend
            composite = self.alpha_blend(fg_img, bg_img)
            composite_pil = Image.fromarray(composite)
            composites.append(composite_pil)
            
            # Create mask
            mask = self.create_mask_from_alpha(fg_img)
            mask_pil = Image.fromarray(mask).convert("RGB")
            if composite_pil.size != mask_pil.size:
                mask_pil = mask_pil.resize(composite_pil.size, Image.LANCZOS)
            masks.append(mask_pil)
            valid_indices.append(True)
        
        # Check if any masks are too large
        filtered_composites = []
        filtered_masks = []
        filtered_indices = []
        
        for i, (comp, mask) in enumerate(zip(composites, masks)):
            mask_array = np.array(mask.convert("L"))
            mask_coverage = np.sum(mask_array > 128) / (mask_array.shape[0] * mask_array.shape[1])
            
            if mask_coverage > 0.8:
                # Skip ControlNet for this frame, use alpha blend
                filtered_composites.append(None)  # Mark for alpha blend
                filtered_masks.append(None)
                filtered_indices.append(i)
            else:
                filtered_composites.append(comp)
                filtered_masks.append(mask)
                filtered_indices.append(i)
        
        # Process valid frames in batch
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.no_grad():
            if self.use_controlnet:
                # Batch ControlNet Inpaint
                valid_composites = [c for c in filtered_composites if c is not None]
                valid_masks = [m for m in filtered_masks if m is not None]
                
                if len(valid_composites) > 0:
                    output = self.pipe(
                        prompt=[prompt] * len(valid_composites),
                        negative_prompt=[negative_prompt] * len(valid_composites),
                        image=valid_composites,
                        mask_image=valid_masks,
                        control_image=valid_composites,
                        strength=strength,
                        guidance_scale=7.5,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        output_type="pil"
                    )
                    batch_results = output.images
                else:
                    batch_results = []
            else:
                # Standard Inpaint batch
                valid_composites = [c for c in filtered_composites if c is not None]
                valid_masks = [m for m in filtered_masks if m is not None]
                
                if len(valid_composites) > 0:
                    output = self.pipe(
                        prompt=[prompt] * len(valid_composites),
                        negative_prompt=[negative_prompt] * len(valid_composites),
                        image=valid_composites,
                        mask_image=valid_masks,
                        strength=strength,
                        guidance_scale=7.5,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        output_type="pil"
                    )
                    batch_results = output.images
                else:
                    batch_results = []
        
        # Reconstruct full batch results
        results = []
        batch_idx = 0
        for i, comp in enumerate(filtered_composites):
            if comp is None:
                # Use alpha blend
                results.append(composites[i])
            else:
                results.append(batch_results[batch_idx])
                batch_idx += 1
        
        return results

    def process_video_sequence(self,
                              render_dir: str,
                              background_dir: str,
                              output_dir: str,
                              strength: float = 0.3,
                              seed: int = 12345,
                              latent_blend_ratio: float = 0.15,
                              prompt: str = "high quality, photorealistic, cinematic lighting, 4k, blend seamlessly with background",
                              negative_prompt: str = "flickering, jittering, distorted, cartoon",
                              batch_size: int = 4) -> List[str]:
        """
        Process complete video sequence with temporal consistency using batch processing.

        Args:
            render_dir: Directory with Blender renders (RGBA)
            background_dir: Directory with background frames
            output_dir: Output directory for refined frames
            strength: Denoising strength (0.3 recommended for ControlNet)
            seed: Fixed seed for all frames
            latent_blend_ratio: Blend ratio with previous latent (not used in ControlNet)
            prompt: Positive prompt
            negative_prompt: Negative prompt
            batch_size: Number of frames to process in parallel (default: 4, increase for A100)

        Returns:
            List of output frame paths
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get frame lists
        render_frames = sorted(Path(render_dir).glob("*.png"))
        bg_frames = sorted(Path(background_dir).glob("*.jpg")) + \
                   sorted(Path(background_dir).glob("*.png"))

        # Handle frame count mismatch (can happen when resuming from previous runs)
        if len(render_frames) != len(bg_frames):
            print(f"\n⚠ WARNING: Frame count mismatch detected:")
            print(f"  - Render frames: {len(render_frames)}")
            print(f"  - Background frames: {len(bg_frames)}")
            
            # Use the minimum count to avoid errors
            min_count = min(len(render_frames), len(bg_frames))
            if min_count == 0:
                raise ValueError(f"No frames found! Renders: {len(render_frames)}, Backgrounds: {len(bg_frames)}")
            
            print(f"  - Using {min_count} frames (matching minimum)")
            
            # Trim to match the minimum count
            render_frames = render_frames[:min_count]
            bg_frames = bg_frames[:min_count]
            
            if len(render_frames) < len(bg_frames):
                print(f"  - Trimming background frames from {len(bg_frames)} to {min_count}")
            elif len(bg_frames) < len(render_frames):
                print(f"  - Trimming render frames from {len(render_frames)} to {min_count}")
                print(f"  - Consider re-running Step 2 (rendering) to generate all frames")

        num_frames = len(render_frames)
        print(f"\n[ControlNet Inpaint] Processing {num_frames} frames with batch size {batch_size}...")
        print(f"[ControlNet Inpaint] Settings:")
        print(f"  - Strength: {strength} (0.3 recommended for ControlNet)")
        print(f"  - Seed: {seed} (FIXED)")
        print(f"  - Batch size: {batch_size} (faster inference)")
        print(f"  - Steps: 12 (DPM++ scheduler)")
        print()

        output_paths = []
        
        # Process in batches
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            batch_render_paths = [str(render_frames[i]) for i in range(batch_start, batch_end)]
            batch_bg_paths = [str(bg_frames[i]) for i in range(batch_start, batch_end)]
            
            # Process batch
            batch_results = self.process_batch(
                foreground_paths=batch_render_paths,
                background_paths=batch_bg_paths,
                strength=strength,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=12
            )
            
            # Save batch results
            for i, result_img in enumerate(batch_results):
                frame_idx = batch_start + i
                output_path = os.path.join(output_dir, f"refined_{frame_idx:06d}.png")
                result_img.save(output_path)
                output_paths.append(output_path)
                
                progress = ((frame_idx + 1) / num_frames) * 100
                print(f"  [{progress:5.1f}%] Frame {frame_idx+1}/{num_frames} → {output_path}")

        print(f"\n[ControlNet Inpaint] ✓ Refined {num_frames} frames (batch processing)")
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

    parser = argparse.ArgumentParser(description="ControlNet Inpaint Temporal-Consistent Compositor")
    parser.add_argument("--render-dir", type=str, required=True,
                       help="Directory with Blender renders (RGBA)")
    parser.add_argument("--bg-dir", type=str, required=True,
                       help="Directory with background frames")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for refined frames")
    parser.add_argument("--video-output", type=str, default="final_output.mp4",
                       help="Final video output path")
    parser.add_argument("--strength", type=float, default=0.3,
                       help="Denoising strength (0.3 recommended for ControlNet)")
    parser.add_argument("--seed", type=int, default=12345,
                       help="Fixed random seed for consistency")
    parser.add_argument("--latent-blend", type=float, default=0.15,
                       help="Latent blending ratio (not used in ControlNet)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Output video FPS")
    parser.add_argument("--crf", type=int, default=18,
                       help="Video quality (18=high, 23=default)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Initialize compositor
    compositor = ICLightCompositor(device=args.device)

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
