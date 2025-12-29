"""
INR-Harmonization Compositor for Vanelia

This module provides a production-ready compositor based on the INR-Harmonization method
which can achieve arbitrary resolution video harmonization.

Paper: Dense Pixel-to-Pixel Harmonization via Continuous Image Representation (TCSVT 2023)
GitHub: https://github.com/WindVChen/INR-Harmonization
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from tqdm import tqdm

# Add INR-Harmonization to path
# Try environment variable first, then common locations
INR_HARMON_PATH = os.getenv('INR_HARMONIZATION_PATH')
if INR_HARMON_PATH and os.path.exists(INR_HARMON_PATH):
    sys.path.insert(0, INR_HARMON_PATH)
else:
    # Try common installation locations
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "INR-Harmonization",  # Sibling to Vanelia
        Path("/home/user/INR-Harmonization"),  # User home
        Path("/workspace/INR-Harmonization"),  # RunPod workspace
    ]
    for path in possible_paths:
        if path.exists():
            sys.path.insert(0, str(path))
            break


class INRHarmonizationCompositor:
    """
    INR-Harmonization based video compositor with pre-trained weights.

    This compositor uses Implicit Neural Representation (INR) for photorealistic
    video harmonization, supporting arbitrary resolution inputs.
    """

    def __init__(
        self,
        device='cuda',
        checkpoint_path=None,
        base_size=256,
        input_size=256,
        auto_download=True
    ):
        """
        Initialize INR-Harmonization compositor.

        Args:
            device (str): Device to run on ('cuda' or 'cpu')
            checkpoint_path (str): Path to Video_HYouTube_256.pth checkpoint
            base_size (int): Encoder input resolution
            input_size (int): Target output resolution
            auto_download (bool): Automatically download checkpoint if not found
        """
        self.device = device
        self.base_size = base_size
        self.input_size = input_size

        # Find or download checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint_path()

        if not Path(checkpoint_path).exists() and auto_download:
            print(f"[INR-Harmonization] Checkpoint not found, attempting download...")
            self._download_checkpoint()
            checkpoint_path = self._get_default_checkpoint_path()

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Please run: python scripts/download_inr_checkpoint.py"
            )

        self.checkpoint_path = checkpoint_path
        print(f"[INR-Harmonization] Using checkpoint: {checkpoint_path}")

        # Build model
        self.model = self._build_model()
        self._load_checkpoint()

        # Transforms
        self.transform_mean = [.5, .5, .5]
        self.transform_var = [.5, .5, .5]
        self.torch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.transform_mean, self.transform_var)
        ])

        print(f"[INR-Harmonization] Initialized on {device}")

    def _get_default_checkpoint_path(self):
        """Get default checkpoint path."""
        return str(Path(__file__).parent.parent.parent / "checkpoints" /
                   "inr_harmonization" / "Video_HYouTube_256.pth")

    def _download_checkpoint(self):
        """Download checkpoint if not exists."""
        try:
            import subprocess
            import sys

            # Find download script relative to this file
            script_path = Path(__file__).parent.parent.parent / "scripts" / "download_inr_checkpoint.py"

            if not script_path.exists():
                raise FileNotFoundError(f"Download script not found: {script_path}")

            print(f"[INR-Harmonization] Running checkpoint download script...")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)

        except Exception as e:
            print(f"[WARNING] Could not auto-download: {e}")
            print(f"[INFO] Please manually run: python scripts/download_inr_checkpoint.py")
            print(f"[INFO] Or download from: https://drive.google.com/file/d/1Tv9aahaPmJ_RGeYdawLCNWNGabZgJo6y/view")

    def _build_model(self):
        """Build the INR-Harmonization model."""
        try:
            from model.build_model import build_model
        except ImportError:
            raise ImportError(
                "INR-Harmonization not found. Please clone the repository:\n"
                f"cd {Path(__file__).parent.parent.parent.parent}\n"
                "git clone https://github.com/WindVChen/INR-Harmonization.git"
            )

        # Create options object
        class Options:
            def __init__(self):
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.base_size = 256
                self.input_size = 256
                self.INR_input_size = 256
                self.INR_MLP_dim = 32
                self.LUT_dim = 7
                self.activation = 'leakyrelu_pe'
                self.param_factorize_dim = 10
                self.embedding_type = "CIPS_embed"
                self.INRDecode = True
                self.isMoreINRInput = True
                self.hr_train = False
                self.isFullRes = False

        opt = Options()
        opt.base_size = self.base_size
        opt.input_size = self.input_size
        opt.INR_input_size = self.input_size
        opt.device = self.device

        model = build_model(opt).to(self.device)
        model.eval()
        return model

    def _load_checkpoint(self):
        """Load pre-trained checkpoint."""
        print(f"[INR-Harmonization] Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Load weights
        self.model.load_state_dict(state_dict, strict=False)
        print(f"[INR-Harmonization] Checkpoint loaded successfully")

    def _normalize(self, tensor, mode='fwd'):
        """Normalize/denormalize tensor."""
        if mode == 'fwd':
            return tensor
        elif mode == 'inv':
            # Denormalize: (x * std) + mean
            mean = torch.tensor(self.transform_mean).view(3, 1, 1).to(tensor.device)
            std = torch.tensor(self.transform_var).view(3, 1, 1).to(tensor.device)
            return tensor * std + mean
        return tensor

    def _create_inr_coordinates(self, height, width):
        """Create INR coordinate grid for given resolution."""
        # Create normalized coordinates [-1, 1]
        y_coords = torch.linspace(-1, 1, height)
        x_coords = torch.linspace(-1, 1, width)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Stack to (H, W, 2)
        coordinates = torch.stack([xx, yy], dim=-1)
        return coordinates

    def process_frame(self, background_frame, render_frame):
        """
        Process a single frame.

        Args:
            background_frame (np.ndarray): Background RGB image (H, W, 3), range [0, 255]
            render_frame (np.ndarray): Rendered RGBA image (H, W, 4), range [0, 255]

        Returns:
            np.ndarray: Harmonized RGB image (H, W, 3), range [0, 255]
        """
        H, W = background_frame.shape[:2]

        # Extract alpha mask and RGB from render
        alpha_mask = render_frame[:, :, 3:4] / 255.0  # (H, W, 1)
        render_rgb = render_frame[:, :, :3]  # (H, W, 3)

        # Create composite image (render over background)
        composite = (render_rgb * alpha_mask + background_frame * (1 - alpha_mask)).astype(np.uint8)

        # Convert mask to binary (>0.5 threshold)
        binary_mask = (alpha_mask > 0.5).astype(np.uint8) * 255  # (H, W, 1)

        # Resize to model input size
        composite_pil = Image.fromarray(composite)
        mask_pil = Image.fromarray(binary_mask.squeeze(), mode='L')

        composite_resized = composite_pil.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask_resized = mask_pil.resize((self.input_size, self.input_size), Image.NEAREST)

        # Convert to tensors
        composite_tensor = self.torch_transform(composite_resized).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        mask_tensor = transforms.ToTensor()(mask_resized).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # Create INR coordinates
        fg_coords = self._create_inr_coordinates(self.input_size, self.input_size)
        fg_coords = fg_coords.unsqueeze(0).to(self.device)  # (1, H, W, 2)

        # Run model
        with torch.no_grad():
            # Model returns: (harmonized_images_list, lut, lut_transformed_image)
            output = self.model(composite_tensor, mask_tensor, fg_coords)

            if isinstance(output, tuple) and len(output) >= 1:
                # Get the final harmonized output
                if isinstance(output[0], list):
                    harmonized = output[0][-1]  # Last in list is final output
                else:
                    harmonized = output[0]
            else:
                harmonized = output

        # Denormalize
        harmonized = self._normalize(harmonized, mode='inv')
        harmonized = torch.clamp(harmonized, 0, 1)

        # Convert back to numpy
        harmonized_np = harmonized.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        harmonized_np = (harmonized_np * 255).astype(np.uint8)

        # Resize back to original resolution
        harmonized_resized = cv2.resize(harmonized_np, (W, H), interpolation=cv2.INTER_LINEAR)

        # Blend: harmonized foreground + original background
        result = (harmonized_resized * alpha_mask + background_frame * (1 - alpha_mask)).astype(np.uint8)

        return result

    def process_video_sequence(
        self,
        render_dir,
        background_dir,
        output_dir,
        strength=1.0,
        seed=None
    ):
        """
        Process a video sequence.

        Args:
            render_dir (str): Directory containing rendered RGBA frames
            background_dir (str): Directory containing background RGB frames
            output_dir (str): Directory to save harmonized frames
            strength (float): Blending strength (0-1), currently unused
            seed (int): Random seed for reproducibility

        Returns:
            list: List of output frame paths
        """
        render_dir = Path(render_dir)
        background_dir = Path(background_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Get frame files
        render_frames = sorted(render_dir.glob("*.png"))
        if not render_frames:
            raise FileNotFoundError(f"No PNG frames found in {render_dir}")

        print(f"[INR-Harmonization] Processing {len(render_frames)} frames...")

        output_paths = []

        for render_path in tqdm(render_frames, desc="Harmonizing"):
            frame_name = render_path.name
            bg_path = background_dir / frame_name

            if not bg_path.exists():
                print(f"[WARNING] Background frame not found: {bg_path}, skipping")
                continue

            # Load frames
            render_frame = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)  # RGBA
            bg_frame = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)  # BGR

            if render_frame is None or bg_frame is None:
                print(f"[WARNING] Failed to load {frame_name}, skipping")
                continue

            # Convert BGR to RGB
            bg_frame_rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)

            if render_frame.shape[2] == 3:
                # Add alpha channel if missing
                render_frame_rgba = np.dstack([render_frame, np.ones(render_frame.shape[:2], dtype=np.uint8) * 255])
            else:
                render_frame_rgba = cv2.cvtColor(render_frame, cv2.COLOR_BGRA2RGBA)

            # Process frame
            harmonized = self.process_frame(bg_frame_rgb, render_frame_rgba)

            # Convert back to BGR for saving
            harmonized_bgr = cv2.cvtColor(harmonized, cv2.COLOR_RGB2BGR)

            # Save
            output_path = output_dir / frame_name
            cv2.imwrite(str(output_path), harmonized_bgr)
            output_paths.append(output_path)

        print(f"[INR-Harmonization] Saved {len(output_paths)} frames to {output_dir}")
        return output_paths

    def encode_video_ffmpeg(self, frame_dir: str, output_path: str,
                           fps: int = 30, crf: int = 18) -> str:
        """
        Encode frames to MP4 video using FFmpeg.

        Args:
            frame_dir (str): Directory containing output frames
            output_path (str): Path for output video file
            fps (int): Frames per second (default: 30)
            crf (int): Constant Rate Factor for quality (18=high, 23=default, 28=low)

        Returns:
            str: Path to encoded video file
        """
        import subprocess

        frame_dir = Path(frame_dir)
        frames = sorted(frame_dir.glob("*.png"))

        if not frames:
            raise FileNotFoundError(f"No PNG frames found in {frame_dir}")

        # Determine frame pattern from first frame
        first_frame = frames[0].name
        # Extract numbering pattern (e.g., frame_000001.png)
        import re
        match = re.search(r'(\d+)', first_frame)
        if match:
            num_digits = len(match.group(1))
            base_name = first_frame[:match.start()]
            extension = first_frame[match.end():]
            pattern = f"{base_name}%0{num_digits}d{extension}"
        else:
            # Fallback to generic pattern
            pattern = "frame_%06d.png"

        frame_pattern = str(frame_dir / pattern)

        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',  # Enable web streaming
            output_path
        ]

        print(f"\n[FFmpeg] Encoding video...")
        print(f"  Input pattern: {frame_pattern}")
        print(f"  Output: {output_path}")
        print(f"  FPS: {fps}, CRF: {crf}")
        print(f"  Total frames: {len(frames)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[FFmpeg] ✓ Video encoded successfully: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[FFmpeg] ✗ Encoding failed!")
            print(f"  stderr: {e.stderr}")
            raise RuntimeError(f"FFmpeg encoding failed: {e.stderr}") from e


if __name__ == "__main__":
    # Test the compositor
    print("Testing INR-Harmonization Compositor...")

    try:
        compositor = INRHarmonizationCompositor(device='cuda')
        print("[SUCCESS] Compositor initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
