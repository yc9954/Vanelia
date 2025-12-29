"""
Photorealistic Video Compositor
Based on "Anything in Any Scene" paper methodology
Implements coarse-to-fine style transfer with contextual attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from typing import List, Optional
import cv2
from tqdm import tqdm


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1, activation='elu'):
    """Generate convolution layer with optional dilation"""
    return GatedConv2d(input_dim, output_dim, kernel_size, stride,
                       padding=padding, dilation=rate, activation=activation)


class GatedConv2d(nn.Module):
    """Gated convolution for better feature learning"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, activation='elu', use_spectral_norm=True):
        super(GatedConv2d, self).__init__()

        self.activation = activation

        # Feature convolution
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                        padding, dilation=dilation)

        # Gating convolution
        gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                             padding, dilation=dilation)

        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(conv)
            self.gate_conv = nn.utils.spectral_norm(gate_conv)
        else:
            self.conv = conv
            self.gate_conv = gate_conv

    def forward(self, x):
        if self.activation == 'elu':
            activation = F.elu(self.conv(x))
        elif self.activation == 'relu':
            activation = F.relu(self.conv(x))
        else:
            activation = self.conv(x)

        gating = torch.sigmoid(self.gate_conv(x))
        return activation * gating


class CoarseGenerator(nn.Module):
    """Coarse generator using multi-scale dilated convolutions"""
    def __init__(self, input_dim=3, cnum=48):
        super(CoarseGenerator, self).__init__()

        # Encoder
        self.conv1 = gen_conv(input_dim + 1 + 3, cnum, 5, 1, 2)  # +1 mask +3 obj
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        # Multi-scale dilated convolutions
        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        # Decoder
        self.conv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, x, mask, obj):
        """
        Args:
            x: Background with object region masked out (B, 3, H, W)
            mask: Binary mask (B, 1, H, W)
            obj: Rendered object (B, 3, H, W)
        """
        # Concatenate inputs
        x = self.conv1(torch.cat([x, mask, obj], dim=1))
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # Multi-scale context
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)

        # Decode
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)

        x_stage1 = torch.clamp(x, -1., 1.)
        return x_stage1


class FineGenerator(nn.Module):
    """Fine generator with contextual attention for detail refinement"""
    def __init__(self, input_dim=3, cnum=48):
        super(FineGenerator, self).__init__()

        # Conv branch
        self.conv1 = gen_conv(input_dim + 1 + 3, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        self.conv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*2, 3, 2, 1)
        self.conv5 = gen_conv(cnum*2, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        # Attention branch (simplified - full contextual attention requires more code)
        self.pmconv1 = gen_conv(input_dim + 1 + 3, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        self.pmconv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        self.pmconv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, activation='relu')
        self.pmconv9 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        # Merge branches
        self.allconv11 = gen_conv(cnum*8, cnum*4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask, obj):
        """Refine coarse result with attention mechanism"""
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        xnow = torch.cat([x1_inpaint, mask, obj], dim=1)

        # Conv branch
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x

        # Attention branch (simplified)
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        # Note: Full contextual attention would go here
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x

        # Merge
        x = torch.cat([x_hallu, pm], dim=1)
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)

        x_stage2 = torch.clamp(x, -1., 1.)
        return x_stage2


class PhotorealisticGenerator(nn.Module):
    """Complete coarse-to-fine generator"""
    def __init__(self, input_dim=3, cnum=48):
        super(PhotorealisticGenerator, self).__init__()
        self.coarse_generator = CoarseGenerator(input_dim, cnum)
        self.fine_generator = FineGenerator(input_dim, cnum)

    def forward(self, x, mask, obj):
        x_stage1 = self.coarse_generator(x, mask, obj)
        x_stage2 = self.fine_generator(x, x_stage1, mask, obj)
        return x_stage1, x_stage2


class PhotorealisticCompositor:
    """
    Photorealistic video compositor using coarse-to-fine style transfer
    Based on "Anything in Any Scene" methodology
    """

    def __init__(self, device='cuda', cnum=48, checkpoint_path=None):
        """
        Initialize photorealistic compositor

        Args:
            device: Device to run on ('cuda' or 'cpu')
            cnum: Base number of channels (controls model capacity)
            checkpoint_path: Path to pre-trained model checkpoint
        """
        self.device = device
        self.model = PhotorealisticGenerator(input_dim=3, cnum=cnum).to(device)

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[Compositor] Loading checkpoint from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print("[Compositor] No checkpoint loaded - using random initialization")
            print("[Compositor] WARNING: Results will be poor without a trained model!")

        self.model.eval()

    def process_frame(self, background_frame: np.ndarray,
                     render_frame: np.ndarray,
                     alpha_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process a single frame pair

        Args:
            background_frame: Background image (H, W, 3) in range [0, 255]
            render_frame: Rendered object with alpha (H, W, 4) in range [0, 255]
            alpha_mask: Optional pre-computed alpha mask (H, W) in range [0, 255]

        Returns:
            Composited frame (H, W, 3) in range [0, 255]
        """
        # Extract alpha channel
        if alpha_mask is None:
            if render_frame.shape[2] == 4:
                alpha_mask = render_frame[:, :, 3]
            else:
                raise ValueError("render_frame must have alpha channel or alpha_mask must be provided")

        # Normalize to [0, 1]
        bg = background_frame.astype(np.float32) / 255.0
        obj_rgb = render_frame[:, :, :3].astype(np.float32) / 255.0
        mask = alpha_mask.astype(np.float32) / 255.0

        # Create masked background (object region set to 0)
        bg_masked = bg * (1 - mask[:, :, np.newaxis])

        # Convert to PyTorch tensors
        bg_tensor = torch.from_numpy(bg_masked.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        obj_tensor = torch.from_numpy(obj_rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)

        # Normalize to [-1, 1]
        bg_tensor = bg_tensor * 2 - 1
        obj_tensor = obj_tensor * 2 - 1

        # Run model
        with torch.no_grad():
            _, result = self.model(bg_tensor, mask_tensor, obj_tensor)

        # Denormalize to [0, 1]
        result = (result + 1) / 2
        result = torch.clamp(result, 0, 1)

        # Convert back to numpy
        result_np = result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        result_np = (result_np * 255).astype(np.uint8)

        return result_np

    def process_video_sequence(self, render_dir: str, background_dir: str,
                               output_dir: str, strength: float = 1.0,
                               seed: int = 12345) -> List[str]:
        """
        Process a sequence of frames

        Args:
            render_dir: Directory containing rendered object frames (RGBA PNGs)
            background_dir: Directory containing background frames
            output_dir: Directory to save composited frames
            strength: Blending strength (0-1, currently unused but kept for compatibility)
            seed: Random seed (kept for compatibility)

        Returns:
            List of output frame paths
        """
        render_path = Path(render_dir)
        bg_path = Path(background_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all render frames
        render_frames = sorted(render_path.glob('*.png'))
        bg_frames = sorted(bg_path.glob('*.jpg')) + sorted(bg_path.glob('*.png'))

        if len(render_frames) == 0:
            raise ValueError(f"No PNG files found in {render_dir}")
        if len(bg_frames) == 0:
            raise ValueError(f"No image files found in {background_dir}")
        if len(render_frames) != len(bg_frames):
            print(f"WARNING: Frame count mismatch - render: {len(render_frames)}, bg: {len(bg_frames)}")

        output_frames = []
        print(f"\n[Compositor] Processing {len(render_frames)} frames...")

        for i, (render_frame_path, bg_frame_path) in enumerate(tqdm(
            zip(render_frames, bg_frames), total=len(render_frames), desc="Compositing")):

            # Load frames
            render_frame = cv2.imread(str(render_frame_path), cv2.IMREAD_UNCHANGED)
            bg_frame = cv2.imread(str(bg_frame_path))

            # Convert BGR to RGB
            if render_frame.shape[2] == 4:
                render_frame = cv2.cvtColor(render_frame, cv2.COLOR_BGRA2RGBA)
            bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)

            # Resize background to match render size if needed
            if bg_frame.shape[:2] != render_frame.shape[:2]:
                bg_frame = cv2.resize(bg_frame, (render_frame.shape[1], render_frame.shape[0]))

            # Process frame
            result = self.process_frame(bg_frame, render_frame)

            # Save result
            output_frame_path = output_path / f"frame_{i:06d}.png"
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_frame_path), result_bgr)
            output_frames.append(str(output_frame_path))

        print(f"[Compositor] ✓ Processed {len(output_frames)} frames")
        return output_frames

    def encode_video_ffmpeg(self, frame_dir: str, output_path: str,
                           fps: int = 30, crf: int = 18) -> str:
        """
        Encode frames to video using FFmpeg

        Args:
            frame_dir: Directory containing frames
            output_path: Output video path
            fps: Frames per second
            crf: Quality (18=high, 23=medium, 28=low)

        Returns:
            Path to output video
        """
        import subprocess

        frame_pattern = str(Path(frame_dir) / "frame_%06d.png")

        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            output_path
        ]

        print(f"\n[FFmpeg] Encoding video: {output_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[FFmpeg] ✓ Video saved: {output_path}")

        return output_path


if __name__ == "__main__":
    # Test the compositor
    compositor = PhotorealisticCompositor(device='cuda')
    print("PhotorealisticCompositor initialized successfully!")
