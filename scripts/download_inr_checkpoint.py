#!/usr/bin/env python3
"""
Download INR-Harmonization checkpoint from Google Drive.

Usage:
    python scripts/download_inr_checkpoint.py
"""

import os
import gdown
import sys

def download_checkpoint():
    """Download Video_HYouTube_256.pth checkpoint from Google Drive."""

    # Create checkpoints directory using absolute path resolution
    from pathlib import Path
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    checkpoint_dir = project_root / "checkpoints" / "inr_harmonization"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "Video_HYouTube_256.pth"

    # Check if already downloaded
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Checkpoint already exists at: {checkpoint_path}")
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"[INFO] File size: {file_size:.2f} MB")
        return checkpoint_path

    # Google Drive file ID for Video_HYouTube_256.pth
    file_id = "1Tv9aahaPmJ_RGeYdawLCNWNGabZgJo6y"
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"[INFO] Downloading Video_HYouTube_256.pth checkpoint...")
    print(f"[INFO] URL: https://drive.google.com/file/d/{file_id}/view")
    print(f"[INFO] Destination: {checkpoint_path}")

    try:
        gdown.download(url, checkpoint_path, quiet=False)
        print(f"[SUCCESS] Downloaded checkpoint to: {checkpoint_path}")
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"[INFO] File size: {file_size:.2f} MB")
        return checkpoint_path
    except Exception as e:
        print(f"[ERROR] Failed to download checkpoint: {e}")
        print(f"\n[MANUAL] Please download manually from:")
        print(f"https://drive.google.com/file/d/{file_id}/view")
        print(f"Save it to: {checkpoint_path}")
        sys.exit(1)

if __name__ == "__main__":
    checkpoint_path = download_checkpoint()
    print(f"\n[INFO] Checkpoint ready at: {checkpoint_path}")
