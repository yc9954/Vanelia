#!/bin/bash
# Vanelia Environment Setup Script
# For RunPod NVIDIA A100 Ubuntu Server with CUDA 12.1

set -e

echo "========================================="
echo "Vanelia Environment Setup"
echo "========================================="

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA GPU not found!"
    exit 1
fi

echo "[GPU] Detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Update system
echo ""
echo "[1/6] Updating system packages..."
apt-get update -qq

# Install system dependencies
echo "[2/6] Installing system dependencies..."
apt-get install -y -qq \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    > /dev/null 2>&1

# Create Python virtual environment
echo "[3/6] Creating Python virtual environment..."
python3 -m pip install --upgrade pip -q
pip install virtualenv -q
python3 -m venv /opt/vanelia_env
source /opt/vanelia_env/bin/activate

# Install Python packages
echo "[4/6] Installing Python packages..."
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q -r /home/user/Vanelia/requirements.txt

# Install Dust3R
echo "[5/6] Installing Dust3R..."
cd /tmp
if [ ! -d "dust3r" ]; then
    git clone https://github.com/naver/dust3r.git --quiet
fi
cd dust3r
pip install -q -e .

# Install Blender
echo "[6/6] Installing Blender 4.0..."
bash /home/user/Vanelia/scripts/install_blender.sh

echo ""
echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Activate environment:"
echo "  source /opt/vanelia_env/bin/activate"
echo ""
echo "Verify installation:"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo "  blender --version"
echo ""
