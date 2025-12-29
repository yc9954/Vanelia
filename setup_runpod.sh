#!/bin/bash
# RunPod Setup Script for Vanelia
# Optimized for RunPod NVIDIA A100 environment

set -e  # Exit on error

echo "=================================================="
echo "  Vanelia RunPod Setup"
echo "=================================================="

# 1. Install Blender 4.0.2
echo ""
echo "[1/5] Installing Blender 4.0.2..."
BLENDER_VERSION="4.0.2"
BLENDER_URL="https://download.blender.org/release/Blender4.0/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_DIR="/workspace/blender"

if [ ! -d "$BLENDER_DIR" ]; then
    cd /workspace
    wget -q --show-progress "${BLENDER_URL}"
    tar -xf "blender-${BLENDER_VERSION}-linux-x64.tar.xz"
    mv "blender-${BLENDER_VERSION}-linux-x64" blender
    rm "blender-${BLENDER_VERSION}-linux-x64.tar.xz"

    # Add to PATH
    echo 'export PATH="/workspace/blender:$PATH"' >> ~/.bashrc
    export PATH="/workspace/blender:$PATH"

    echo "✓ Blender installed to /workspace/blender"
else
    echo "✓ Blender already installed"
fi

# 2. Install Python dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
cd /workspace/Vanelia
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Python dependencies installed"

# 3. Install Dust3R
echo ""
echo "[3/5] Installing Dust3R..."
if [ ! -d "/workspace/dust3r" ]; then
    cd /workspace
    git clone https://github.com/naver/dust3r.git
    cd dust3r

    # Install dependencies if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi

    # Try to install if setup files exist, otherwise just use PYTHONPATH
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        pip install -e .
    else
        echo "  Note: Dust3R will be loaded via PYTHONPATH (no setup.py found)"
    fi

    # Add to PYTHONPATH permanently
    echo 'export PYTHONPATH="/workspace/dust3r:$PYTHONPATH"' >> ~/.bashrc
    export PYTHONPATH="/workspace/dust3r:$PYTHONPATH"

    echo "✓ Dust3R installed to /workspace/dust3r"
else
    echo "✓ Dust3R already installed"
    # Ensure PYTHONPATH is set
    if ! grep -q "/workspace/dust3r" ~/.bashrc; then
        echo 'export PYTHONPATH="/workspace/dust3r:$PYTHONPATH"' >> ~/.bashrc
    fi
    export PYTHONPATH="/workspace/dust3r:$PYTHONPATH"
fi

# 4. Set PyTorch memory allocation config
echo ""
echo "[4/5] Configuring PyTorch memory management..."
echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128' >> ~/.bashrc
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo "✓ PyTorch memory config set"

# 5. Verify installation
echo ""
echo "[5/5] Verifying installation..."

# Check Blender
if command -v blender &> /dev/null; then
    BLENDER_VER=$(blender --version | head -n 1)
    echo "✓ Blender: $BLENDER_VER"
else
    echo "✗ Blender not found in PATH"
fi

# Check Python packages
python -c "import torch; print(f'✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')" || echo "✗ PyTorch not installed"
python -c "import diffusers; print(f'✓ Diffusers: {diffusers.__version__}')" || echo "✗ Diffusers not installed"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" || echo "✗ Transformers not installed"

# Check Dust3R
if [ -d "/workspace/dust3r" ]; then
    echo "✓ Dust3R: /workspace/dust3r"
else
    echo "✗ Dust3R not found"
fi

echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "Quick Start:"
echo "  cd /workspace/Vanelia"
echo "  python vanelia_pipeline.py --input video.mp4 --model model.glb --output output.mp4"
echo ""
echo "For memory-safe processing (prevents OOM):"
echo "  python memory_safe_chunking.py --input video.mp4 --model model.glb --output output.mp4 --chunk-duration 2"
echo ""
