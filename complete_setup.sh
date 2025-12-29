#!/bin/bash
# Quick fix script - run this now to complete the setup

echo "Completing Dust3R setup..."

# 1. Set PYTHONPATH
export PYTHONPATH="/workspace/dust3r:$PYTHONPATH"
if ! grep -q "/workspace/dust3r" ~/.bashrc 2>/dev/null; then
    echo 'export PYTHONPATH="/workspace/dust3r:$PYTHONPATH"' >> ~/.bashrc
fi

# 2. Set PyTorch memory config (if not already set)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if ! grep -q "PYTORCH_CUDA_ALLOC_CONF" ~/.bashrc 2>/dev/null; then
    echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128' >> ~/.bashrc
fi

# 3. Add Blender to PATH (if not already)
export PATH="/workspace/blender:$PATH"
if ! grep -q "/workspace/blender" ~/.bashrc 2>/dev/null; then
    echo 'export PATH="/workspace/blender:$PATH"' >> ~/.bashrc
fi

echo ""
echo "✓ Setup completed!"
echo ""
echo "Verifying installation..."

# Verify
blender --version 2>&1 | head -n 1
python -c "import torch; print(f'✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import diffusers; print(f'✓ Diffusers: {diffusers.__version__}')"
ls -d /workspace/dust3r && echo "✓ Dust3R: /workspace/dust3r"

echo ""
echo "=================================================="
echo "  Ready to use!"
echo "=================================================="
echo ""
echo "Test run:"
echo "  cd /workspace/Vanelia"
echo "  python vanelia_pipeline.py --help"
echo ""
