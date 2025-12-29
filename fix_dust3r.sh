#!/bin/bash
# Quick fix for Dust3R installation issue

echo "Fixing Dust3R installation..."

# Add Dust3R to PYTHONPATH
export PYTHONPATH="/workspace/dust3r:$PYTHONPATH"

# Make it permanent
if ! grep -q "/workspace/dust3r" ~/.bashrc 2>/dev/null; then
    echo 'export PYTHONPATH="/workspace/dust3r:$PYTHONPATH"' >> ~/.bashrc
    echo "✓ Added Dust3R to ~/.bashrc"
fi

# Install Dust3R dependencies if needed
if [ -d "/workspace/dust3r" ]; then
    cd /workspace/dust3r
    if [ -f "requirements.txt" ]; then
        echo "Installing Dust3R dependencies..."
        pip install -r requirements.txt -q
    fi
fi

echo ""
echo "✓ Dust3R is now ready!"
echo ""
echo "You can now run:"
echo "  cd /workspace/Vanelia"
echo "  python vanelia_pipeline.py --input video.mp4 --model model.glb --output output.mp4"
echo ""
