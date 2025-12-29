#!/bin/bash
# Memory-Safe Video Chunking Pipeline Runner
# Handles environment activation gracefully

set -e

cd /workspace/Vanelia

# Activate environment if it exists
if [ -f "/opt/vanelia_env/bin/activate" ]; then
    source /opt/vanelia_env/bin/activate
    echo "[Env] Activated vanelia_env"
else
    echo "[Env] No virtual environment found, using system Python"
fi

# Run memory-safe chunking pipeline
python memory_safe_chunking.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --strength 0.25 \
    --seed 12345 \
    --latent-blend 0.15

