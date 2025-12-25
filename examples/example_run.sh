#!/bin/bash
# Example: Run Vanelia Pipeline
# This script demonstrates how to use the complete pipeline

set -e

echo "Vanelia Pipeline Example"
echo "========================"
echo ""

# Configuration
INPUT_VIDEO="input_video.mp4"
GLB_MODEL="brand_asset.glb"
OUTPUT_VIDEO="output/final_video.mp4"

# Check if input files exist
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "ERROR: Input video not found: $INPUT_VIDEO"
    echo "Please place your video file as: $INPUT_VIDEO"
    exit 1
fi

if [ ! -f "$GLB_MODEL" ]; then
    echo "ERROR: 3D model not found: $GLB_MODEL"
    echo "Please place your .glb model as: $GLB_MODEL"
    exit 1
fi

# Activate environment
if [ -f "/opt/vanelia_env/bin/activate" ]; then
    source /opt/vanelia_env/bin/activate
fi

# Run pipeline
echo "Starting Vanelia pipeline..."
echo ""

python ../vanelia_pipeline.py \
    --input "$INPUT_VIDEO" \
    --model "$GLB_MODEL" \
    --output "$OUTPUT_VIDEO" \
    --frame-interval 1 \
    --max-frames 100 \
    --strength 0.25 \
    --seed 12345 \
    --latent-blend 0.15 \
    --fps 30 \
    --crf 18

echo ""
echo "✓ Pipeline complete!"
echo "Output: $OUTPUT_VIDEO"
