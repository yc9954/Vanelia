#!/bin/bash
# Memory-Safe Pipeline Runner
# Ensures sufficient swap space and monitors memory during execution

set -e

cd /workspace/Vanelia

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "Memory-Safe Pipeline Runner"
echo "==========================================${NC}"

# Check if swap space exists
SWAP_SIZE=$(free -g | grep Swap | awk '{print $2}')
if [ "$SWAP_SIZE" -lt 8 ]; then
    echo -e "${YELLOW}⚠ WARNING: Swap space is only ${SWAP_SIZE} GB (recommended: 8+ GB)${NC}"
    echo -e "${YELLOW}⚠ This may cause SSH disconnections if system memory runs out${NC}"
    echo ""
    
    # Check if we're in Docker (common case)
    if [ -f /.dockerenv ] || [ -n "$container" ]; then
        echo -e "${YELLOW}⚠ Running in Docker container - swap must be configured at host level${NC}"
        echo -e "${YELLOW}⚠ Memory management will still work, but swap is limited${NC}"
        echo ""
    else
        read -p "Create swap space now? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Setting up swap space..."
            bash setup_swap.sh 16
        else
            echo -e "${YELLOW}Continuing without additional swap space...${NC}"
        fi
        echo ""
    fi
fi

# Activate environment if it exists
if [ -f "/opt/vanelia_env/bin/activate" ]; then
    source /opt/vanelia_env/bin/activate
    echo "[Env] Activated vanelia_env"
else
    echo "[Env] No virtual environment found, using system Python"
fi

# Check if psutil is available (for memory monitoring)
python3 -c "import psutil" 2>/dev/null || {
    echo -e "${YELLOW}⚠ psutil not found. Installing for memory monitoring...${NC}"
    pip install psutil --quiet
}

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}⚠ ffmpeg not found. Installing...${NC}"
    apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
    echo -e "${GREEN}✓ ffmpeg installed${NC}"
fi

# Check Python dependencies
echo -e "${GREEN}Checking Python dependencies...${NC}"
MISSING_DEPS=()
python3 -c "import cv2" 2>/dev/null || MISSING_DEPS+=("opencv-python")
python3 -c "import torch" 2>/dev/null || MISSING_DEPS+=("torch")
python3 -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy")
python3 -c "import PIL" 2>/dev/null || MISSING_DEPS+=("pillow")
python3 -c "import psutil" 2>/dev/null || MISSING_DEPS+=("psutil")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠ Missing Python dependencies: ${MISSING_DEPS[*]}${NC}"
    echo -e "${YELLOW}Installing from requirements.txt...${NC}"
    pip install -q -r requirements.txt 2>/dev/null || {
        echo -e "${YELLOW}Installing missing packages individually...${NC}"
        for dep in "${MISSING_DEPS[@]}"; do
            pip install -q "$dep" 2>/dev/null || true
        done
    }
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ All core dependencies available${NC}"
fi
echo ""

# Print initial memory status
echo ""
echo "Initial Memory Status:"
free -h
echo ""

# Set memory-friendly environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4  # Limit CPU threads to reduce memory pressure

# Run memory-safe chunking pipeline
echo -e "${GREEN}Starting memory-safe pipeline...${NC}"
echo ""

python memory_safe_chunking.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --strength 0.25 \
    --seed 12345 \
    --latent-blend 0.15 \
    "$@"

EXIT_CODE=$?

# Print final memory status
echo ""
echo "Final Memory Status:"
free -h
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Pipeline completed successfully${NC}"
else
    echo -e "${RED}✗ Pipeline failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE

