#!/bin/bash
# Blender 4.0.2 Installation Script for Linux (Headless Mode)
# For RunPod NVIDIA A100 Ubuntu Server

set -e  # Exit on error

BLENDER_VERSION="4.0.2"
BLENDER_URL="https://download.blender.org/release/Blender4.0/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
INSTALL_DIR="/opt/blender"

echo "========================================="
echo "Blender ${BLENDER_VERSION} Installation"
echo "========================================="

# Install dependencies
echo "[1/5] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    wget \
    xz-utils \
    libxrender1 \
    libxi6 \
    libxkbcommon0 \
    libsm6 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxxf86vm1 \
    > /dev/null 2>&1

# Download Blender
echo "[2/5] Downloading Blender ${BLENDER_VERSION}..."
cd /tmp
if [ ! -f "blender-${BLENDER_VERSION}-linux-x64.tar.xz" ]; then
    wget -q --show-progress "${BLENDER_URL}"
fi

# Extract
echo "[3/5] Extracting Blender..."
tar -xf "blender-${BLENDER_VERSION}-linux-x64.tar.xz"

# Install to /opt
echo "[4/5] Installing to ${INSTALL_DIR}..."
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
mv "blender-${BLENDER_VERSION}-linux-x64"/* "${INSTALL_DIR}/"

# Create symlink
echo "[5/5] Creating symbolic link..."
ln -sf "${INSTALL_DIR}/blender" /usr/local/bin/blender

# Verify installation
echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
blender --version

# Cleanup
rm -rf /tmp/blender-*

echo ""
echo "Usage:"
echo "  blender --background --python your_script.py"
echo ""
