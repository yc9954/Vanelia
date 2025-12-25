#!/bin/bash
# Swap Space Setup Script
# Creates swap file if swap space is insufficient to prevent OOM and SSH disconnections

set -e

SWAP_SIZE_GB=${1:-16}  # Default 16GB swap
SWAP_FILE="/swapfile"
MIN_SWAP_GB=8

# Check if running as root or if sudo is available
if [ "$EUID" -eq 0 ]; then
    # Running as root, no need for sudo
    SUDO_CMD=""
elif command -v sudo >/dev/null 2>&1; then
    # sudo is available
    SUDO_CMD="sudo"
else
    # Not root and no sudo - check if we can write to /swapfile location
    if [ ! -w "/" ]; then
        echo "✗ ERROR: Need root privileges to create swap file"
        echo "   Please run as root or install sudo"
        exit 1
    fi
    # We can write, so no sudo needed
    SUDO_CMD=""
fi

echo "=========================================="
echo "Swap Space Setup"
echo "=========================================="

# Check current swap
CURRENT_SWAP=$(free -g | grep Swap | awk '{print $2}')
echo "Current swap: ${CURRENT_SWAP} GB"

if [ "$CURRENT_SWAP" -ge "$MIN_SWAP_GB" ]; then
    echo "✓ Sufficient swap space already exists (${CURRENT_SWAP} GB)"
    exit 0
fi

echo "⚠ Swap space is insufficient (${CURRENT_SWAP} GB < ${MIN_SWAP_GB} GB)"
echo "Creating ${SWAP_SIZE_GB} GB swap file..."

# Check if swap file already exists
if [ -f "$SWAP_FILE" ]; then
    echo "⚠ Swap file already exists: $SWAP_FILE"
    read -p "Remove existing swap file? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing swap..."
        $SUDO_CMD swapoff "$SWAP_FILE" 2>/dev/null || true
        $SUDO_CMD rm -f "$SWAP_FILE"
    else
        echo "Keeping existing swap file. Exiting."
        exit 0
    fi
fi

# Check available disk space
AVAILABLE_SPACE=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt "$SWAP_SIZE_GB" ]; then
    echo "✗ ERROR: Not enough disk space (${AVAILABLE_SPACE} GB available, need ${SWAP_SIZE_GB} GB)"
    exit 1
fi

# Create swap file
echo "Creating ${SWAP_SIZE_GB} GB swap file (this may take a few minutes)..."
$SUDO_CMD fallocate -l ${SWAP_SIZE_GB}G "$SWAP_FILE" 2>/dev/null || $SUDO_CMD dd if=/dev/zero of="$SWAP_FILE" bs=1G count=${SWAP_SIZE_GB} status=progress

# Set correct permissions
$SUDO_CMD chmod 600 "$SWAP_FILE"

# Format as swap
echo "Formatting swap file..."
$SUDO_CMD mkswap "$SWAP_FILE"

# Enable swap
echo "Enabling swap..."
if $SUDO_CMD swapon "$SWAP_FILE" 2>/dev/null; then
    echo "✓ Swap file activated successfully"
    
    # Make it permanent (add to /etc/fstab)
    if [ -w /etc/fstab ] || [ -n "$SUDO_CMD" ]; then
        if ! grep -q "$SWAP_FILE" /etc/fstab 2>/dev/null; then
            echo "Adding to /etc/fstab for persistence..."
            echo "$SWAP_FILE none swap sw 0 0" | $SUDO_CMD tee -a /etc/fstab >/dev/null
        fi
    else
        echo "⚠ Warning: Cannot write to /etc/fstab (swap will not persist after reboot)"
    fi
else
    echo "⚠ WARNING: Failed to activate swap file"
    echo "⚠ This is common in Docker containers or restricted environments"
    echo "⚠ The swap file was created but not activated"
    echo ""
    echo "Possible solutions:"
    echo "  1. If running in Docker: Configure swap at the host level"
    echo "  2. If running in a VM: Ensure swap is enabled in VM settings"
    echo "  3. The memory management will still work, but without swap"
    echo ""
    echo "For Docker: Add '--memory-swap' flag when running container:"
    echo "  docker run --memory-swap=32g ..."
    echo ""
    # Don't exit with error, just warn
fi

# Verify
NEW_SWAP=$(free -g | grep Swap | awk '{print $2}')
echo ""
echo "=========================================="
if [ "$NEW_SWAP" -ge "$MIN_SWAP_GB" ]; then
    echo "✓ Swap setup complete!"
else
    echo "⚠ Swap file created but not active"
fi
echo "=========================================="
echo "Current swap space: ${NEW_SWAP} GB"
if [ -f "$SWAP_FILE" ]; then
    SWAP_FILE_SIZE=$(du -h "$SWAP_FILE" | cut -f1)
    echo "Swap file: $SWAP_FILE (${SWAP_FILE_SIZE})"
fi
echo ""
echo "To verify: free -h"
if [ -n "$SUDO_CMD" ]; then
    echo "To remove swap later: $SUDO_CMD swapoff $SWAP_FILE 2>/dev/null; $SUDO_CMD rm $SWAP_FILE"
else
    echo "To remove swap later: swapoff $SWAP_FILE 2>/dev/null; rm $SWAP_FILE"
fi

