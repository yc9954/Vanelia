#!/usr/bin/env python3
"""
System Memory Management Utilities
Provides functions to monitor and manage system RAM to prevent OOM and SSH disconnections.
"""

import os
import sys
import subprocess
import gc
import psutil
from typing import Tuple, Optional
import shutil
from pathlib import Path


def get_system_memory_info() -> dict:
    """
    Get current system memory usage information.
    
    Returns:
        Dictionary with memory stats (total, available, used, percent, swap)
    """
    try:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent,
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_percent': swap.percent
        }
    except Exception as e:
        print(f"[Memory] Warning: Could not get memory info: {e}")
        return {}


def check_memory_threshold(threshold_percent: float = 85.0) -> bool:
    """
    Check if system memory usage exceeds threshold.
    
    Args:
        threshold_percent: Memory usage threshold (default: 85%)
        
    Returns:
        True if memory usage is above threshold
    """
    try:
        mem = psutil.virtual_memory()
        return mem.percent >= threshold_percent
    except Exception:
        return False


def clear_system_caches():
    """
    Clear system caches to free up memory.
    This is safe and only clears filesystem caches, not application memory.
    """
    try:
        # Clear filesystem caches (requires root, but safe to try)
        # This only clears page cache, dentries, and inodes - safe for applications
        result = subprocess.run(
            ['sync'],  # Sync filesystem first
            capture_output=True,
            text=True
        )
        
        # Try to drop caches (requires root, may fail)
        # Only drop page cache (level 1) - safest option
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('1')  # Clear page cache only
            print("[Memory] ✓ Cleared filesystem page cache")
            return True
        except PermissionError:
            print("[Memory] ⚠ Cannot clear system caches (requires root)")
            return False
        except FileNotFoundError:
            print("[Memory] ⚠ /proc/sys/vm/drop_caches not available")
            return False
    except Exception as e:
        print(f"[Memory] ⚠ Could not clear system caches: {e}")
        return False


def clear_python_memory():
    """
    Aggressively clear Python memory by forcing garbage collection.
    """
    collected = 0
    for _ in range(3):  # Run multiple times to catch cyclic references
        collected += gc.collect()
    print(f"[Memory] ✓ Python garbage collection: freed {collected} objects")
    return collected


def cleanup_temp_files(temp_dirs: list, min_free_gb: float = 5.0):
    """
    Clean up temporary files if system memory is low.
    
    Args:
        temp_dirs: List of temporary directory paths to clean
        min_free_gb: Minimum free memory in GB before cleaning (default: 5GB)
    """
    mem_info = get_system_memory_info()
    if not mem_info:
        return False
    
    available_gb = mem_info.get('available_gb', 0)
    
    if available_gb < min_free_gb:
        print(f"[Memory] Low memory ({available_gb:.2f} GB available), cleaning temp files...")
        
        cleaned = 0
        for temp_dir in temp_dirs:
            temp_path = Path(temp_dir)
            if temp_path.exists():
                try:
                    # Calculate size before deletion
                    size_before = sum(f.stat().st_size for f in temp_path.rglob('*') if f.is_file())
                    size_gb = size_before / (1024**3)
                    
                    # Remove directory
                    shutil.rmtree(temp_path)
                    cleaned += size_gb
                    print(f"  - Removed {temp_path}: {size_gb:.2f} GB")
                except Exception as e:
                    print(f"  - Failed to remove {temp_path}: {e}")
        
        if cleaned > 0:
            print(f"[Memory] ✓ Cleaned {cleaned:.2f} GB of temporary files")
            return True
    
    return False


def check_swap_space(min_swap_gb: float = 8.0) -> Tuple[bool, Optional[str]]:
    """
    Check if swap space exists and is sufficient.
    
    Args:
        min_swap_gb: Minimum required swap space in GB
        
    Returns:
        Tuple of (has_sufficient_swap, message)
    """
    try:
        swap = psutil.swap_memory()
        swap_total_gb = swap.total / (1024**3)
        
        if swap_total_gb < min_swap_gb:
            return False, f"Swap space is {swap_total_gb:.2f} GB (recommended: {min_swap_gb} GB)"
        else:
            return True, f"Swap space: {swap_total_gb:.2f} GB"
    except Exception:
        return False, "Could not check swap space"


def monitor_memory_during_processing(callback, *args, **kwargs):
    """
    Monitor memory during a processing function and clear caches if needed.
    
    Args:
        callback: Function to execute
        *args, **kwargs: Arguments to pass to callback
        
    Returns:
        Result of callback function
    """
    # Check memory before
    mem_before = get_system_memory_info()
    if mem_before:
        print(f"\n[Memory] Before: {mem_before['used_gb']:.2f} GB / {mem_before['total_gb']:.2f} GB "
              f"({mem_before['percent']:.1f}%)")
    
    # Clear Python memory before processing
    clear_python_memory()
    
    # Execute callback
    try:
        result = callback(*args, **kwargs)
    finally:
        # Clear memory after processing
        clear_python_memory()
        
        # Check memory after
        mem_after = get_system_memory_info()
        if mem_after:
            print(f"[Memory] After: {mem_after['used_gb']:.2f} GB / {mem_after['total_gb']:.2f} GB "
                  f"({mem_after['percent']:.1f}%)")
        
        # If memory is still high, try clearing system caches
        if mem_after and mem_after['percent'] > 80:
            print("[Memory] High memory usage detected, clearing system caches...")
            clear_system_caches()
    
    return result


def get_process_memory_usage() -> dict:
    """
    Get memory usage of current process.
    
    Returns:
        Dictionary with process memory stats
    """
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            'rss_gb': mem_info.rss / (1024**3),  # Resident Set Size
            'vms_gb': mem_info.vms / (1024**3),  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except Exception:
        return {}


def print_memory_status():
    """Print current memory status for debugging."""
    mem_info = get_system_memory_info()
    proc_info = get_process_memory_usage()
    
    print("\n" + "="*70)
    print("MEMORY STATUS")
    print("="*70)
    
    if mem_info:
        print(f"System RAM: {mem_info['used_gb']:.2f} GB / {mem_info['total_gb']:.2f} GB "
              f"({mem_info['percent']:.1f}%)")
        print(f"Available: {mem_info['available_gb']:.2f} GB")
        print(f"Swap: {mem_info['swap_used_gb']:.2f} GB / {mem_info['swap_total_gb']:.2f} GB "
              f"({mem_info['swap_percent']:.1f}%)")
    
    if proc_info:
        print(f"Process RSS: {proc_info['rss_gb']:.2f} GB ({proc_info['percent']:.1f}%)")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    # Test memory utilities
    print_memory_status()
    
    has_swap, swap_msg = check_swap_space()
    print(f"Swap check: {swap_msg}")
    
    if check_memory_threshold(85.0):
        print("⚠ Memory usage is above 85%")
    else:
        print("✓ Memory usage is normal")

