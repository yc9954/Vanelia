#!/usr/bin/env python3
"""렌더 프레임 분석 스크립트"""
import cv2
import numpy as np
from pathlib import Path
import sys

render_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("chunk_workspace/chunk_workspaces/clip_001_workspace/render_frames/frame_000001.png")

if not render_path.exists():
    print(f"✗ 파일을 찾을 수 없습니다: {render_path}")
    sys.exit(1)

img = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)

if img is None:
    print(f"✗ 이미지를 로드할 수 없습니다: {render_path}")
    sys.exit(1)

print(f"이미지 정보:")
print(f"  - Shape: {img.shape}")
print(f"  - Dtype: {img.dtype}")

if len(img.shape) == 3 and img.shape[2] == 4:
    print(f"  - RGBA 이미지")
    alpha = img[:, :, 3]
    non_transparent = np.sum(alpha > 0)
    total_pixels = alpha.size
    coverage = (non_transparent / total_pixels) * 100
    print(f"  - 불투명 픽셀: {non_transparent:,} / {total_pixels:,} ({coverage:.2f}%)")
    print(f"  - Alpha 값 범위: {alpha.min()} ~ {alpha.max()}")
    
    if non_transparent > 0:
        object_mask = alpha > 128
        if np.any(object_mask):
            y_coords, x_coords = np.where(object_mask)
            obj_center_x = int(np.mean(x_coords))
            obj_center_y = int(np.mean(y_coords))
            obj_min_x, obj_max_x = int(np.min(x_coords)), int(np.max(x_coords))
            obj_min_y, obj_max_y = int(np.min(y_coords)), int(np.max(y_coords))
            
            print(f"\n객체 영역:")
            print(f"  - 중심: ({obj_center_x}, {obj_center_y})")
            print(f"  - 영역: ({obj_min_x}, {obj_min_y}) ~ ({obj_max_x}, {obj_max_y})")
            print(f"  - 크기: {obj_max_x - obj_min_x} x {obj_max_y - obj_min_y} 픽셀")
            print(f"  - 화면 비율: {(obj_max_x - obj_min_x)/img.shape[1]*100:.1f}% x {(obj_max_y - obj_min_y)/img.shape[0]*100:.1f}%")
        else:
            print("\n⚠ 객체가 렌더링되지 않았습니다 (모든 픽셀이 투명)")
    else:
        print("\n⚠ 완전히 투명한 이미지입니다 (객체가 카메라 시야에 없음)")
elif len(img.shape) == 3 and img.shape[2] == 3:
    print(f"  - RGB 이미지 (Alpha 채널 없음)")
else:
    print(f"  - 예상치 못한 이미지 형식")

