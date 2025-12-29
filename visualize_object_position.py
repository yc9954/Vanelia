#!/usr/bin/env python3
"""
객체 위치 시각화 도구
렌더링된 프레임과 배경 프레임을 나란히 비교하여 객체가 어디에 삽입되었는지 확인합니다.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import argparse

def create_comparison_image(render_path: Path, bg_path: Path = None, output_path: Path = None):
    """
    렌더 프레임과 배경 프레임을 나란히 비교 이미지 생성.
    
    Args:
        render_path: 렌더링된 프레임 경로
        bg_path: 배경 프레임 경로 (선택)
        output_path: 출력 이미지 경로 (선택)
    """
    # 렌더 이미지 로드
    render_img = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
    if render_img is None:
        print(f"✗ 렌더 프레임을 로드할 수 없습니다: {render_path}")
        return None
    
    print(f"렌더 이미지: {render_img.shape}")
    
    # Alpha 채널 처리
    if len(render_img.shape) == 3 and render_img.shape[2] == 4:
        # RGBA -> RGB 변환 (투명 배경)
        alpha = render_img[:, :, 3] / 255.0
        render_rgb = render_img[:, :, :3]
        
        # 객체 마스크 생성
        object_mask = alpha > 0.1
        
        # 객체 영역 계산
        if np.any(object_mask):
            y_coords, x_coords = np.where(object_mask)
            obj_center_x = int(np.mean(x_coords))
            obj_center_y = int(np.mean(y_coords))
            obj_min_x, obj_max_x = int(np.min(x_coords)), int(np.max(x_coords))
            obj_min_y, obj_max_y = int(np.min(y_coords)), int(np.max(y_coords))
            
            print(f"\n객체 위치:")
            print(f"  - 중심: ({obj_center_x}, {obj_center_y})")
            print(f"  - 영역: ({obj_min_x}, {obj_min_y}) ~ ({obj_max_x}, {obj_max_y})")
            print(f"  - 크기: {obj_max_x - obj_min_x} x {obj_max_y - obj_min_y} 픽셀")
            print(f"  - 화면 비율: {(obj_max_x - obj_min_x)/render_img.shape[1]*100:.1f}% x {(obj_max_y - obj_min_y)/render_img.shape[0]*100:.1f}%")
            
            # 객체 영역에 빨간 박스 그리기
            render_with_box = render_rgb.copy()
            cv2.rectangle(render_with_box, (obj_min_x, obj_min_y), (obj_max_x, obj_max_y), (0, 0, 255), 3)
            cv2.circle(render_with_box, (obj_center_x, obj_center_y), 5, (0, 255, 0), -1)
        else:
            print("\n⚠ 객체가 렌더링되지 않았습니다 (모든 픽셀이 투명)")
            render_with_box = render_rgb.copy()
    else:
        render_with_box = render_img.copy() if len(render_img.shape) == 3 else cv2.cvtColor(render_img, cv2.COLOR_GRAY2BGR)
    
    # 배경 이미지와 합성
    if bg_path and bg_path.exists():
        bg_img = cv2.imread(str(bg_path))
        if bg_img is not None:
            # 크기 맞추기
            if bg_img.shape[:2] != render_rgb.shape[:2]:
                bg_img = cv2.resize(bg_img, (render_rgb.shape[1], render_rgb.shape[0]))
            
            # Alpha 합성
            if len(render_img.shape) == 3 and render_img.shape[2] == 4:
                alpha = render_img[:, :, 3:4] / 255.0
                composite = (render_rgb * alpha + bg_img * (1 - alpha)).astype(np.uint8)
            else:
                composite = render_rgb
            
            # 객체 영역에 박스 그리기 (합성 이미지에도)
            if np.any(object_mask):
                cv2.rectangle(composite, (obj_min_x, obj_min_y), (obj_max_x, obj_max_y), (0, 0, 255), 3)
                cv2.circle(composite, (obj_center_x, obj_center_y), 5, (0, 255, 0), -1)
            
            # 나란히 비교 이미지 생성
            comparison = np.hstack([bg_img, render_with_box, composite])
            
            # 레이블 추가
            h, w = comparison.shape[:2]
            cv2.putText(comparison, "Background", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Rendered Object", (w//3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Composite", (2*w//3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            comparison = render_with_box
    else:
        comparison = render_with_box
    
    # 저장
    if output_path:
        cv2.imwrite(str(output_path), comparison)
        print(f"\n✓ 비교 이미지 저장: {output_path}")
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description="객체 위치 시각화")
    parser.add_argument("workspace", help="클립 워크스페이스 디렉토리")
    parser.add_argument("--frame", type=int, default=0, help="프레임 인덱스 (기본: 0)")
    parser.add_argument("--output", help="출력 이미지 경로")
    
    args = parser.parse_args()
    
    workspace = Path(args.workspace)
    render_dir = workspace / "render_frames"
    bg_dir = workspace / "background_frames"
    
    if not render_dir.exists():
        print(f"✗ 렌더 프레임 디렉토리가 없습니다: {render_dir}")
        sys.exit(1)
    
    render_frames = sorted(render_dir.glob("*.png"))
    if len(render_frames) == 0:
        print(f"✗ 렌더 프레임이 없습니다: {render_dir}")
        sys.exit(1)
    
    frame_idx = min(args.frame, len(render_frames) - 1)
    render_path = render_frames[frame_idx]
    
    bg_path = None
    if bg_dir.exists():
        bg_frames = sorted(bg_dir.glob("*.jpg")) + sorted(bg_dir.glob("*.png"))
        if frame_idx < len(bg_frames):
            bg_path = bg_frames[frame_idx]
    
    output_path = Path(args.output) if args.output else workspace / f"placement_visualization_frame_{frame_idx}.png"
    
    print(f"프레임 {frame_idx} 분석 중...")
    print(f"렌더: {render_path.name}")
    if bg_path:
        print(f"배경: {bg_path.name}")
    print()
    
    comparison = create_comparison_image(render_path, bg_path, output_path)
    
    if comparison is not None:
        print(f"\n✓ 시각화 완료!")
        print(f"이미지를 확인하세요: {output_path}")

if __name__ == '__main__':
    main()

