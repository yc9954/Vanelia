#!/usr/bin/env python3
"""
객체 배치 위치 디버깅 도구
렌더링된 프레임과 배경 프레임을 비교하여 객체 위치를 확인합니다.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def visualize_placement(workspace_dir: str, frame_idx: int = 0):
    """
    렌더링된 프레임과 배경 프레임을 비교하여 객체 위치를 시각화합니다.
    
    Args:
        workspace_dir: 클립 워크스페이스 디렉토리
        frame_idx: 확인할 프레임 인덱스
    """
    workspace = Path(workspace_dir)
    
    # 경로 확인
    render_dir = workspace / "render_frames"
    bg_dir = workspace / "background_frames"
    refined_dir = workspace / "refined_frames"
    
    print(f"\n{'='*70}")
    print("객체 배치 위치 디버깅")
    print(f"{'='*70}")
    print(f"Workspace: {workspace}")
    print(f"Render frames: {render_dir.exists()}")
    print(f"Background frames: {bg_dir.exists()}")
    print(f"Refined frames: {refined_dir.exists()}")
    
    # 렌더 프레임 확인
    render_frames = sorted(render_dir.glob("*.png")) if render_dir.exists() else []
    bg_frames = sorted(bg_dir.glob("*.jpg")) + sorted(bg_dir.glob("*.png")) if bg_dir.exists() else []
    refined_frames = sorted(refined_dir.glob("*.png")) if refined_dir.exists() else []
    
    print(f"\n프레임 수:")
    print(f"  - 렌더 프레임: {len(render_frames)}")
    print(f"  - 배경 프레임: {len(bg_frames)}")
    print(f"  - 합성 프레임: {len(refined_frames)}")
    
    if len(render_frames) == 0:
        print("\n⚠ 렌더 프레임이 없습니다. Step 2 (Blender 렌더링)가 완료되지 않았습니다.")
        return
    
    # 첫 번째 프레임 로드
    frame_idx = min(frame_idx, len(render_frames) - 1)
    render_path = render_frames[frame_idx]
    bg_path = bg_frames[frame_idx] if frame_idx < len(bg_frames) else None
    
    print(f"\n분석할 프레임: {frame_idx}")
    print(f"  - 렌더: {render_path.name}")
    if bg_path:
        print(f"  - 배경: {bg_path.name}")
    
    # 이미지 로드
    render_img = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
    if render_img is None:
        print(f"✗ 렌더 프레임을 로드할 수 없습니다: {render_path}")
        return
    
    print(f"\n렌더 이미지 정보:")
    print(f"  - 크기: {render_img.shape}")
    print(f"  - 채널: {render_img.shape[2] if len(render_img.shape) > 2 else 1}")
    
    # Alpha 채널 확인
    if len(render_img.shape) == 3 and render_img.shape[2] == 4:
        alpha = render_img[:, :, 3]
        non_transparent = np.sum(alpha > 0)
        total_pixels = alpha.size
        coverage = (non_transparent / total_pixels) * 100
        print(f"  - Alpha 채널: 있음")
        print(f"  - 불투명 픽셀: {non_transparent:,} / {total_pixels:,} ({coverage:.1f}%)")
        
        # 객체 영역 찾기
        object_mask = alpha > 128  # 50% 이상 불투명
        if np.any(object_mask):
            y_coords, x_coords = np.where(object_mask)
            obj_center_x = int(np.mean(x_coords))
            obj_center_y = int(np.mean(y_coords))
            obj_width = int(np.max(x_coords) - np.min(x_coords))
            obj_height = int(np.max(y_coords) - np.min(y_coords))
            
            print(f"\n객체 위치 (픽셀 좌표):")
            print(f"  - 중심: ({obj_center_x}, {obj_center_y})")
            print(f"  - 크기: {obj_width} x {obj_height} 픽셀")
            print(f"  - 화면 비율: {obj_width/render_img.shape[1]*100:.1f}% x {obj_height/render_img.shape[0]*100:.1f}%")
        else:
            print("\n⚠ 객체가 렌더링되지 않았습니다 (모든 픽셀이 투명)")
    else:
        print("  - Alpha 채널: 없음 (RGBA가 아님)")
    
    # 배경과 합성 비교
    if bg_path and bg_path.exists():
        bg_img = cv2.imread(str(bg_path))
        if bg_img is not None:
            print(f"\n배경 이미지 정보:")
            print(f"  - 크기: {bg_img.shape}")
            
            # 합성 결과 확인
            if frame_idx < len(refined_frames):
                refined_path = refined_frames[frame_idx]
                refined_img = cv2.imread(str(refined_path))
                if refined_img is not None:
                    print(f"\n합성 결과 확인:")
                    print(f"  - 합성 프레임: {refined_path.name}")
                    print(f"  - 크기: {refined_img.shape}")
    
    # 카메라 데이터 확인
    camera_data_dir = workspace / "camera_data"
    if camera_data_dir.exists():
        poses_path = camera_data_dir / "camera_poses.npy"
        metadata_path = camera_data_dir / "camera_metadata.json"
        
        if poses_path.exists():
            poses = np.load(poses_path)
            print(f"\n카메라 포즈 정보:")
            print(f"  - 총 프레임: {len(poses)}")
            if len(poses) > frame_idx:
                cam_pose = poses[frame_idx]
                cam_pos = cam_pose[:3, 3]
                print(f"  - 프레임 {frame_idx} 카메라 위치: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})")
        
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"\n카메라 메타데이터:")
            print(f"  - 해상도: {metadata.get('resolution', 'N/A')}")
            print(f"  - FPS: {metadata.get('fps', 'N/A')}")
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_placement.py <workspace_dir> [frame_idx]")
        print("\nExample:")
        print("  python debug_placement.py chunk_workspace/chunk_workspaces/clip_001_workspace 0")
        sys.exit(1)
    
    workspace_dir = sys.argv[1]
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    visualize_placement(workspace_dir, frame_idx)

