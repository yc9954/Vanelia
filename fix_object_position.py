#!/usr/bin/env python3
"""
카메라 포즈를 분석하여 적절한 객체 위치를 계산합니다.
"""
import numpy as np
from pathlib import Path
import sys

def calculate_object_location(workspace_dir: str, distance: float = 2.5):
    """
    카메라 포즈를 분석하여 객체를 카메라 앞에 배치할 위치를 계산합니다.
    
    Args:
        workspace_dir: 워크스페이스 디렉토리
        distance: 카메라로부터의 거리 (미터)
    """
    workspace = Path(workspace_dir)
    poses_path = workspace / 'camera_data' / 'camera_poses.npy'
    
    if not poses_path.exists():
        print(f"✗ 카메라 포즈 파일을 찾을 수 없습니다: {poses_path}")
        return None
    
    poses = np.load(poses_path)
    
    # 모든 프레임의 카메라 위치와 방향 계산
    cam_positions = np.array([pose[:3, 3] for pose in poses])
    # 카메라가 보는 방향: Blender에서 카메라는 -Z 방향을 봄
    cam_forward_dirs = np.array([-pose[:3, 2] for pose in poses])
    
    # 평균 계산
    avg_cam_pos = cam_positions.mean(axis=0)
    avg_cam_forward = cam_forward_dirs.mean(axis=0)
    avg_cam_forward = avg_cam_forward / np.linalg.norm(avg_cam_forward)
    
    print(f"카메라 분석 결과:")
    print(f"  - 평균 카메라 위치: ({avg_cam_pos[0]:.3f}, {avg_cam_pos[1]:.3f}, {avg_cam_pos[2]:.3f})")
    print(f"  - 평균 카메라 방향: ({avg_cam_forward[0]:.3f}, {avg_cam_forward[1]:.3f}, {avg_cam_forward[2]:.3f})")
    
    # 객체를 카메라 앞에 배치
    obj_location = avg_cam_pos + avg_cam_forward * distance
    
    print(f"\n추천 객체 위치 (카메라 앞 {distance}m):")
    print(f"  ({obj_location[0]:.6f}, {obj_location[1]:.6f}, {obj_location[2]:.6f})")
    
    # 커맨드라인 인자로 출력
    print(f"\n사용 방법:")
    print(f"python memory_safe_chunking.py \\")
    print(f"    --input video.mp4 \\")
    print(f"    --model product.glb \\")
    print(f"    --output final.mp4 \\")
    print(f"    --no-auto-placement \\")
    print(f"    --object-location {obj_location[0]:.6f} {obj_location[1]:.6f} {obj_location[2]:.6f} \\")
    print(f"    --object-scale 3.0")
    
    return tuple(obj_location)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_object_position.py <workspace_dir> [distance]")
        print("\nExample:")
        print("  python fix_object_position.py chunk_workspace/chunk_workspaces/clip_001_workspace 2.5")
        sys.exit(1)
    
    workspace_dir = sys.argv[1]
    distance = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
    
    calculate_object_location(workspace_dir, distance)

