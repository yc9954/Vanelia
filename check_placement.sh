#!/bin/bash
# 객체 배치 위치 확인 스크립트

cd /workspace/Vanelia

echo "=========================================="
echo "객체 배치 위치 확인"
echo "=========================================="

# 워크스페이스 찾기
WORKSPACE=$(find chunk_workspace -type d -name "*workspace" | head -1)

if [ -z "$WORKSPACE" ]; then
    echo "⚠ 워크스페이스를 찾을 수 없습니다."
    echo "파이프라인을 먼저 실행하세요."
    exit 1
fi

echo "Workspace: $WORKSPACE"
echo ""

# 렌더 프레임 확인
RENDER_DIR="$WORKSPACE/render_frames"
if [ -d "$RENDER_DIR" ]; then
    RENDER_COUNT=$(find "$RENDER_DIR" -name "*.png" | wc -l)
    echo "✓ 렌더 프레임: $RENDER_COUNT 개"
    
    if [ "$RENDER_COUNT" -gt 0 ]; then
        FIRST_FRAME=$(find "$RENDER_DIR" -name "*.png" | sort | head -1)
        echo "  첫 번째 프레임: $(basename $FIRST_FRAME)"
        
        # 이미지 정보 확인
        if command -v identify &> /dev/null; then
            echo "  이미지 정보:"
            identify "$FIRST_FRAME" 2>/dev/null | head -1
        fi
    fi
else
    echo "✗ 렌더 프레임 디렉토리가 없습니다."
fi

# 배경 프레임 확인
BG_DIR="$WORKSPACE/background_frames"
if [ -d "$BG_DIR" ]; then
    BG_COUNT=$(find "$BG_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
    echo "✓ 배경 프레임: $BG_COUNT 개"
else
    echo "✗ 배경 프레임 디렉토리가 없습니다."
fi

# 카메라 데이터 확인
CAMERA_DIR="$WORKSPACE/camera_data"
if [ -d "$CAMERA_DIR" ]; then
    echo "✓ 카메라 데이터 디렉토리 존재"
    
    if [ -f "$CAMERA_DIR/camera_poses.npy" ]; then
        echo "  - camera_poses.npy 존재"
        python3 -c "
import numpy as np
poses = np.load('$CAMERA_DIR/camera_poses.npy')
print(f'  - 총 프레임: {len(poses)}')
if len(poses) > 0:
    first_pose = poses[0]
    cam_pos = first_pose[:3, 3]
    print(f'  - 첫 프레임 카메라 위치: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})')
" 2>/dev/null
    fi
    
    if [ -f "$CAMERA_DIR/point_cloud.npy" ]; then
        echo "  - point_cloud.npy 존재"
    fi
fi

# 디버깅 스크립트 실행
if [ -f "debug_placement.py" ]; then
    echo ""
    echo "상세 분석 실행 중..."
    python3 debug_placement.py "$WORKSPACE" 0
fi

echo ""
echo "=========================================="

