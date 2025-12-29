# 객체 가시성 문제 해결 가이드

## 문제: 최종 비디오에서 객체가 보이지 않음

객체가 렌더링되었지만 최종 비디오에서 보이지 않는 경우, 다음 원인들을 확인하세요:

## 1. 객체 위치 확인

### 현재 배치된 위치 확인

```bash
# 파이프라인 실행 시 출력된 위치 정보 확인
# 또는 다음 명령으로 확인:
python3 -c "
import numpy as np
from pathlib import Path
from vanelia.modules.object_placement import analyze_placement_from_camera_data

workspace = Path('chunk_workspace/chunk_workspaces/clip_001_workspace')
camera_data = workspace / 'camera_data'

if camera_data.exists():
    location, scale, plane_normal = analyze_placement_from_camera_data(
        camera_data_dir=camera_data,
        auto_placement=True
    )
    print(f'Location: {location}')
    print(f'Scale: {scale}')
else:
    print('카메라 데이터를 찾을 수 없습니다.')
"
```

### 위치가 카메라 시야 밖인 경우

객체가 카메라 시야 밖에 배치되었을 수 있습니다. 수동으로 위치를 조정하세요:

```bash
python memory_safe_chunking.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --no-auto-placement \
    --object-location 0.0 0.0 -2.0 \
    --object-scale 2.0
```

**Blender 좌표계 참고:**
- **X**: 오른쪽 (+), 왼쪽 (-)
- **Y**: 앞쪽 (+), 뒤쪽 (-)
- **Z**: 위쪽 (+), 아래쪽 (-)
- 카메라는 보통 음의 Z 방향을 향하므로, 객체는 `(0, 0, -2.0)` 정도에 배치하는 것이 일반적입니다.

## 2. 객체 크기(스케일) 조정

객체가 너무 작아서 보이지 않을 수 있습니다. 스케일을 키워보세요:

```bash
# 스케일 2배로 증가
python memory_safe_chunking.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --model-scale 2.0

# 또는 더 크게
python memory_safe_chunking.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --model-scale 3.0
```

## 3. 수동 위치 및 스케일 조정

### 추천 설정 (테스트용)

```bash
# 중앙, 적당한 거리, 큰 스케일
python memory_safe_chunking.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --no-auto-placement \
    --object-location 0.0 0.0 -2.5 \
    --object-scale 2.5
```

### 위치별 테스트

```bash
# 왼쪽
--object-location -1.0 0.0 -2.0

# 오른쪽
--object-location 1.0 0.0 -2.0

# 앞쪽
--object-location 0.0 1.0 -2.0

# 뒤쪽
--object-location 0.0 -1.0 -2.0

# 더 가까이
--object-location 0.0 0.0 -1.5

# 더 멀리
--object-location 0.0 0.0 -3.5
```

## 4. 렌더링 결과 확인

### 렌더 프레임 직접 확인

```bash
# 렌더 프레임이 있는 워크스페이스 찾기
find chunk_workspace -name "render_frames" -type d

# 첫 번째 렌더 프레임 확인
find chunk_workspace -path "*/render_frames/*.png" | head -1 | xargs ls -lh

# 이미지 뷰어로 확인 (가능한 경우)
find chunk_workspace -path "*/render_frames/*.png" | head -1 | xargs file
```

### 시각화 도구 사용

```bash
# 객체 위치 시각화
python3 visualize_object_position.py \
    chunk_workspace/chunk_workspaces/clip_001_workspace \
    --frame 0 \
    --output placement_check.png
```

## 5. 디버깅 체크리스트

- [ ] 렌더 프레임이 생성되었는가?
  ```bash
  find chunk_workspace -name "*.png" -path "*/render_frames/*" | wc -l
  ```

- [ ] 렌더 프레임에 객체가 있는가?
  - Alpha 채널 확인
  - 객체 영역이 비어있지 않은지 확인

- [ ] 객체 위치가 카메라 시야 안에 있는가?
  - 카메라 포즈와 객체 위치 비교

- [ ] 객체 스케일이 적절한가?
  - 너무 작으면 보이지 않음
  - 너무 크면 화면 밖으로 나감

- [ ] 합성 단계에서 객체가 제대로 합성되었는가?
  - `refined_frames` 확인

## 6. 빠른 테스트 방법

가장 간단한 설정으로 테스트:

```bash
python memory_safe_chunking.py \
    --input video.mp4 \
    --model product.glb \
    --output test_output.mp4 \
    --no-auto-placement \
    --object-location 0.0 0.0 -2.0 \
    --object-scale 3.0 \
    --chunk-duration 1 \
    --max-frames 30
```

이 설정으로:
- 위치: 원점 앞 2미터 (카메라 시야 중앙)
- 스케일: 3배 (크게)
- 프레임: 처음 30프레임만 (빠른 테스트)

## 7. 일반적인 문제 해결

### 문제: 객체가 화면 왼쪽/오른쪽에만 보임
**해결**: X 좌표를 0.0으로 설정하거나, 카메라 위치에 맞춰 조정

### 문제: 객체가 화면 위/아래에만 보임
**해결**: Z 좌표를 조정 (일반적으로 -2.0 ~ -3.0)

### 문제: 객체가 너무 작음
**해결**: `--object-scale` 또는 `--model-scale` 값을 증가 (2.0, 3.0, 4.0 등)

### 문제: 객체가 너무 큼
**해결**: 스케일 값을 감소 (0.5, 0.7, 1.0 등)

### 문제: 객체가 카메라 뒤에 있음
**해결**: Z 좌표를 더 음수로 설정 (예: -3.0, -4.0)

## 8. 로그 확인

파이프라인 실행 시 출력되는 다음 정보를 확인하세요:

```
======================================================================
OBJECT PLACEMENT SUMMARY
======================================================================
Location (X, Y, Z): (x.xxxxxx, y.yyyyyy, z.zzzzzz)
Scale: s.sss
======================================================================
```

이 정보를 바탕으로 위치와 스케일을 조정하세요.

## 9. 다음 단계

위 방법으로도 해결되지 않으면:
1. 원본 .glb 파일의 크기 확인
2. 카메라 포즈 데이터 확인
3. 렌더링 설정 확인 (해상도, 샘플 수 등)

