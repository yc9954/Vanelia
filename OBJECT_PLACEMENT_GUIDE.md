# 객체 위치 및 크기 자동 결정 가이드

## 문제점

기존에는 객체가 항상 `(0, 0, 0)` 위치에 고정되어 있었습니다. 이로 인해:
- 실제 비디오에서 그 위치에 다른 객체가 있으면 충돌/겹침 발생
- 객체의 적절한 크기를 알 수 없음

## 해결 방법

### 1. 자동 배치 (Auto-Placement) - 기본값

Dust3R의 포인트 클라우드를 분석하여:
- **빈 공간 찾기**: 포인트 밀도가 낮은 영역 탐색
- **카메라 거리 고려**: 너무 가깝거나 먼 곳 제외
- **적절한 크기 계산**: 카메라 거리에 기반한 스케일 자동 조정

```bash
# 자동 배치 사용 (기본값)
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4
```

### 2. 수동 배치

원하는 위치와 크기를 직접 지정:

```bash
# 수동 위치 지정 (Blender 좌표계)
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --no-auto-placement \
    --object-location 1.5 -0.5 -2.0 \
    --object-scale 1.2
```

### 3. 자동 배치 비활성화

자동 배치를 끄고 기본값 사용:

```bash
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --no-auto-placement
```

## Blender 좌표계

Blender는 다음 좌표계를 사용합니다:
- **X**: 오른쪽 (+)
- **Y**: 앞쪽 (+)
- **Z**: 위쪽 (+)

카메라는 보통 음의 Z 방향(아래쪽)을 향하므로, 객체는 보통 `(x, y, -z)` 형태로 배치됩니다.

## 동작 원리

### 1. 포인트 클라우드 분석

```
[Dust3R] → 포인트 클라우드 추출 → point_cloud.npy 저장
```

### 2. 빈 공간 탐색

```python
# 그리드 기반 밀도 분석
- 50cm 그리드 셀 생성
- 각 셀의 포인트 밀도 계산
- 낮은 밀도 영역 = 빈 공간
```

### 3. 위치 결정

```python
# 우선순위:
1. 포인트 밀도가 낮은 영역
2. 카메라로부터 적절한 거리 (0.5m ~ 5m)
3. 카메라 앞쪽 (음의 Z 방향) 선호
```

### 4. 크기 결정

```python
# 카메라 거리 기반
scale = average_camera_distance / 3.0
# 범위: 0.3 ~ 2.0
```

## CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--auto-placement` | 자동 배치 활성화 | `True` |
| `--no-auto-placement` | 자동 배치 비활성화 | - |
| `--object-location X Y Z` | 수동 위치 지정 (Blender 좌표) | `None` |
| `--object-scale SCALE` | 수동 크기 지정 | `None` (자동) |
| `--model-scale SCALE` | 레거시: 모델 스케일 (자동 배치가 우선) | `None` |

## 예제

### 예제 1: 기본 자동 배치

```bash
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4
```

**결과**: 포인트 클라우드를 분석하여 최적의 위치와 크기 자동 결정

### 예제 2: 특정 위치에 배치

```bash
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --object-location 0.0 0.0 -2.5 \
    --object-scale 1.5
```

**결과**: 객체를 `(0, 0, -2.5)` 위치에 1.5배 크기로 배치

### 예제 3: 자동 배치 + 크기만 수동 지정

```bash
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --object-scale 2.0
```

**결과**: 위치는 자동 결정, 크기는 2.0배로 고정

## 메모리 안전 청킹과 함께 사용

`memory_safe_chunking.py`도 자동 배치를 지원합니다:

```bash
python memory_safe_chunking.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --chunk-duration 2.0
```

각 클립마다 독립적으로 자동 배치가 수행됩니다.

## 문제 해결

### 포인트 클라우드가 없는 경우

```
[Object Placement] WARNING: Point cloud not found
[Object Placement] Using camera-based placement only
```

**해결**: Dust3R이 포인트 클라우드를 생성하지 못한 경우, 카메라 위치 기반으로 폴백합니다.

### 빈 공간을 찾을 수 없는 경우

```
[Object Placement] WARNING: No empty space found
[Object Placement] Using camera-relative placement
```

**해결**: 평균 카메라 위치 앞쪽 2m 지점에 배치합니다.

## 기술 세부사항

### 파일 구조

```
vanelia/modules/
├── object_placement.py      # 배치 분석기
├── dust3r_camera_extraction.py  # 포인트 클라우드 저장
└── blender_render.py        # 위치/크기 적용
```

### 주요 함수

- `ObjectPlacementAnalyzer.analyze_empty_space()`: 빈 공간 탐색
- `analyze_placement_from_camera_data()`: 카메라 데이터에서 배치 결정
- `BlenderRenderer.load_glb_model()`: 위치/크기 적용

## 참고

- 포인트 클라우드는 `camera_data/point_cloud.npy`에 저장됩니다
- 배치 정보는 콘솔에 출력됩니다
- 각 클립마다 독립적으로 배치가 결정됩니다 (청킹 모드)

