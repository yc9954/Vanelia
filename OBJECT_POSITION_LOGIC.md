# 객체 위치와 크기 결정 로직

## 개요

Vanelia 파이프라인에서 3D 객체의 위치와 크기는 다음과 같이 결정됩니다:

## 1. 객체 크기 (Scale)

### 설정 위치
- **파일**: `vanelia/modules/blender_render.py:97`
- **코드**: `imported_obj.scale = (scale, scale, scale)`
- **기본값**: `1.0` (원본 크기)

### 제어 방법
```bash
# 커맨드라인에서
python memory_safe_chunking.py \
    --model-scale 2.0 \  # 2배 크기
    ...

# 또는 vanelia_pipeline.py에서
pipeline.run_full_pipeline(
    model_scale=2.0,  # 2배 크기
    ...
)
```

### 특징
- **균일 스케일**: X, Y, Z 모두 동일한 비율로 확대/축소
- **상대적 크기**: 원본 .glb 파일의 크기를 기준으로 배율 적용

---

## 2. 객체 위치 (Location)

### 현재 상태
- **파일**: `vanelia/modules/blender_render.py:98`
- **코드**: `imported_obj.location = location` (기본값: `(0, 0, 0)`)
- **현재**: 항상 **원점 (0, 0, 0)**에 고정

### 좌표계
- **Blender 좌표계**: Right (+X), Up (+Z), Back (-Y)
- **원점**: 월드 좌표계의 중심
- **Shadow Catcher**: (0, 0, 0) 위치에 크기 20.0의 평면

### 위치 결정 원리
```
객체 위치 = (0, 0, 0)  ← 고정
카메라 위치 = Dust3R이 추출한 실제 카메라 포즈
```

**핵심**: 객체는 움직이지 않고, **카메라가 움직이면서** 객체를 촬영합니다.

---

## 3. 카메라 움직임 (Camera Animation)

### 추출 과정
1. **Dust3R (Module A)**: 비디오에서 카메라 포즈 추출
   - 각 프레임마다 카메라의 위치와 회전 추출
   - OpenCV 좌표계 → Blender 좌표계로 변환

2. **Blender (Module B)**: 카메라 포즈 적용
   ```python
   # blender_render.py:223
   camera_obj.matrix_world = cam_matrix  # 프레임별 카메라 포즈
   ```

### 동작 방식
```
Frame 1: 카메라가 위치 A에서 객체를 촬영
Frame 2: 카메라가 위치 B로 이동하여 객체를 촬영
Frame 3: 카메라가 위치 C로 이동하여 객체를 촬영
...
```

**비유**: 
- 객체 = 무대 위 배우 (고정)
- 카메라 = 촬영 감독 (움직임)
- 결과 = 배우를 다양한 각도에서 촬영한 영상

---

## 4. Shadow Catcher (그림자 받이)

### 설정
- **위치**: (0, 0, 0) - 객체와 동일한 위치
- **크기**: 20.0 (Blender 단위)
- **특성**: 투명하지만 그림자를 받음

### 역할
- 객체가 바닥에 그림자를 드리움
- 그림자만 렌더링되어 배경과 자연스럽게 합성됨

---

## 5. 실제 렌더링 과정

```
1. 객체 로드
   - 위치: (0, 0, 0)
   - 크기: model_scale 배율
   
2. Shadow Catcher 생성
   - 위치: (0, 0, 0)
   - 크기: 20.0
   
3. 카메라 설정
   - 초점 거리: Dust3R intrinsics에서 계산
   - 프레임별 위치/회전: Dust3R poses에서 적용
   
4. 렌더링
   - 각 프레임마다 카메라 위치가 변경됨
   - 객체는 항상 같은 위치에 있음
   - 결과: 객체를 다양한 각도에서 본 RGBA 이미지
```

---

## 6. 위치/크기 조정이 필요한 경우

### 현재 제한사항
- 객체 위치는 (0, 0, 0)으로 고정
- 크기만 조정 가능

### 개선 방안 (필요시)
객체 위치를 조정하려면:

```python
# blender_render.py 수정 예시
model_obj = renderer.load_glb_model(
    args.glb, 
    scale=args.scale,
    location=(x_offset, y_offset, z_offset)  # 위치 조정
)
```

또는 커맨드라인 인자 추가:
```bash
--model-position 1.0 0.5 -0.3  # X, Y, Z 오프셋
```

---

## 요약

| 항목 | 현재 값 | 조정 가능 여부 |
|------|---------|--------------|
| **크기** | `model_scale` (기본 1.0) | ✅ 가능 (`--model-scale`) |
| **위치** | (0, 0, 0) 고정 | ❌ 불가능 (코드 수정 필요) |
| **카메라** | Dust3R 포즈 자동 적용 | ✅ 자동 (비디오에서 추출) |
| **Shadow Catcher** | (0, 0, 0), 크기 20.0 | ❌ 고정 |

**핵심**: 객체는 고정, 카메라가 움직이는 방식으로 비디오의 카메라 움직임을 재현합니다.

