# RANSAC 평면 감지 및 객체 정렬 가이드

## 개요

제미나이와의 대화 내용을 바탕으로, **"3D 공간에 물체를 놓는 것"**의 핵심 기능을 구현했습니다:

1. **RANSAC 평면 감지**: 포인트 클라우드에서 가장 큰 평면(바닥/테이블) 자동 감지
2. **법선 벡터 정렬**: 객체의 Z축을 평면의 법선과 일치시켜 바닥에 딱 붙게 배치
3. **레이캐스팅**: 2D 화면 클릭 → 3D 좌표 변환 (구현 완료, 옵션 사용 가능)

## 핵심 원리

### 1. 위치 결정: RANSAC 평면 감지

**기존 방식 (밀도 기반)**:
- 그리드 셀의 포인트 밀도 분석
- 빈 공간 찾기

**개선된 방식 (평면 감지)**:
- RANSAC 알고리즘으로 가장 큰 평면 감지
- 평면의 중심점(Centroid)에 객체 배치
- 더 정확하고 자연스러운 배치

```python
# RANSAC 평면 감지
plane_center, plane_normal, inliers = detect_plane_ransac(points)
# → 평면 중심에 객체 배치
object_location = plane_center
```

### 2. 바닥에 딱 붙는 원리: 법선 벡터 정렬

**문제**: 객체가 비스듬한 바닥을 뚫고 들어가거나 이상하게 설 수 있음

**해결**: 객체의 Z축(Up-vector)을 평면의 법선(Normal)과 일치시킴

```python
# 평면의 법선 벡터
plane_normal = [nx, ny, nz]  # 평면의 기울기 방향

# 객체의 기본 Z축 (0, 0, 1)을 plane_normal과 일치시키는 회전 계산
rotation_quat = default_up.rotation_difference(plane_normal)
object.rotation_quaternion = rotation_quat
```

**결과**: 책상이 기울어져 있어도 객체는 그 기울기에 맞춰서 **착! 달라붙게** 됩니다.

### 3. 크기 결정: 상대적 비율

**원근감 기반 자동 계산**:
- Dust3R이 카메라 초점거리(Focal Length)를 알고 있음
- 물체가 카메라에서 멀어질수록 작아지는 비율을 정확히 계산
- 배경 영상과 100% 일치하는 원근감

```python
# 카메라 거리 기반 스케일
avg_distance = mean(camera_distances)
scale = clip(avg_distance / 3.0, 0.3, 2.0)
```

## 사용 방법

### 기본 사용 (자동 평면 감지)

```bash
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4
```

**동작**:
1. Dust3R이 포인트 클라우드 추출
2. RANSAC으로 가장 큰 평면 감지 (바닥/테이블)
3. 평면 중심에 객체 배치
4. 법선 벡터에 맞춰 객체 회전
5. 카메라 거리 기반 크기 자동 조정

### 수동 위치 지정 (평면 감지 비활성화)

```bash
python vanelia_pipeline.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4 \
    --object-location 0.0 0.0 -2.5 \
    --object-scale 1.5
```

## 구현 세부사항

### RANSAC 평면 감지 알고리즘

```python
def detect_plane_ransac(points, max_iterations=1000, distance_threshold=0.1):
    """
    1. 랜덤하게 3개 점 선택
    2. 3개 점으로 평면 방정식 계산
    3. 모든 점 중 평면에 가까운 점들(inliers) 찾기
    4. 가장 많은 inliers를 가진 평면 선택
    5. SVD로 정제된 법선 벡터 계산
    """
    # ... RANSAC 반복 ...
    return plane_center, plane_normal, inlier_indices
```

### 법선 벡터 정렬

```python
# Blender에서 객체 회전 적용
target_normal = Vector(plane_normal)
default_up = Vector((0, 0, 1))  # 객체의 기본 Z축

# 회전 쿼터니언 계산
rotation_quat = default_up.rotation_difference(target_normal)
object.rotation_quaternion = rotation_quat
```

### 레이캐스팅 (옵션)

```python
# 2D 픽셀 좌표 → 3D 좌표 변환
pixel_coords = (u, v)  # 화면 좌표
ray_dir = unproject(pixel_coords, camera_pose, intrinsics)
hit_point = raycast_to_point_cloud(ray_dir, point_cloud)
```

## 출력 예시

```
[Plane Detection] Running RANSAC on 15234 points...
[Plane Detection] ✓ Found plane with 8234 inliers
[Plane Detection]   Center: [0.12, -0.45, -1.89]
[Plane Detection]   Normal: [0.01, 0.02, 0.99]

[Object Placement] ✓ Using plane-based placement
[Object Placement]   Location: [0.12, -0.45, -1.89]
[Object Placement]   Normal: [0.01, 0.02, 0.99]
[Object Placement]   Scale: 0.63

[Blender] ✓ Aligned object Z-axis with plane normal: [0.01, 0.02, 0.99]
```

## 장점

### 기존 방식 vs 개선된 방식

| 항목 | 기존 (밀도 기반) | 개선 (평면 감지) |
|------|-----------------|-----------------|
| **정확도** | 대략적 | 정확한 평면 감지 |
| **자연스러움** | 빈 공간에 배치 | 실제 바닥/테이블에 배치 |
| **기울기 대응** | ❌ 수직 배치만 | ✅ 평면 기울기에 맞춰 회전 |
| **안정성** | 포인트 밀도에 의존 | RANSAC으로 견고함 |

## 문제 해결

### 평면을 찾을 수 없는 경우

```
[Plane Detection] WARNING: Only 45 inliers found (min: 100)
[Object Placement] Falling back to density-based placement...
```

**원인**: 포인트 클라우드가 부족하거나 평면이 명확하지 않음

**해결**: 자동으로 밀도 기반 방식으로 폴백

### 법선 벡터가 이상한 경우

**확인**: 평면의 법선 벡터가 수직(0, 0, 1)에 가까운지 확인

**조정**: `distance_threshold` 파라미터 조정 (기본값: 0.1m)

## 참고

- **RANSAC**: Random Sample Consensus, 노이즈가 많은 데이터에서 모델을 찾는 알고리즘
- **법선 벡터**: 평면에 수직인 방향 벡터
- **SVD**: Singular Value Decomposition, 더 정확한 법선 계산에 사용

## 다음 단계

향후 개선 가능한 기능:
- [ ] 사용자 클릭 좌표 입력 지원 (레이캐스팅 활용)
- [ ] 여러 평면 감지 및 선택 옵션
- [ ] 평면 크기 기반 객체 크기 조정
- [ ] 평면 경계 내 배치 (테이블 위에만 배치)

