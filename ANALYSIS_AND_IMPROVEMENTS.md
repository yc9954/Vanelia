# Vanelia Pipeline 상세 분석 및 개선안

## 📊 현재 파이프라인 평가

### Architecture Overview
```
Video → Dust3R (Camera) → Blender (Render) → IC-Light (Relight) → Final Video
```

---

## 🔍 Module-by-Module 분석

### 1️⃣ Module A: Dust3R Camera Extraction

#### ✅ 장점
- 빠른 추론 속도 (~0.5s/frame)
- Monocular video에서 작동
- Point cloud 동시 생성

#### ❌ 문제점 및 개선안

**문제 1: 카메라 포즈 정확도 부족**
- Dust3R은 relative pose만 정확, absolute scale 부정확
- 빠른 카메라 움직임에서 drift 발생
- SfM 기반 방법 대비 정밀도 낮음

**개선안 A1: MASt3R로 업그레이드**
```python
# MASt3R (2024): Dust3R의 개선 버전
# https://github.com/naver/mast3r
from mast3r.model import AsymmetricMASt3R

# 장점:
# - Matching + Stereo 통합
# - 더 정확한 depth estimation
# - Better scale consistency
```

**개선안 A2: COLMAP + SuperPoint/SuperGlue 하이브리드**
```python
# 더 정확한 SfM pipeline
# 1. SuperPoint로 feature extraction
# 2. SuperGlue로 matching
# 3. COLMAP로 bundle adjustment
# 4. Dust3R로 dense reconstruction

# 장점: Production-level 정확도
# 단점: 느림 (~5-10s/frame)
```

**개선안 A3: DROID-SLAM (실시간 SLAM)**
```python
# https://github.com/princeton-vl/DROID-SLAM
# 장점:
# - Real-time tracking
# - Loop closure detection
# - Better scale estimation
# 단점: GPU 메모리 많이 사용
```

---

### 2️⃣ Module B: Blender Rendering

#### ✅ 장점
- 물리 기반 렌더링 (Cycles)
- GLB 재질 지원
- Shadow catcher

#### ❌ 문제점 및 개선안

**문제 1: 재질이 여전히 어둡거나 부정확**
```python
# 현재 코드 (blender_render.py:95-99)
bpy.ops.import_scene.gltf(
    filepath=glb_path,
    import_shading='FLAT',  # ❌ 문제!
    merge_vertices=False
)
```

**개선안 B1: 재질 Import 수정**
```python
# FLAT 대신 NODES 사용
bpy.ops.import_scene.gltf(
    filepath=glb_path,
    import_shading='NORMALS',  # ✅ PBR 노드 유지
    merge_vertices=False,
    bone_heuristic='TEMPERANCE'
)

# Import 후 재질 검증 및 수정
for obj in imported_objects:
    if obj.type == 'MESH':
        for slot in obj.material_slots:
            mat = slot.material
            if mat and mat.use_nodes:
                # PBR 노드 확인
                nodes = mat.node_tree.nodes
                bsdf = nodes.get('Principled BSDF')
                if bsdf:
                    # Base Color가 검은색이면 기본값으로
                    if bsdf.inputs['Base Color'].default_value[0] < 0.01:
                        bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
```

**문제 2: 환경 조명이 부자연스러움**
```python
# 현재: procedural sky만 사용 (blender_render.py:258-262)
node_sky.sky_type = 'NISHITA'  # ❌ 실제 배경과 안 맞음
```

**개선안 B2: Background Image에서 HDRI 추출**
```python
# 배경 이미지를 분석해서 조명 추정
def estimate_lighting_from_background(bg_image_path):
    """
    배경 이미지에서 조명 방향/색상 추정
    """
    import cv2
    from scipy.ndimage import gaussian_filter

    img = cv2.imread(bg_image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 밝은 영역 = 광원
    brightness = img_hsv[:, :, 2]
    smooth_bright = gaussian_filter(brightness, sigma=20)

    # 가장 밝은 영역 찾기
    max_loc = np.unravel_index(smooth_bright.argmax(), smooth_bright.shape)

    # 각도 계산 (이미지 중심 기준)
    h, w = img.shape[:2]
    dx = max_loc[1] - w/2
    dy = max_loc[0] - h/2

    sun_azimuth = np.arctan2(dy, dx)
    sun_elevation = np.radians(45)  # 기본값, 깊이 정보로 개선 가능

    # 색온도 추정
    bright_region = img[smooth_bright > np.percentile(smooth_bright, 90)]
    avg_color = np.mean(bright_region, axis=0) / 255.0

    return {
        'sun_azimuth': sun_azimuth,
        'sun_elevation': sun_elevation,
        'color': avg_color
    }

# Blender에서 적용
lighting = estimate_lighting_from_background(background_frames[0])
node_sky.sun_rotation = lighting['sun_azimuth']
node_sky.sun_elevation = lighting['sun_elevation']
```

**문제 3: Shadow Catcher가 지면과 안 맞음**
```python
# 현재: 항상 Z=0 평면 (blender_render.py:363)
renderer.create_shadow_catcher(size=20.0)  # ❌ 지면 각도 무시
```

**개선안 B3: Ground Plane에 맞춘 Shadow Catcher**
```python
def create_aligned_shadow_catcher(self, ground_plane: dict, size: float = 20.0):
    """
    검출된 지면 평면에 정렬된 Shadow Catcher 생성
    """
    # 평면 생성
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    plane = bpy.context.active_object

    if ground_plane:
        # 지면 normal에 맞춰 회전
        normal = np.array(ground_plane['normal'])

        # Z축을 normal 방향으로 회전하는 quaternion 계산
        from mathutils import Vector, Matrix
        z_axis = Vector((0, 0, 1))
        normal_vec = Vector(normal)

        rotation_matrix = z_axis.rotation_difference(normal_vec).to_matrix().to_4x4()
        plane.matrix_world = rotation_matrix

        # 지면 위치로 이동
        A, B, C, D = ground_plane['A'], ground_plane['B'], ground_plane['C'], ground_plane['D']
        # 원점에서 평면까지 거리
        if abs(C) > 1e-6:
            z_offset = -D / C
            plane.location.z = z_offset

    # Shadow catcher 설정
    plane.is_shadow_catcher = True
    # ...
```

**문제 4: GPU 최적화 부족**
```python
# 현재 (blender_render.py:59)
scene.cycles.device = 'GPU'  # ❌ 어떤 GPU인지 명시 안함
```

**개선안 B4: GPU 설정 최적화**
```python
def optimize_gpu_settings(self):
    """
    CUDA/Optix 최적화
    """
    import bpy

    # GPU 활성화
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'  # 또는 'OPTIX' for RTX

    # 모든 GPU 활성화
    for device in prefs.devices:
        device.use = True

    scene = bpy.context.scene
    scene.cycles.device = 'GPU'

    # Optix denoiser (RTX GPU 전용, 훨씬 빠름)
    scene.cycles.denoiser = 'OPTIX'

    # Tile 크기 최적화
    scene.render.tile_x = 256
    scene.render.tile_y = 256

    # 성능 설정
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.samples = 64  # 128에서 낮춰도 Optix denoiser로 깨끗
```

---

### 3️⃣ Module C: IC-Light Compositing

#### ✅ 장점
- Relighting 품질 좋음
- Fixed seed로 일관성 유지

#### ❌ 문제점 및 개선안

**문제 1: IC-Light SD1.5 기반 (구형)**
```python
# 현재 (iclight_compositor.py:36)
model_id = "lllyasviel/ic-light-sd15-fc"  # ❌ SD1.5 (2022)
```

**개선안 C1: SDXL 기반 모델로 업그레이드**
```python
# Option 1: IC-Light SDXL (더 나은 품질)
# https://huggingface.co/lllyasviel/ic-light-sdxl

# Option 2: ControlNet + SDXL
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# 장점:
# - 더 높은 해상도 (1024x1024)
# - 더 나은 디테일
# - 최신 아키텍처
```

**문제 2: Frame-by-frame 처리 (Temporal Consistency 약함)**
```python
# 현재: 각 프레임 독립적으로 처리
for idx, (render_path, bg_path) in enumerate(zip(render_frames, bg_frames)):
    output_img, current_latent = self.process_frame(...)
```

**개선안 C2: Video Diffusion Model 사용**
```python
# Stable Video Diffusion (SVD) for temporal consistency
from diffusers import StableVideoDiffusionPipeline

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16
).to("cuda")

# 장점:
# - Native video consistency
# - 자동으로 temporal coherence 유지
# - Flickering 거의 없음

# 단점:
# - 느림 (전체 비디오 한번에 처리)
# - 메모리 많이 사용
```

**개선안 C3: TokenFlow (더 나은 consistency)**
```python
# https://github.com/omerbt/TokenFlow
# - Cross-frame attention으로 일관성 유지
# - Existing diffusion model에 플러그인 가능

from tokenflow import TokenFlow

tokenflow = TokenFlow(pipe, num_frames=len(frames))
output_frames = tokenflow.generate(
    frames=composite_frames,
    prompt=prompt,
    strength=0.25,
    seed=seed
)
```

**문제 3: Alpha Blending만 사용 (Depth 무시)**
```python
# 현재 (iclight_compositor.py:68)
blended = (fg_rgb * alpha + background * (1 - alpha))  # ❌ Depth 고려 안함
```

**개선안 C4: Depth-Aware Compositing**
```python
def depth_aware_composite(self, fg_rgba, bg_rgb, fg_depth, bg_depth):
    """
    깊이 정보를 활용한 합성
    """
    # Foreground depth < Background depth인 픽셀만 합성
    fg_rgb = fg_rgba[:, :, :3]
    alpha = fg_rgba[:, :, 3:4] / 255.0

    # Depth mask (fg가 bg보다 앞에 있는 곳만)
    depth_mask = (fg_depth < bg_depth).astype(np.float32)

    # Alpha와 depth mask 결합
    final_alpha = alpha * depth_mask[:, :, np.newaxis]

    # Compositing
    blended = (fg_rgb * final_alpha +
               bg_rgb * (1 - final_alpha)).astype(np.uint8)

    return blended

# Background depth 추정
from transformers import pipeline
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

bg_depth = depth_estimator(bg_img)['depth']
```

---

## 🚀 대안 파이프라인 제안

### Option 1: NeRF/3DGS 기반 (최고 품질)

```
Video → COLMAP → NeRF/3DGS Training → Insert 3D Object → Novel View Rendering
```

**장점:**
- 완벽한 조명 일치
- 정확한 geometry
- Photo-realistic 결과

**단점:**
- 매우 느림 (NeRF training 수 시간)
- GPU 메모리 많이 필요

**구현:**
```python
# 1. Nerfstudio로 씬 재구성
ns-train nerfacto --data video.mp4

# 2. 3D 객체 삽입 (Blender)
# 3. NeRF rendering with object

# 참고: https://github.com/nerfstudio-project/nerfstudio
```

### Option 2: Depth-ControlNet (빠르고 효과적)

```
Video → Depth Estimation → Blender Render → ControlNet Refinement
```

**장점:**
- Depth로 geometry 제약
- 빠름 (~1s/frame)
- 좋은 품질

**구현:**
```python
# 1. ZoeDepth로 depth map 추출
from zoedepth.models.builder import build_model
model = build_model("zoedepth_nk")

# 2. Blender 렌더링 (depth map도 같이)
# 3. ControlNet-Depth로 refinement
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth"
)
```

### Option 3: Blender Only (단순하지만 빠름)

```
Video → COLMAP → Blender Compositing (No AI)
```

**장점:**
- 매우 빠름
- Predictable
- No flickering

**단점:**
- 조명 일치 수동 작업
- AI refinement 없음

---

## 🎯 최종 권장 사항

### 즉시 적용 가능한 개선 (High Impact, Low Effort)

**1. Blender 재질 수정**
```python
# blender_render.py:95
import_shading='NORMALS'  # FLAT → NORMALS
```

**2. GPU 최적화**
```python
scene.cycles.denoiser = 'OPTIX'
scene.cycles.samples = 64  # 128 → 64
```

**3. Background-based 조명 추정**
```python
lighting = estimate_lighting_from_background(bg_frames[0])
# 위에서 제시한 함수 추가
```

**4. IC-Light SDXL 업그레이드**
```python
model_id = "lllyasviel/ic-light-sdxl"  # SD1.5 → SDXL
```

### 중기 개선 (Better Quality)

**1. MASt3R로 카메라 추정 교체**
```bash
pip install mast3r
```

**2. ControlNet-Depth 추가**
- ZoeDepth 통합
- Depth-guided refinement

**3. Aligned Shadow Catcher**
- Ground plane에 정렬

### 장기 개선 (Production Quality)

**1. NeRF/3DGS 파이프라인 구축**
- NerfStudio 통합
- 완벽한 조명 일치

**2. Video Diffusion Model**
- SVD 또는 TokenFlow
- Perfect temporal consistency

---

## 📝 코드 품질 개선

### 에러 처리 부족

**현재:**
```python
# dust3r_camera_extraction.py:48
self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
# ❌ 모델 로드 실패 시 에러 처리 없음
```

**개선:**
```python
try:
    self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
except Exception as e:
    print(f"ERROR: Failed to load Dust3R model: {e}")
    print("Trying to download...")
    # Fallback to download
    raise
```

### 메모리 관리

**현재:**
```python
# iclight_compositor.py:218
# 모든 latent를 메모리에 유지
previous_latent = current_latent  # ❌ 메모리 누적
```

**개선:**
```python
# Latent를 CPU로 이동
previous_latent = current_latent.cpu()
torch.cuda.empty_cache()
```

### 진행률 표시 개선

**개선:**
```python
from tqdm import tqdm

for idx in tqdm(range(len(frames)), desc="IC-Light Processing"):
    # ...
```

---

## 📊 성능 벤치마크 (예상)

| Pipeline | Speed (per frame) | Quality | Memory |
|----------|------------------|---------|---------|
| **현재** | ~4-7s | ⭐⭐⭐ | 12GB |
| **개선 (즉시)** | ~2-3s | ⭐⭐⭐⭐ | 10GB |
| **ControlNet-Depth** | ~3-4s | ⭐⭐⭐⭐ | 14GB |
| **SVD** | ~10s | ⭐⭐⭐⭐⭐ | 24GB |
| **NeRF/3DGS** | ~60s+ | ⭐⭐⭐⭐⭐ | 40GB+ |

---

## 🔧 다음 단계

1. ✅ **즉시 개선 적용** (1-2시간)
   - Blender 재질 수정
   - GPU 최적화
   - 조명 추정

2. 📅 **중기 개선 계획** (1-2일)
   - MASt3R 통합
   - ControlNet-Depth 추가

3. 🎯 **장기 로드맵** (1주+)
   - NeRF 파이프라인 연구
   - Production 최적화
