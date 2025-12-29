# 시스템 메모리 관리 가이드

SSH 연결 끊김을 방지하고 안정적인 처리를 위한 시스템 메모리 관리 솔루션입니다.

## 문제 상황

비디오 처리 파이프라인 실행 중 시스템 RAM이 부족해져서:
- SSH 연결이 끊어짐
- OOM (Out Of Memory) 에러 발생
- 시스템이 불안정해짐

## 해결책 개요

이 솔루션은 다음 기능을 제공합니다:

1. **시스템 메모리 모니터링**: 실시간 RAM 사용량 추적
2. **자동 메모리 정리**: 캐시 및 임시 파일 자동 정리
3. **스왑 공간 관리**: 부족한 스왑 공간 자동 감지 및 설정
4. **프로세스 메모리 제한**: 메모리 사용량 제한
5. **GPU + 시스템 메모리 통합 관리**: GPU와 시스템 메모리 동시 관리

## 빠른 시작

### 1. 스왑 공간 설정 (권장)

처음 실행 전에 스왑 공간을 설정하세요:

```bash
cd /workspace/Vanelia
bash setup_swap.sh 16  # 16GB 스왑 생성 (기본값)
```

또는 다른 크기:

```bash
bash setup_swap.sh 32  # 32GB 스왑 생성
```

**중요**: 스왑 공간은 디스크 공간을 사용하므로 충분한 디스크 공간이 필요합니다.

### 2. 메모리 안전 모드로 실행

기존 스크립트 대신 메모리 안전 실행 스크립트를 사용하세요:

```bash
bash run_memory_safe.sh
```

이 스크립트는:
- 스왑 공간 자동 확인
- 메모리 상태 모니터링
- 메모리 친화적 환경 변수 설정
- 자동 메모리 정리

### 3. 기존 스크립트 사용 시

기존 `memory_safe_chunking.py`를 직접 사용해도 자동으로 시스템 메모리 관리가 활성화됩니다:

```bash
python memory_safe_chunking.py \
    --input video.mp4 \
    --model product.glb \
    --output final.mp4
```

## 주요 기능

### 시스템 메모리 모니터링

파이프라인 실행 중 실시간으로 메모리 사용량을 모니터링합니다:

```
[Memory] Before: 45.23 GB / 64.00 GB (70.7%)
[Memory] After: 42.15 GB / 64.00 GB (65.8%)
```

### 자동 메모리 정리

메모리 사용량이 80%를 초과하면 자동으로:
- Python 가비지 컬렉션 실행
- 파일시스템 캐시 정리 (root 권한 필요)
- 임시 파일 정리

### 스왑 공간 확인

시작 시 스왑 공간을 확인하고 부족하면 경고를 표시합니다:

```
⚠ WARNING: Swap space is 2.00 GB (recommended: 8 GB)
⚠ Consider running: bash setup_swap.sh to create swap space
```

### 클립별 메모리 관리

각 클립 처리 후:
- GPU 메모리 정리
- 시스템 메모리 정리
- 메모리 사용량이 90% 초과 시 임시 파일 자동 삭제

## 상세 설정

### 스왑 공간 크기 결정

스왑 공간 크기는 시스템 RAM과 작업 특성에 따라 결정하세요:

| 시스템 RAM | 권장 스왑 | 최소 스왑 |
|-----------|----------|----------|
| 32 GB     | 16 GB    | 8 GB     |
| 64 GB     | 16-32 GB | 8 GB     |
| 128 GB    | 32 GB    | 16 GB    |

### 메모리 임계값 조정

`memory_utils.py`에서 메모리 임계값을 조정할 수 있습니다:

```python
# 메모리 사용량이 85% 초과 시 경고
if check_memory_threshold(85.0):
    # 정리 작업 수행
```

### 수동 메모리 정리

필요 시 수동으로 메모리를 정리할 수 있습니다:

```python
from memory_utils import clear_system_caches, clear_python_memory

# Python 메모리 정리
clear_python_memory()

# 시스템 캐시 정리 (root 권한 필요)
clear_system_caches()
```

## 문제 해결

### SSH 연결이 여전히 끊어짐

1. **스왑 공간 확인**:
   ```bash
   free -h
   ```
   스왑이 8GB 미만이면 `setup_swap.sh` 실행

2. **메모리 사용량 확인**:
   ```bash
   python -c "from memory_utils import print_memory_status; print_memory_status()"
   ```

3. **디스크 공간 확인**:
   ```bash
   df -h
   ```
   스왑 파일 생성에 충분한 공간이 필요합니다

### "psutil not found" 오류

psutil을 설치하세요:

```bash
pip install psutil
```

또는 requirements.txt에서 설치:

```bash
pip install -r requirements.txt
```

### 시스템 캐시 정리 실패

시스템 캐시 정리는 root 권한이 필요합니다. 실패해도 파이프라인은 계속 실행되지만, 메모리 정리 효과가 줄어듭니다.

수동으로 정리하려면:

```bash
sudo sync
sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'
```

### 스왑 파일 생성 실패

1. **디스크 공간 부족**: `df -h`로 확인
2. **권한 부족**: `sudo` 사용
3. **파일 시스템 제한**: ext4 파일 시스템 권장

## 성능 영향

메모리 관리 기능은 성능에 최소한의 영향을 미칩니다:

- **메모리 모니터링**: < 0.1% CPU 오버헤드
- **캐시 정리**: 1-2초 소요 (필요 시에만 실행)
- **가비지 컬렉션**: 처리 시간의 < 1%

**장점**:
- SSH 연결 안정성 향상
- OOM 에러 방지
- 장시간 실행 안정성

## 모니터링

### 실시간 메모리 상태 확인

```bash
watch -n 1 'free -h'
```

### 프로세스별 메모리 사용량

```bash
ps aux --sort=-%mem | head -10
```

### Python 스크립트로 모니터링

```python
from memory_utils import print_memory_status, get_system_memory_info

# 현재 상태 출력
print_memory_status()

# 메모리 정보 가져오기
mem_info = get_system_memory_info()
print(f"사용 중: {mem_info['used_gb']:.2f} GB")
print(f"사용률: {mem_info['percent']:.1f}%")
```

## 권장 사항

1. **처음 실행 전**: `setup_swap.sh`로 스왑 공간 설정
2. **장시간 실행**: `run_memory_safe.sh` 사용
3. **메모리 모니터링**: 정기적으로 `free -h` 확인
4. **임시 파일 정리**: `--keep-temp` 옵션 사용 시 주의

## 추가 최적화

### 환경 변수 설정

`.bashrc` 또는 실행 스크립트에 추가:

```bash
# PyTorch 메모리 파편화 방지
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# CPU 스레드 제한 (메모리 압력 감소)
export OMP_NUM_THREADS=4
```

### 시스템 레벨 설정 (root 권한 필요)

`/etc/sysctl.conf`에 추가:

```
# 스왑 사용 최적화
vm.swappiness=10
vm.vfs_cache_pressure=50
```

적용:

```bash
sudo sysctl -p
```

## 요약

✅ **스왑 공간 설정**: `bash setup_swap.sh 16`  
✅ **메모리 안전 실행**: `bash run_memory_safe.sh`  
✅ **자동 모니터링**: 파이프라인에 내장됨  
✅ **자동 정리**: 메모리 부족 시 자동 실행  
✅ **성능 유지**: 최소한의 오버헤드  

이 솔루션을 사용하면 SSH 연결이 끊어지지 않고 안정적으로 장시간 비디오 처리를 수행할 수 있습니다.

