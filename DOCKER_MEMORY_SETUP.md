# Docker 환경에서 메모리 관리 설정

Docker 컨테이너에서는 스왑 파일 생성이 제한될 수 있습니다. 대신 호스트 레벨에서 메모리와 스왑을 설정해야 합니다.

## Docker 컨테이너 실행 시 메모리 설정

### 1. 메모리 및 스왑 제한 설정

Docker 컨테이너를 실행할 때 메모리와 스왑을 명시적으로 설정하세요:

```bash
docker run \
    --memory=64g \
    --memory-swap=80g \
    --memory-reservation=32g \
    ...
```

**설명**:
- `--memory=64g`: 최대 64GB RAM 사용
- `--memory-swap=80g`: RAM + 스왑 합계 80GB (스왑 = 16GB)
- `--memory-reservation=32g`: 최소 32GB 보장

### 2. Docker Compose 설정

`docker-compose.yml`에서:

```yaml
services:
  vanelia:
    deploy:
      resources:
        limits:
          memory: 64g
          memory-swap: 80g
        reservations:
          memory: 32g
```

또는 간단하게:

```yaml
services:
  vanelia:
    mem_limit: 64g
    memswap_limit: 80g
```

### 3. 호스트 레벨 스왑 설정

컨테이너 내부가 아닌 호스트에서 스왑을 설정:

```bash
# 호스트에서 실행
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구적으로 설정
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 4. 컨테이너 내부에서 확인

컨테이너 내부에서는 스왑이 제한될 수 있지만, 메모리 관리 기능은 여전히 작동합니다:

```bash
# 메모리 상태 확인
free -h

# 메모리 관리 유틸리티 사용
python -c "from memory_utils import print_memory_status; print_memory_status()"
```

## 권장 설정

### RunPod / 클라우드 환경

RunPod나 다른 클라우드 환경에서는:

1. **인스턴스 타입 선택**: 충분한 RAM이 있는 인스턴스 선택
2. **스왑 활성화**: 인스턴스 설정에서 스왑 활성화
3. **메모리 모니터링**: 정기적으로 메모리 사용량 확인

### 로컬 Docker

```bash
# 메모리 제한과 함께 실행
docker run -it \
    --memory=64g \
    --memory-swap=80g \
    --shm-size=8g \
    -v /workspace:/workspace \
    your-image
```

## 문제 해결

### "swapon failed: Operation not permitted"

이는 정상입니다. Docker 컨테이너에서는 스왑 파일을 직접 활성화할 수 없습니다.

**해결책**:
- 호스트 레벨에서 스왑 설정
- Docker 실행 시 `--memory-swap` 옵션 사용
- 메모리 관리 기능은 스왑 없이도 작동합니다

### 메모리 부족으로 SSH 연결 끊김

1. **호스트 메모리 확인**:
   ```bash
   # 호스트에서
   free -h
   ```

2. **컨테이너 메모리 제한 확인**:
   ```bash
   docker stats
   ```

3. **메모리 제한 증가**:
   ```bash
   docker update --memory=128g --memory-swap=144g container_name
   ```

## 메모리 관리 기능

스왑이 없어도 다음 기능들이 작동합니다:

✅ **자동 메모리 모니터링**  
✅ **Python 가비지 컬렉션**  
✅ **임시 파일 자동 정리**  
✅ **GPU 메모리 관리**  
✅ **메모리 사용량 경고**  

스왑이 있으면 더 안정적이지만, 없어도 메모리 관리 기능으로 안정성을 크게 향상시킬 수 있습니다.

