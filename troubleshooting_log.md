# FlashAttention CUDA 프로젝트 - 트러블슈팅 로그
> RTX 4060 Ti / Windows 11 / Python 3.11 / CUDA 12.1

---

## 1. AttributeError: total_mem
- **에러:** `'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`
- **원인:** PyTorch 버전마다 속성명이 다름
- **해결:** `props.total_mem` → `props.total_memory`

## 2. AttributeError: max_shared_memory_size
- **에러:** `has no attribute 'max_shared_memory_size'`
- **원인:** 같은 이유. PyTorch 2.5.1에서 해당 속성 없음
- **해결:** `torch.cuda.get_device_capability(0)` 사용으로 대체

## 3. ModuleNotFoundError: No module named 'awq'
- **에러:** autoawq 패키지 미설치
- **해결:** `pip install autoawq`
- **주의:** 소스 빌드 실패 시 `pip install wheel` 먼저, 그 다음 `pip install autoawq --no-build-isolation`

## 4. autoawq 설치 시 torch 다운그레이드
- **에러:** autoawq가 torch 2.5.1 → 2.3.1로 다운그레이드함
- **원인:** autoawq==0.2.6이 torch==2.3.1 의존성 가짐
- **해결:** 버전을 맞추는 게 핵심. torch==2.3.1+cu121 + autoawq==0.2.6 조합이 안정적

## 5. HuggingFace 401 Unauthorized / Repository Not Found
- **에러:** `RepositoryNotFoundError: 401 Client Error`
- **원인 1:** Llama 3는 gated 모델 → HuggingFace 로그인 + Meta License 동의 필요
- **원인 2:** `casperhansen/Meta-Llama-3-8B-Instruct-awq` repo 삭제됨 (404)
- **해결:** 
  - `huggingface-cli login` 으로 토큰 등록
  - ungated 모델로 변경: `TechxGenus/Meta-Llama-3-8B-Instruct-AWQ`

## 6. ValueError: tokenizer does not have a padding token
- **에러:** `Asking to pad but the tokenizer does not have a padding token`
- **원인:** Llama 3 토크나이저에 pad_token 미설정
- **해결:** `tokenizer.pad_token = tokenizer.eos_token` 추가

## 7. AWQ DLL 로드 실패 → 0.6~0.9 tok/s 비정상 속도
- **에러:** UserWarning 5개 (ExLlama, ExLlamaV2, GEMM, GEMV, GEMVFast 커널 전부 실패)
- **원인:** autoawq-kernels와 torch 버전 불일치 (torch 2.5.1 + autoawq-kernels 0.0.7)
- **증상:** fallback 모드로 동작 → 추론 속도 극단적으로 느림
- **해결:** torch==2.3.1+cu121 + autoawq==0.2.6 + autoawq-kernels==0.0.7 조합으로 통일

## 8. ImportError: shard_checkpoint
- **에러:** `cannot import name 'shard_checkpoint' from 'transformers.modeling_utils'`
- **원인:** transformers 5.1.0이 너무 새 버전
- **해결:** `pip install transformers==4.45.0`

## 9. torch가 CPU 버전으로 설치됨
- **에러:** `CUDA available: False`
- **원인:** `pip install torch==2.3.1`이 CPU 버전 다운로드
- **해결:** 반드시 CUDA 인덱스 지정: `pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121`

## 10. NumPy 2.x 호환성
- **에러:** `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.5`
- **원인:** torch 2.3.1이 numpy 1.x 기준으로 빌드됨
- **해결:** `pip install "numpy<2"`

## 11. torchvision register_fake 에러
- **에러:** `module 'torch.library' has no attribute 'register_fake'`
- **원인:** torchvision 0.20.1이 torch 2.5.1 기능 사용, 현재 torch 2.3.1
- **해결:** `pip uninstall torchvision torchaudio -y` (프로젝트에 불필요)

## 12. Nsight Compute "No kernels were profiled"
- **에러:** `==WARNING== No kernels were profiled`
- **원인:** `--kernel-name` 옵션에 정확한 커널 이름 미지정
- **해결:** Available kernels 목록 확인 후 정확한 이름 사용
  - ❌ `--kernel-name flash_attn_fwd`
  - ✅ `--kernel-name flash_attn_fwd_kernel`

---

## 최종 안정 패키지 조합 (RTX 4060 Ti 기준)
```
Python 3.11
torch==2.3.1+cu121
transformers==4.45.0
autoawq==0.2.6
autoawq-kernels==0.0.7
numpy<2
huggingface-hub (latest)
tokenizers (latest)
accelerate (latest)
# torchvision, torchaudio 불필요 → 제거
```

## Jetson AGX Orin 이식 시 주의사항
- Jetson은 JetPack SDK로 torch 설치 (pip이 아님)
- CUDA 버전이 다를 수 있음 (JetPack 6.x → CUDA 12.x)
- autoawq-kernels가 aarch64 빌드 지원 안 할 수 있음 → 소스 빌드 필요
- shared memory carveout이 다름 → cudaDeviceGetAttribute()로 확인
- Compute Capability 8.7 (Ampere) vs 8.9 (Ada) → 커널 arch 플래그 변경 필요: `-arch=sm_87`
