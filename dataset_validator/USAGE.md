# Dataset Validator 사용법

## 1. 데이터셋 검증 (validate_dataset.py)

### 기본 실행
```bash
python dataset_validator/validate_dataset.py \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --reference-dir E:\backup\aihub2025_gpu\output\reference_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --report-dir ./reports \
  --checkpoint-dir ./checkpoints
```

### 소량 테스트 (10건)
```bash
python dataset_validator/validate_dataset.py \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --reference-dir E:\backup\aihub2025_gpu\output\reference_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --max-samples 10 \
  --report-dir ./test_reports \
  --checkpoint-dir ./test_checkpoints
```

### 모델 변경
```bash
python dataset_validator/validate_dataset.py \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --reference-dir E:\backup\aihub2025_gpu\output\reference_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --model qwen2.5-vl-7b \
  --report-dir ./reports
```

### 커스텀 임계값
```bash
# 헤어 항목 8점, naturalness 6점
python dataset_validator/validate_dataset.py \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --reference-dir E:\backup\aihub2025_gpu\output\reference_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --threshold-hair 8 \
  --threshold-naturalness 6 \
  --report-dir ./reports
```

### 중단 후 이어서 실행
```bash
python dataset_validator/validate_dataset.py \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --reference-dir E:\backup\aihub2025_gpu\output\reference_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --resume \
  --report-dir ./reports \
  --checkpoint-dir ./checkpoints
```

### 멀티GPU
```bash
python dataset_validator/validate_dataset.py \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --reference-dir E:\backup\aihub2025_gpu\output\reference_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --num-gpus 2 \
  --report-dir ./reports
```

### 백엔드별 실행 (Local / vLLM / Ollama)

3가지 추론 백엔드를 지원합니다:
- **local**: HuggingFace 모델을 직접 GPU에 로딩 (기본값)
- **vllm**: 외부 vLLM 서버에 API 요청 (로컬 GPU 불필요)
- **ollama**: Ollama 서버에 API 요청 (로컬 GPU 불필요)

#### Local (HuggingFace 직접 로딩)
```bash
# 기본 — GPU에 모델을 직접 로딩하여 추론
python dataset_validator/validate_dataset.py \
  --input-dir ./data/input \
  --reference-dir ./data/reference \
  --output-dir ./data/output \
  --model qwen3.5-9b \
  --backend local \
  --report-dir ./reports
```

#### vLLM 서버
```bash
# 1단계: vLLM 서버 시작
python dataset_validator/serve_vllm.py --model qwen3.5-9b

# 2단계: 검증 실행 (별도 터미널)
python dataset_validator/validate_dataset.py \
  --input-dir ./data/input \
  --reference-dir ./data/reference \
  --output-dir ./data/output \
  --model qwen3.5-9b \
  --backend vllm \
  --vllm-url http://localhost:8000 \
  --report-dir ./reports
```

#### Ollama
```bash
# 1단계: Ollama에서 모델 다운로드
ollama pull qwen3.5:9b

# 2단계: 검증 실행 (Ollama가 실행 중이어야 함)
python dataset_validator/validate_dataset.py \
  --input-dir ./data/input \
  --reference-dir ./data/reference \
  --output-dir ./data/output \
  --model qwen3.5-9b \
  --backend ollama \
  --ollama-url http://localhost:11434 \
  --report-dir ./reports

# test inference

```



> **참고**: Ollama 모델명 매핑은 자동으로 처리됩니다. `--model qwen3.5-9b`로 지정하면 내부적으로 Ollama 태그 `qwen3.5:9b`로 변환됩니다.

## 2. 불합격 데이터 분리 (separate_failed.py)

### 불합격 데이터 복사
```bash
python dataset_validator/separate_failed.py \
  --report ./reports/results.json \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --reference-dir E:\backup\aihub2025_gpu\output\reference_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --failed-dir ./failed
```

### 불합격 데이터 이동 (원본에서 제거)
```bash
python dataset_validator/separate_failed.py \
  --report ./reports/results.json \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --reference-dir E:\backup\aihub2025_gpu\output\reference_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --failed-dir ./failed \
  --mode move
```

### 임계값 재지정하여 분리
```bash
# 리포트 생성 시 threshold=7이었지만, threshold=8로 다시 필터링
python dataset_validator/separate_failed.py \
  --report ./reports/results.json \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --reference-dir E:\backup\aihub2025_gpu\output\reference_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --failed-dir ./failed_strict \
  --threshold 8
```

## 3. 전체 인자 목록

### validate_dataset.py

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input-dir` | **필수** | input 이미지 폴더 |
| `--reference-dir` | **필수** | reference 이미지 폴더 |
| `--output-dir` | **필수** | output 이미지 폴더 |
| `--model` | `qwen3-vl-8b` | VLM 모델 |
| `--quantization` | `int4` | 양자화: int4, int8, fp16 |
| `--num-gpus` | `1` | GPU 수 |
| `--batch-size` | `4` | GPU당 배치 크기 |
| `--backend` | `local` | 추론 백엔드: local, vllm, ollama |
| `--vllm-url` | `http://localhost:8000` | vLLM 서버 URL |
| `--ollama-url` | `http://localhost:11434` | Ollama 서버 URL |
| `--threshold` | `7` | 전체 항목 합격 임계값 |
| `--threshold-hair` | None | 헤어 항목 임계값 |
| `--threshold-face` | None | 얼굴 항목 임계값 |
| `--report-dir` | `./reports` | 리포트 출력 디렉토리 |
| `--checkpoint-dir` | `./checkpoints` | 체크포인트 디렉토리 |
| `--resume` | False | 체크포인트에서 이어서 실행 |
| `--checkpoint-interval` | `100` | 체크포인트 저장 주기 |
| `--resize-short-side` | `512` | 이미지 리사이즈 기준 |
| `--max-samples` | None | 최대 검사 수 (디버깅용) |
| `--seed` | `42` | 랜덤 시드 |
| `--log-level` | `info` | 로깅 레벨 |

### separate_failed.py

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--report` | **필수** | results.json 경로 |
| `--input-dir` | **필수** | input 이미지 원본 폴더 |
| `--reference-dir` | **필수** | reference 이미지 원본 폴더 |
| `--output-dir` | **필수** | output 이미지 원본 폴더 |
| `--failed-dir` | `./failed` | 불합격 데이터 이동 대상 |
| `--mode` | `copy` | copy 또는 move |
| `--threshold` | None | 임계값 재지정 |

## 4. 지원 모델

### 기존 모델

| 모델명 | VRAM (INT4) | VRAM (FP16) |
|--------|-------------|-------------|
| qwen2.5-vl-7b | ~5-6GB | ~14GB |
| **qwen3-vl-8b** (기본) | ~6-7GB | ~16GB |
| internvl2.5-8b | ~6-7GB | ~16GB |
| internvl3-8b | ~6-7GB | ~16GB |
| qwen3-vl-30b-a3b (MoE) | ~7-8GB | ~18GB |
| minicpm-v-2.6 | ~6GB | ~16GB |
| gemma-3-12b-vision | ~8-9GB | ~24GB |

### Qwen3.5 (통합 비전-언어 모델)

Qwen3.5는 별도 VL 변형 없이 기본 모델 자체가 이미지를 지원하는 통합 모델입니다.
Local, vLLM, Ollama 3가지 백엔드 모두 사용 가능합니다.

| 모델명 | 파라미터 | VRAM (INT4) | VRAM (FP16) | Ollama 태그 |
|--------|----------|-------------|-------------|-------------|
| qwen3.5-0.8b | 0.8B | ~1GB | ~2GB | qwen3.5:0.8b |
| qwen3.5-2b | 2B | ~2GB | ~4GB | qwen3.5:2b |
| qwen3.5-4b | 4B | ~3GB | ~8GB | qwen3.5:4b |
| qwen3.5-9b | 9B | ~6GB | ~18GB | qwen3.5:9b |
| qwen3.5-27b | 27B | ~16GB | ~54GB | qwen3.5:27b |
| qwen3.5-35b-a3b | 35B (MoE, 활성 3B) | ~8GB | ~20GB | qwen3.5:35b-a3b |
| qwen3.5-122b-a10b | 122B (MoE, 활성 10B) | ~30GB | ~80GB | qwen3.5:122b |
| qwen3.5-397b-a17b | 397B (MoE, 활성 17B) | ~60GB | ~150GB | qwen3.5:397b |

## 5. 평가 항목

| 항목 | 설명 |
|------|------|
| `hair_similarity_overall` | OUTPUT 헤어스타일이 REFERENCE와 전체적으로 일치하는지 |
| `hair_color` | OUTPUT 헤어 색상이 REFERENCE와 일치하는지 |
| `hair_length` | OUTPUT 헤어 길이가 REFERENCE와 일치하는지 |
| `hair_texture` | OUTPUT 헤어 질감이 REFERENCE와 일치하는지 |
| `hair_shape` | OUTPUT 헤어 형태/실루엣이 REFERENCE와 일치하는지 |
| `hair_sharpness_vs_reference` | OUTPUT 헤어 선명도가 REFERENCE와 비교하여 유사한지 |
| `hair_detail` | OUTPUT 헤어의 세밀한 디테일 표현 수준 |
| `naturalness` | 헤어 편집이 자연스럽게 보이는지 |

## 6. 출력 파일

| 파일 | 설명 |
|------|------|
| `reports/results.json` | 전체 상세 결과 (점수, 사유, 통계) |
| `reports/results.csv` | CSV 요약 |
| `reports/summary.html` | Chart.js 대시보드 (브라우저로 열기) |
| `reports/mismatch_report.json` | 파일명 불일치/손상 목록 |
