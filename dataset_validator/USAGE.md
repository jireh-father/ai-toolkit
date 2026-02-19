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
# 헤어 항목(hair_shape, bangs_shape, bangs_length) 8점, 얼굴 항목(face_shape/color) 6점
python dataset_validator/validate_dataset.py \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --reference-dir E:\backup\aihub2025_gpu\output\reference_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --threshold-hair 8 \
  --threshold-face 6 \
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

| 모델명 | VRAM (INT4) | VRAM (FP16) |
|--------|-------------|-------------|
| qwen2.5-vl-7b | ~5-6GB | ~14GB |
| **qwen3-vl-8b** (기본) | ~6-7GB | ~16GB |
| internvl2.5-8b | ~6-7GB | ~16GB |
| internvl3-8b | ~6-7GB | ~16GB |
| qwen3-vl-30b-a3b (MoE) | ~7-8GB | ~18GB |
| minicpm-v-2.6 | ~6GB | ~16GB |
| gemma-3-12b-vision | ~8-9GB | ~24GB |

## 5. 평가 항목

| 항목 | 설명 |
|------|------|
| `hair_shape` | OUTPUT 헤어 형태/실루엣이 REFERENCE와 일치하는지 |
| `bangs_shape` | OUTPUT 앞머리 스타일이 REFERENCE와 일치하는지 |
| `bangs_length` | OUTPUT 앞머리 길이가 REFERENCE와 일치하는지 |
| `face_shape_preservation` | OUTPUT 얼굴 형태가 INPUT과 동일한지 |
| `face_color_preservation` | OUTPUT 얼굴/피부 색상이 INPUT과 동일한지 |

## 6. 출력 파일

| 파일 | 설명 |
|------|------|
| `reports/results.json` | 전체 상세 결과 (점수, 사유, 통계) |
| `reports/results.csv` | CSV 요약 |
| `reports/summary.html` | Chart.js 대시보드 (브라우저로 열기) |
| `reports/mismatch_report.json` | 파일명 불일치/손상 목록 |
