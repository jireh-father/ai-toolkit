# PRD: Hair Transfer 학습 데이터셋 자동 품질 검사 시스템

## 1. 개요

### 1.1 목적
이미지 에디트 모델(Hair Transfer) 학습용 데이터셋의 품질을 오픈소스 VLM(Vision Language Model)을 활용하여 자동으로 검사하는 시스템을 개발한다.

### 1.2 배경
- Hair Transfer 모델은 input 이미지의 헤어스타일을 reference 이미지의 헤어스타일로 변환하여 output을 생성하는 모델이다.
- 학습 데이터는 **input / reference / output** 3장의 이미지가 한 쌍이며, 각 폴더에 **동일한 파일명**으로 저장되어 있다.
- 데이터셋 규모가 **10,000쌍 이상**으로 수동 검수가 비현실적이므로 자동 검사가 필요하다.

### 1.3 핵심 검사 기준
| 기준 | 설명 |
|------|------|
| **헤어스타일 유사도** | output의 헤어스타일이 reference의 헤어스타일과 유사한지 (각도는 input 기준) |
| **비변경 영역 보존** | output에서 헤어 외 영역(얼굴, 배경 등)이 input과 동일한지 |
| **합성 자연스러움** | 헤어 합성 경계가 부자연스럽거나 아티팩트가 없는지 |

> **중요**: reference의 헤어스타일 각도와 output의 헤어스타일 각도는 달라도 된다. input 사진의 각도에 맞게 자연스럽게 합성되었는지가 핵심이다.

---

## 2. 오픈소스 VLM 모델 리서치

### 2.1 추천 모델 리스트 (VRAM 효율 기준 정렬)

| 순위 | 모델 | 파라미터 | 양자화 시 VRAM | 멀티이미지 | 라이선스 | 비고 |
|------|------|----------|---------------|-----------|----------|------|
| 1 | **Qwen2.5-VL-7B-Instruct** | 7B | ~5-6GB (INT4) | O | Apache 2.0 | 멀티이미지 네이티브 지원, 동적 해상도, 한국어 지원 |
| 2 | **Qwen3-VL-8B-Instruct** | 8B | ~6-7GB (INT4) | O | Apache 2.0 | 최신 모델, 향상된 시각 추론, 256K 컨텍스트 |
| 3 | **InternVL2.5-8B** | 8B | ~6-7GB (INT4) | O | MIT | MMMU 벤치마크 강세, 멀티이미지 비교에 강점 |
| 4 | **InternVL3-8B** | 8B | ~6-7GB (INT4) | O | MIT | InternVL 최신, 시각 추론 개선 |
| 5 | **Qwen3-VL-30B-A3B-Instruct** | 30B (MoE, 활성 3B) | ~7-8GB (INT4) | O | Apache 2.0 | MoE 구조로 실제 활성 파라미터 3B, VRAM 효율적 |
| 6 | **MiniCPM-V-2.6** | ~8B | ~6GB (INT4) | O | Apache 2.0 | 경량 모델 중 가장 안정적 성능, 멀티이미지 지원 |
| 7 | **Gemma-3-12B-Vision** | 12B | ~8-9GB (INT4) | O | Gemma | Google 오픈소스, 128K 컨텍스트 |

### 2.2 모델 선택 가이드

```
GPU VRAM별 권장 모델:
┌─────────────┬──────────────────────────────────────┐
│ 8GB VRAM    │ Qwen2.5-VL-7B (INT4)                 │
│             │ MiniCPM-V-2.6 (INT4)                  │
├─────────────┼──────────────────────────────────────┤
│ 10GB VRAM   │ Qwen3-VL-8B (INT4)                    │
│             │ InternVL2.5-8B (INT4)                  │
├─────────────┼──────────────────────────────────────┤
│ 12GB VRAM   │ Qwen3-VL-30B-A3B (INT4, MoE)          │
│             │ Gemma-3-12B-Vision (INT4)              │
├─────────────┼──────────────────────────────────────┤
│ 24GB VRAM   │ Qwen2.5-VL-7B (FP16, 최고 정확도)      │
│             │ InternVL3-8B (FP16)                    │
├─────────────┼──────────────────────────────────────┤
│ 48GB+ VRAM  │ Qwen3-VL-32B (FP16)                   │
│             │ InternVL2.5-78B (INT4)                 │
└─────────────┴──────────────────────────────────────┘
```

### 2.3 1순위 추천: Qwen2.5-VL-7B-Instruct
- **이유**: 멀티이미지 네이티브 지원, 동적 해상도 처리, INT4 양자화 시 8GB VRAM으로 구동 가능
- **강점**: 이미지 비교 태스크에서 높은 정확도, 한국어 프롬프트 지원, 활발한 커뮤니티
- **추론 프레임워크**: vLLM (AWQ/GPTQ 양자화) 또는 llama.cpp (GGUF 양자화) 중 VRAM이 더 적게 드는 방식 선택

---

## 3. 시스템 아키텍처

### 3.1 전체 구조

```
[데이터셋 폴더]          [검사 시스템]              [출력]

input/              ┌──────────────────┐     reports/
  ├─ 001.jpg        │  Image Loader    │       ├─ results.json
  ├─ 002.jpg   ──►  │  (리사이즈/전처리) │       ├─ results.csv
  └─ ...            └────────┬─────────┘       ├─ summary.html
                             │                  └─ failed_samples/
reference/                   ▼                       ├─ 001_comparison.jpg
  ├─ 001.jpg        ┌──────────────────┐             ├─ 002_comparison.jpg
  ├─ 002.jpg   ──►  │  VLM Evaluator   │             └─ ...
  └─ ...            │  (배치 + 멀티GPU) │
                    └────────┬─────────┘
output/                      │
  ├─ 001.jpg        ┌────────▼─────────┐
  ├─ 002.jpg   ──►  │  Report Generator │
  └─ ...            │  (JSON/CSV/HTML)  │
                    └──────────────────┘
```

### 3.2 처리 파이프라인

```
1. 초기화
   ├─ CLI 인자 파싱
   ├─ 데이터셋 폴더 스캔 (input/reference/output 매칭)
   ├─ 체크포인트 로드 (이전 진행 상태 확인)
   └─ VLM 모델 로드 (양자화 적용)

2. 전처리
   ├─ 이미지 쌍 리스트 생성
   ├─ 종횡비 유지 리사이즈 (짧은 변 기준 512px)
   └─ 배치 구성

3. VLM 평가 (배치 + 멀티GPU 병렬)
   ├─ 3장 이미지 동시 입력
   ├─ 구조화된 JSON 응답 파싱
   └─ 체크포인트 주기적 저장

4. 결과 집계
   ├─ 항목별 점수 기록
   ├─ 임계값 기반 Pass/Fail 판정
   └─ 통계 계산

5. 리포트 생성
   ├─ JSON 상세 리포트
   ├─ CSV 요약 리포트
   ├─ HTML 시각화 리포트
   └─ 불합격 샘플 비교 이미지 생성
```

---

## 4. 상세 기능 명세

### 4.1 VLM 평가 항목

각 이미지 쌍(input, reference, output)에 대해 VLM이 아래 7개 항목을 평가한다:

| # | 필드명 | 점수 범위 | 설명 |
|---|--------|----------|------|
| 1 | `hair_similarity_overall` | 0~10 | reference 대비 output 헤어스타일 전체적 유사도 |
| 2 | `hair_color` | 0~10 | 헤어 색상 일치도 |
| 3 | `hair_length` | 0~10 | 헤어 길이 일치도 |
| 4 | `hair_texture` | 0~10 | 헤어 질감 일치도 (웨이브/컬/스트레이트/파마 등) |
| 5 | `non_hair_preservation` | 0~10 | 비변경 영역(얼굴, 의상, 배경 등) 보존도 |
| 6 | `naturalness` | 0~10 | 합성 자연스러움 (경계, 아티팩트, 조명 일관성) |
| 7 | `reason` | string | 판단 근거 텍스트 (영어 또는 한국어) |

### 4.2 합격/불합격 판정 로직

```python
# 기본값: 모든 항목 7점 이상이어야 Pass
def is_pass(scores: dict, threshold: int = 7) -> bool:
    score_fields = [
        'hair_similarity_overall', 'hair_color', 'hair_length',
        'hair_texture', 'non_hair_preservation', 'naturalness'
    ]
    return all(scores[field] >= threshold for field in score_fields)
```

- **임계값 기본값**: 모든 항목 7점
- **CLI에서 조정 가능**: `--threshold 8` (전체 항목 동일 적용)
- **항목별 개별 임계값도 지원**:
  - `--threshold-hair`: `hair_similarity_overall`, `hair_color`, `hair_length`, `hair_texture` 4개 항목에 적용
  - `--threshold-preservation`: `non_hair_preservation` 항목에 적용
  - `--threshold-naturalness`: `naturalness` 항목에 적용
  - 미지정 시 `--threshold` 값을 공통 적용

### 4.3 이미지 전처리

```
리사이즈 전략:
- 종횡비 유지
- 짧은 변(가로/세로 중 작은 쪽)을 512px로 리사이즈
- 긴 변은 비율에 따라 자동 조정
- 예: 1024x768 → 683x512, 768x1024 → 512x683
```

**지원 이미지 포맷**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`

### 4.4 데이터 무결성 검증 (전처리 단계)

VLM 평가 전에 아래 사전 검증을 수행한다:

| 검증 항목 | 처리 방식 |
|-----------|-----------|
| **파일명 불일치**: input에는 있지만 reference/output에 없는 파일 | 경고 로그 출력 + 해당 파일 스킵, 불일치 목록을 리포트에 기록 |
| **파일명 불일치**: reference/output에만 있는 파일 | 경고 로그 출력 + 해당 파일 스킵 |
| **이미지 손상**: 열 수 없는 파일 | 에러 로그 + 스킵, 손상 목록을 리포트에 기록 |
| **3개 폴더 모두 매칭되는 파일만** 검사 대상에 포함 | 매칭 결과 요약을 시작 시 출력 |

```
예시 로그:
[INFO] 데이터셋 스캔 완료: input=10,500, reference=10,480, output=10,450
[INFO] 3폴더 공통 매칭: 10,420건 (검사 대상)
[WARN] 매칭 실패 80건 → mismatch_report.json 참조
```

### 4.5 VLM 프롬프트 설계 (예시)

```
You are an expert image quality assessor for hair transfer (hair editing) models.

You are given three images:
1. INPUT: The original photo of a person before hair editing
2. REFERENCE: A photo showing the target hairstyle to be applied
3. OUTPUT: The result image after applying the REFERENCE hairstyle onto the INPUT person

## Key Rules
- The hairstyle in OUTPUT should replicate the REFERENCE hairstyle (color, length, texture/curl pattern).
- However, the hair ANGLE and POSE in OUTPUT must match the INPUT person's head angle and pose,
  NOT the REFERENCE photo's angle. The REFERENCE may show a completely different person from a
  different angle — only the hairstyle itself (style, color, length, texture) matters.
- Everything EXCEPT the hair region in OUTPUT must remain identical to INPUT
  (face, skin, eyes, clothing, background, accessories, etc.).
- The hair-to-face boundary in OUTPUT should look natural with no visible seams, artifacts,
  color bleeding, or unnatural lighting transitions.

## Evaluation Criteria
Evaluate the OUTPUT image and respond ONLY with a valid JSON object (no markdown, no extra text):
{
  "hair_similarity_overall": <0-10>,
  "hair_color": <0-10>,
  "hair_length": <0-10>,
  "hair_texture": <0-10>,
  "non_hair_preservation": <0-10>,
  "naturalness": <0-10>,
  "reason": "<1-2 sentence explanation of key observations>"
}

## Field Descriptions
- hair_similarity_overall: How well does the OUTPUT hairstyle match the REFERENCE hairstyle overall?
- hair_color: Does the hair color in OUTPUT match the REFERENCE? (consider highlights, roots, gradient)
- hair_length: Does the hair length in OUTPUT match the REFERENCE?
- hair_texture: Does the hair texture in OUTPUT match the REFERENCE? (straight/wavy/curly/permed)
- non_hair_preservation: Are all non-hair regions (face, body, background) in OUTPUT identical to INPUT?
- naturalness: Does the hair edit look natural? (no artifacts, seamless blending, consistent lighting)

## Scoring Guide
- 0-3: Very poor — obvious failures, wrong hairstyle, severe artifacts, face changed
- 4-6: Mediocre — noticeable issues, partial match, visible seams or color mismatch
- 7-8: Good — minor issues only, overall convincing result
- 9-10: Excellent — near perfect, indistinguishable from real photo
```

---

## 5. CLI 인터페이스

### 5.1 메인 검사 스크립트: `validate_dataset.py`

```bash
python validate_dataset.py \
  --input-dir ./data/input \
  --reference-dir ./data/reference \
  --output-dir ./data/output \
  --model qwen2.5-vl-7b \
  --num-gpus 1 \
  --batch-size 4 \
  --threshold 7 \
  --report-dir ./reports \
  --checkpoint-dir ./checkpoints
```

#### 전체 인자 목록

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--input-dir` | str | **필수** | input 이미지 폴더 경로 |
| `--reference-dir` | str | **필수** | reference 이미지 폴더 경로 |
| `--output-dir` | str | **필수** | output 이미지 폴더 경로 |
| `--model` | str | `qwen2.5-vl-7b` | VLM 모델 선택 (아래 지원 모델 목록 참조) |
| `--num-gpus` | int | `1` | 사용할 GPU 수 (병렬 처리) |
| `--batch-size` | int | `4` | 배치 크기 (GPU당) |
| `--threshold` | int | `7` | 전체 항목 합격 임계값 (0~10) |
| `--threshold-hair` | int | None | 헤어 관련 항목 임계값 (미지정 시 --threshold 사용) |
| `--threshold-preservation` | int | None | 비변경 영역 보존 임계값 (미지정 시 --threshold 사용) |
| `--threshold-naturalness` | int | None | 자연스러움 임계값 (미지정 시 --threshold 사용) |
| `--report-dir` | str | `./reports` | 리포트 출력 디렉토리 |
| `--checkpoint-dir` | str | `./checkpoints` | 체크포인트 저장 디렉토리 |
| `--resume` | flag | False | 이전 체크포인트에서 이어서 실행 |
| `--checkpoint-interval` | int | `100` | 체크포인트 저장 주기 (N건마다 저장) |
| `--resize-short-side` | int | `512` | 리사이즈 기준 짧은 변 크기 |
| `--quantization` | str | `int4` | 양자화 방식: `int4`, `int8`, `fp16` |
| `--max-samples` | int | None | 최대 검사 수 (디버깅용, 미지정 시 전체) |
| `--seed` | int | `42` | 랜덤 시드 |
| `--log-level` | str | `info` | 로깅 레벨: `debug`, `info`, `warning`, `error` |

#### 지원 모델 목록 (`--model` 인자)

```
사용법: --model <모델명>

모델명                          VRAM (INT4)   VRAM (FP16)   정확도
─────────────────────────────────────────────────────────────────
qwen2.5-vl-7b                   ~5-6GB        ~14GB         ★★★★☆
qwen3-vl-8b                     ~6-7GB        ~16GB         ★★★★★
internvl2.5-8b                  ~6-7GB        ~16GB         ★★★★☆
internvl3-8b                    ~6-7GB        ~16GB         ★★★★★
qwen3-vl-30b-a3b (MoE)         ~7-8GB        ~18GB         ★★★★★
minicpm-v-2.6                   ~6GB          ~16GB         ★★★☆☆
gemma-3-12b-vision              ~8-9GB        ~24GB         ★★★★☆
```

### 5.2 불합격 데이터 분리 스크립트: `separate_failed.py`

```bash
python separate_failed.py \
  --report ./reports/results.json \
  --input-dir ./data/input \
  --reference-dir ./data/reference \
  --output-dir ./data/output \
  --failed-dir ./data/failed
```

#### 인자 목록

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--report` | str | **필수** | validate_dataset.py가 생성한 JSON 리포트 경로 |
| `--input-dir` | str | **필수** | input 이미지 원본 폴더 |
| `--reference-dir` | str | **필수** | reference 이미지 원본 폴더 |
| `--output-dir` | str | **필수** | output 이미지 원본 폴더 |
| `--failed-dir` | str | `./failed` | 불합격 데이터 이동 대상 폴더 |
| `--mode` | str | `copy` | `copy` 또는 `move` (기본: 복사) |
| `--threshold` | int | None | 임계값 재지정 (리포트의 원래 임계값 대신 이 값으로 재필터링) |

#### 출력 구조

```
failed/
  ├─ input/
  │    ├─ 001.jpg
  │    └─ 053.jpg
  ├─ reference/
  │    ├─ 001.jpg
  │    └─ 053.jpg
  └─ output/
       ├─ 001.jpg
       └─ 053.jpg
```

---

## 6. 출력 리포트 상세

### 6.1 JSON 리포트 (`results.json`)

```json
{
  "metadata": {
    "model": "qwen2.5-vl-7b",
    "threshold": 7,
    "total_samples": 10500,
    "passed": 9200,
    "failed": 1300,
    "pass_rate": 87.6,
    "timestamp": "2026-02-07T15:30:00",
    "elapsed_time_sec": 12600
  },
  "statistics": {
    "hair_similarity_overall": {"mean": 7.8, "std": 1.2, "min": 2, "max": 10},
    "hair_color": {"mean": 8.1, "std": 1.0, "min": 3, "max": 10},
    "hair_length": {"mean": 7.9, "std": 1.1, "min": 2, "max": 10},
    "hair_texture": {"mean": 7.5, "std": 1.3, "min": 1, "max": 10},
    "non_hair_preservation": {"mean": 8.5, "std": 0.9, "min": 4, "max": 10},
    "naturalness": {"mean": 7.7, "std": 1.2, "min": 2, "max": 10}
  },
  "results": [
    {
      "filename": "001.jpg",
      "pass": true,
      "scores": {
        "hair_similarity_overall": 8,
        "hair_color": 9,
        "hair_length": 8,
        "hair_texture": 7,
        "non_hair_preservation": 9,
        "naturalness": 8
      },
      "reason": "Hairstyle matches reference well. Color and length are accurate..."
    },
    {
      "filename": "053.jpg",
      "pass": false,
      "scores": {
        "hair_similarity_overall": 5,
        "hair_color": 6,
        "hair_length": 4,
        "hair_texture": 5,
        "non_hair_preservation": 8,
        "naturalness": 3
      },
      "reason": "Hair length significantly shorter than reference. Visible artifacts at hairline boundary..."
    }
  ]
}
```

### 6.2 CSV 리포트 (`results.csv`)

```csv
filename,pass,hair_similarity_overall,hair_color,hair_length,hair_texture,non_hair_preservation,naturalness,reason
001.jpg,true,8,9,8,7,9,8,"Hairstyle matches reference well..."
053.jpg,false,5,6,4,5,8,3,"Hair length significantly shorter..."
```

### 6.3 HTML 리포트 (`summary.html`)

HTML 리포트에 포함되는 내용:

1. **대시보드 요약**
   - 전체 Pass/Fail 비율 (파이 차트)
   - 항목별 평균 점수 (바 차트)
   - 항목별 점수 분포 (히스토그램)

2. **불합격 샘플 갤러리**
   - 각 불합격 샘플마다 input | reference | output 3장 가로 나란히 배치
   - 각 항목별 점수 표시
   - VLM 판단 사유 텍스트
   - 점수 낮은 순으로 정렬

3. **합격 샘플 갤러리** (상위 N개만)
   - 고득점 합격 샘플 예시

4. **필터링/정렬 기능**
   - 항목별 점수 필터
   - Pass/Fail 필터
   - 파일명 검색

---

## 7. 병렬 처리 및 성능

### 7.1 멀티GPU 병렬 처리

```
[GPU 0] ──► 배치 0, 4, 8, 12, ...
[GPU 1] ──► 배치 1, 5, 9, 13, ...
[GPU 2] ──► 배치 2, 6, 10, 14, ...
[GPU 3] ──► 배치 3, 7, 11, 15, ...
```

- 각 GPU에 독립적인 VLM 인스턴스를 로드
- 데이터를 GPU 수만큼 분할하여 병렬 처리
- `multiprocessing` 또는 `torch.distributed` 활용

### 7.2 배치 처리

- GPU당 배치 사이즈를 `--batch-size`로 설정
- VLM의 멀티이미지 입력 지원을 활용하여 한 번에 여러 쌍 평가
- I/O 병목 방지를 위해 CPU 워커에서 이미지 전처리를 미리 수행 (prefetch)

### 7.3 체크포인트 / 리즘

- `--checkpoint-interval N`으로 설정한 주기(기본 100건)마다 체크포인트 자동 저장
- 체크포인트 파일에는 처리 완료된 파일명 목록과 결과가 포함
- `--resume` 플래그로 이전 체크포인트에서 이어서 실행
- 비정상 종료(Ctrl+C, OOM 등) 시에도 마지막 체크포인트까지의 결과는 보존

```json
// checkpoint 파일 예시: checkpoints/checkpoint_002100.json
{
  "processed_files": ["001.jpg", "002.jpg", ...],
  "processed_count": 2100,
  "total_count": 10420,
  "results": [...],
  "model": "qwen2.5-vl-7b",
  "quantization": "int4",
  "timestamp": "2026-02-07T16:00:00"
}
```

### 7.4 진행률 표시

- `tqdm` 기반 프로그레스 바 표시
- 현재 처리 건수 / 전체 건수 / 예상 남은 시간(ETA) / 처리 속도(it/s) 표시
- 멀티GPU 시 각 GPU의 진행률을 개별 표시

```
[GPU 0] ████████████████████░░░░  82% | 4100/5000 [1:15:30<16:40, 0.9it/s]
[GPU 1] ███████████████████░░░░░  78% | 3900/5000 [1:15:30<21:00, 0.87it/s]
[Total] ████████████████████░░░░  80% | 8000/10000 [1:15:30<18:45]
```

### 7.5 예상 처리 시간

```
모델: qwen2.5-vl-7b (INT4), 단일 GPU 12GB 기준
이미지 3장 평가 1건: ~2-3초
배치 사이즈 4: ~6-8초 (4건)
10,000건 처리: ~1.5-2시간 (배치 4 기준)

멀티GPU (x2): ~0.75-1시간
멀티GPU (x4): ~0.4-0.5시간
```

---

## 8. 기술 스택

| 구성 요소 | 기술 | 선택 이유 |
|-----------|------|-----------|
| 언어 | Python 3.10+ | ML 생태계 표준 |
| VLM 추론 | vLLM (AWQ 양자화) | VRAM 최적화, 배치 추론 지원, PagedAttention으로 메모리 효율 극대화 |
| 대안 추론 | llama-cpp-python (GGUF) | GGUF 양자화 시 vLLM보다 VRAM 더 절약 가능, 모델에 따라 선택 |
| 이미지 처리 | Pillow, OpenCV | 리사이즈, 비교 이미지 생성 |
| 병렬 처리 | multiprocessing | 멀티GPU 워커 관리 |
| 리포트 | Jinja2 | HTML 리포트 템플릿 렌더링 |
| 양자화 | AutoAWQ / AutoGPTQ / bitsandbytes | INT4/INT8 양자화로 VRAM 최소화 |
| 진행률 | tqdm | 프로그레스 바 표시, ETA 계산 |
| CLI | argparse | 표준 라이브러리, 의존성 없음 |
| 데이터 | pandas | CSV 처리, 통계 계산 |

---

## 9. 파일 구조

```
dataset_validator/
├── validate_dataset.py       # 메인 검사 스크립트 (엔트리포인트)
├── separate_failed.py        # 불합격 데이터 분리 스크립트
├── config/
│   └── models.yaml           # 지원 모델별 설정 (모델경로, VRAM, 양자화 등)
├── core/
│   ├── __init__.py
│   ├── evaluator.py          # VLM 평가 로직 (프롬프트, 파싱)
│   ├── image_loader.py       # 이미지 로드 및 전처리
│   ├── checkpoint.py         # 체크포인트 관리
│   └── parallel.py           # 멀티GPU 병렬 처리
├── report/
│   ├── __init__.py
│   ├── generator.py          # JSON/CSV/HTML 리포트 생성
│   └── templates/
│       └── summary.html      # HTML 리포트 Jinja2 템플릿
└── requirements.txt
```

---

## 10. 리스크 및 제약사항

| 리스크 | 영향 | 대응 방안 |
|--------|------|-----------|
| VLM 판단 일관성 부족 | 동일 이미지에 대해 다른 점수 | temperature=0 설정, 프롬프트 고정, seed 고정 |
| VRAM 부족 (8GB) | 대형 모델 사용 불가 | INT4 양자화 필수, 이미지 리사이즈 512px |
| JSON 파싱 실패 | VLM이 잘못된 형식 출력 | JSON 파싱 재시도 로직, 실패 시 재평가 (최대 3회) |
| 처리 속도 | 10,000건 기준 수 시간 소요 | 배치 처리, 멀티GPU, 체크포인트 리즘 |
| 헤어스타일 세부 판단 한계 | 경량 VLM의 세밀한 판단 능력 제한 | 프롬프트 최적화, 필요시 상위 모델로 교체 |
| 데이터 불일치 | 3폴더 간 파일명 매칭 안 됨 | 사전 검증 단계에서 불일치 감지 및 스킵, 불일치 목록 리포트 |
| 이미지 손상 | 열 수 없는 이미지 파일 | 사전 검증에서 감지 후 스킵, 손상 목록 리포트 |

---

## 11. 향후 확장 가능성

- **커스텀 평가 항목 추가**: 프롬프트 수정만으로 평가 항목 확장 가능
- **2-Stage 검증**: 1차 경량 모델 필터링 → 2차 대형 모델 정밀 검사
- **API 서버화**: FastAPI 기반 REST API로 웹 서비스 전환
- **재생성 파이프라인 연동**: 불합격 데이터를 ComfyUI 파이프라인으로 자동 재생성

---

## 참고 자료

- [Qwen2.5-VL-7B-Instruct - Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [InternVL2.5 Blog](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)
- [InternVL3.5-8B - Hugging Face](https://huggingface.co/OpenGVLab/InternVL3_5-8B)
- [Top Open-Source VLMs - Labellerr](https://www.labellerr.com/blog/top-open-source-vision-language-models/)
- [Multimodal AI: Open-Source VLMs - BentoML](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [Top 10 VLMs 2026 - DataCamp](https://www.datacamp.com/blog/top-vision-language-models)
- [Benchmarking VLMs - Clarifai](https://www.clarifai.com/blog/benchmarking-best-open-source-vision-language-models)
