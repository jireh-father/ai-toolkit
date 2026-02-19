# Spec: 서브에이전트 기반 Hair Transfer 데이터셋 품질 검사 시스템

## 1. 개요

### 1.1 목적
Claude Code의 Task 도구(서브에이전트)와 Opus 4.6의 내장 비전 기능을 활용하여, GPU VLM 없이 Hair Transfer 학습 데이터셋의 품질을 자동 검사하는 시스템을 구축한다.

### 1.2 기존 시스템과의 차이

| 항목 | 기존 (`dataset_validator/`) | 변경 (서브에이전트 기반) |
|------|---------------------------|----------------------|
| 평가 엔진 | 로컬 GPU VLM (Qwen, InternVL 등) | Claude Code Task 도구 + Opus 4.6 비전 |
| GPU 필요 여부 | 필수 (INT4 기준 5~9GB VRAM) | 불필요 |
| ML 의존성 | torch, transformers, bitsandbytes 등 | 없음 |
| 병렬 처리 | `multiprocessing` + 멀티GPU | Task(run_in_background=true) + 서브에이전트 |
| 이미지 입력 방식 | `PIL.Image.open()` + 리사이즈 | 서브에이전트가 Read 도구로 이미지 파일 직접 읽기 (Opus 4.6 멀티모달) |
| Python 의존성 | `requirements.txt` 전체 | Pillow, numpy, Jinja2만 필요 (리포트 생성용) |

### 1.3 핵심 검사 기준 (기존 PRD와 동일)
| 기준 | 설명 |
|------|------|
| **헤어스타일 유사도** | output의 헤어스타일이 reference의 헤어스타일과 유사한지 (각도는 input 기준) |
| **비변경 영역 보존** | output에서 헤어 외 영역(얼굴, 배경 등)이 input과 동일한지 |
| **합성 자연스러움** | 헤어 합성 경계가 부자연스럽거나 아티팩트가 없는지 |

> **중요**: reference의 헤어스타일 각도와 output의 헤어스타일 각도는 달라도 된다. input 사진의 각도에 맞게 자연스럽게 합성되었는지가 핵심이다.

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
[데이터셋 폴더]          [Claude Code 메인 에이전트]           [출력]

input/              ┌──────────────────────────┐       reports/
  ├─ 001.jpg        │ 1. scan_dataset.py 실행    │         ├─ results.json
  ├─ 002.jpg   ──►  │ 2. validate_images.py 실행 │         ├─ results.csv
  └─ ...            │ 3. 배치 분할               │         ├─ summary.html
                    │ 4. 서브에이전트 N개 병렬 실행 │         └─ mismatch_report.json
reference/          │ 5. 결과 수집/병합           │
  ├─ 001.jpg   ──►  │ 6. 체크포인트 저장          │
  └─ ...            │ 7. generate_reports.py 실행 │
                    └────────────┬─────────────┘
output/                          │
  ├─ 001.jpg                     ▼
  └─ ...            ┌──────────────────────────┐
                    │  서브에이전트 (general-purpose)│
                    │  ├─ Read로 이미지 3장 읽기   │
                    │  ├─ Opus 4.6 비전으로 평가   │
                    │  └─ JSON 결과 Write         │
                    └──────────────────────────┘
```

### 2.2 처리 파이프라인

```
1. 데이터셋 스캔 (메인 에이전트)
   ├─ Bash: python scan_dataset.py → matched.json 생성
   │    (기존 image_loader.py의 scan_dataset() 재활용)
   └─ 매칭 결과: matched 건수, mismatched 건수 확인

2. 이미지 무결성 검증 (메인 에이전트)
   ├─ Bash: python validate_images.py → valid_entries.json 생성
   │    (기존 image_loader.py의 filter_corrupted() 재활용)
   └─ 손상 이미지 목록 기록

3. 체크포인트 로드 (메인 에이전트)
   ├─ Read: checkpoint.json 확인
   └─ 이미 처리된 파일 스킵

4. 배치 분할 + 서브에이전트 실행 (메인 에이전트)
   ├─ valid_entries를 서브에이전트당 3~5건씩 분할
   ├─ Task(subagent_type="general-purpose", run_in_background=true) × N
   ├─ 각 서브에이전트: Read로 이미지 읽고, 평가 후 JSON Write
   └─ 배치 완료 시마다 체크포인트 저장

5. 결과 수집/병합 (메인 에이전트)
   ├─ Read: 각 서브에이전트 output 파일 읽기
   ├─ JSON 파싱 및 검증
   └─ 전체 결과 병합

6. 리포트 생성 (메인 에이전트)
   ├─ Bash: python generate_reports.py → results.json, results.csv, summary.html
   │    (기존 generator.py의 generate_all_reports() 재활용)
   └─ 불일치/손상 리포트 생성
```

---

## 3. 컴포넌트 상세 설계

### 3.1 메인 에이전트 (오케스트레이터)

메인 에이전트는 Claude Code의 최상위 에이전트로, 전체 파이프라인을 조율한다.

#### 3.1.1 Step 1: 데이터셋 스캔

기존 `image_loader.py`의 `scan_dataset()` 함수를 Python 스크립트로 래핑하여 실행한다.

**실행할 스크립트: `dataset_validator/scripts/scan_dataset.py`**

```python
"""Scan dataset directories and output matched/mismatched files as JSON."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dataset_validator.core.image_loader import scan_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    matched, mismatched = scan_dataset(
        Path(args.input_dir), Path(args.reference_dir), Path(args.output_dir)
    )

    # Convert Path objects to strings for JSON serialization
    for m in matched:
        for key in ("input", "reference", "output"):
            m[key] = str(m[key])

    result = {"matched": matched, "mismatched": mismatched}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Matched: {len(matched)}, Mismatched: {len(mismatched)}")

if __name__ == "__main__":
    main()
```

**메인 에이전트 실행:**
```
Bash: python dataset_validator/scripts/scan_dataset.py \
  --input-dir <input_dir> \
  --reference-dir <reference_dir> \
  --output-dir <output_dir> \
  --out ./workspace/scan_result.json
```

#### 3.1.2 Step 2: 이미지 무결성 검증

기존 `image_loader.py`의 `filter_corrupted()` 함수를 래핑하여 실행한다.

**실행할 스크립트: `dataset_validator/scripts/validate_images.py`**

```python
"""Validate image integrity and output valid/corrupted entries as JSON."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dataset_validator.core.image_loader import filter_corrupted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-result", required=True, help="scan_dataset.py output JSON")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.scan_result, "r", encoding="utf-8") as f:
        scan = json.load(f)

    # Restore Path objects
    matched = scan["matched"]
    for m in matched:
        for key in ("input", "reference", "output"):
            m[key] = Path(m[key])

    valid, corrupted = filter_corrupted(matched)

    # Convert back to strings for JSON
    for v in valid:
        for key in ("input", "reference", "output"):
            v[key] = str(v[key])

    result = {"valid": valid, "corrupted": corrupted}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Valid: {len(valid)}, Corrupted: {len(corrupted)}")

if __name__ == "__main__":
    main()
```

#### 3.1.3 Step 3: 체크포인트 로드

메인 에이전트가 Read 도구로 체크포인트 파일을 직접 읽는다.

**체크포인트 파일 형식:** (기존 `checkpoint.py`의 `CheckpointManager` 패턴 참고)

```json
{
  "processed_files": ["001.jpg", "002.jpg"],
  "processed_count": 2,
  "total_count": 10420,
  "results": [
    {
      "filename": "001.jpg",
      "scores": {"hair_similarity_overall": 8, ...},
      "reason": "...",
      "error": false
    }
  ],
  "engine": "claude-opus-4-6",
  "timestamp": "2026-02-08T15:00:00"
}
```

**체크포인트 파일 위치:** `<checkpoint_dir>/subagent_checkpoint.json`

메인 에이전트는:
1. Read로 체크포인트 파일 존재 여부 확인
2. 존재하면 `processed_files` 목록을 읽어 처리 완료된 stem 필터링
3. 미처리 항목만 서브에이전트에 할당

#### 3.1.4 Step 4: 배치 분할 + 서브에이전트 병렬 실행

메인 에이전트가 미처리 트리플릿을 배치로 분할하고, 각 배치를 서브에이전트에 할당한다.

**배치 분할 로직:**
- 서브에이전트 동시 실행 수: 기본 3 (최대 5)
- 서브에이전트당 트리플릿 수: 3~5건 (이미지 9~15장/서브에이전트)
- 라운드: 모든 미처리 항목을 소진할 때까지 반복

**서브에이전트 호출 예시:**
```
Task(
  subagent_type="general-purpose",
  run_in_background=true,
  prompt="[서브에이전트 프롬프트 - 섹션 4 참조]"
)
```

**라운드 실행 흐름:**
```
라운드 1:
  ├─ 서브에이전트 A: 트리플릿 [001, 002, 003] → batch_result_001.json
  ├─ 서브에이전트 B: 트리플릿 [004, 005, 006] → batch_result_002.json
  └─ 서브에이전트 C: 트리플릿 [007, 008, 009] → batch_result_003.json
  → 완료 대기 → 결과 수집 → 체크포인트 저장

라운드 2:
  ├─ 서브에이전트 D: 트리플릿 [010, 011, 012] → batch_result_004.json
  ...
```

#### 3.1.5 Step 5: 결과 수집/병합

각 라운드 완료 후:
1. `TaskOutput` 또는 `Read`로 서브에이전트 출력 파일 읽기
2. JSON 파싱 및 `_validate_scores()` 로직으로 검증 (기존 `evaluator.py` 참조)
3. 전체 결과 배열에 병합
4. 체크포인트 파일 업데이트 (Write)

#### 3.1.6 Step 6: 체크포인트 저장

매 라운드(서브에이전트 배치) 완료 시:
1. 메인 에이전트가 Write로 체크포인트 JSON 저장
2. 누적된 `processed_files`, `results` 포함
3. 리줌 시 이 파일을 읽어 이미 처리된 항목 스킵

#### 3.1.7 Step 7: 리포트 생성

기존 `generator.py`의 `generate_all_reports()` 함수를 래핑하여 실행한다.

**실행할 스크립트: `dataset_validator/scripts/generate_reports.py`**

```python
"""Generate reports from evaluation results JSON."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dataset_validator.report.generator import generate_all_reports

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Checkpoint JSON with results")
    parser.add_argument("--scan-result", required=True, help="scan_dataset.py output JSON")
    parser.add_argument("--report-dir", default="./reports")
    parser.add_argument("--threshold", type=int, default=7)
    parser.add_argument("--threshold-hair", type=int, default=None)
    parser.add_argument("--threshold-preservation", type=int, default=None)
    parser.add_argument("--threshold-naturalness", type=int, default=None)
    args = parser.parse_args()

    # Load results
    with open(args.results, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    results = checkpoint["results"]

    # Load scan result for entries_map (image paths for HTML thumbnails)
    with open(args.scan_result, "r", encoding="utf-8") as f:
        scan = json.load(f)

    entries_map = {}
    for entry in scan["matched"]:
        entries_map[entry["stem"]] = {
            "stem": entry["stem"],
            "input": Path(entry["input"]),
            "reference": Path(entry["reference"]),
            "output": Path(entry["output"]),
        }

    metadata = {
        "engine": checkpoint.get("engine", "claude-opus-4-6"),
        "timestamp": checkpoint.get("timestamp", ""),
        "elapsed_time_sec": checkpoint.get("elapsed_time_sec", 0),
    }

    report_paths = generate_all_reports(
        results=results,
        metadata=metadata,
        entries_map=entries_map,
        report_dir=Path(args.report_dir),
        threshold=args.threshold,
        threshold_hair=args.threshold_hair,
        threshold_preservation=args.threshold_preservation,
        threshold_naturalness=args.threshold_naturalness,
    )

    print(f"Reports generated: {args.report_dir}")
    for fmt, path in report_paths.items():
        if fmt != "metadata":
            print(f"  {fmt}: {path}")

if __name__ == "__main__":
    main()
```

### 3.2 서브에이전트 (general-purpose)

서브에이전트는 `Task(subagent_type="general-purpose")`로 실행되며, Opus 4.6의 멀티모달 비전 기능을 활용하여 이미지를 직접 읽고 평가한다.

#### 3.2.1 서브에이전트 라이프사이클

```
1. 프롬프트 수신: 트리플릿 목록 + 평가 기준 + 출력 파일 경로
2. 각 트리플릿 처리:
   a. Read 도구로 input 이미지 읽기 (멀티모달 — Opus 4.6이 이미지 내용을 시각적으로 인식)
   b. Read 도구로 reference 이미지 읽기
   c. Read 도구로 output 이미지 읽기
   d. 3장 이미지를 시각적으로 비교 분석
   e. 평가 기준에 따라 6개 항목 점수 + 사유 생성
3. 결과 JSON 배열을 Write 도구로 지정된 파일에 저장
```

#### 3.2.2 서브에이전트에서 Read로 이미지 읽기

Claude Code의 Read 도구는 이미지 파일(PNG, JPG 등)을 읽을 수 있으며, Opus 4.6은 멀티모달 LLM이므로 이미지 내용을 시각적으로 인식한다. 별도의 리사이즈나 전처리 없이 원본 이미지를 직접 Read한다.

```
Read(file_path="D:/data/input/001.jpg")      → Opus 4.6이 이미지 시각적 인식
Read(file_path="D:/data/reference/001.jpg")   → Opus 4.6이 이미지 시각적 인식
Read(file_path="D:/data/output/001.jpg")      → Opus 4.6이 이미지 시각적 인식
```

> **참고**: Read 도구 문서에 명시: "This tool allows Claude Code to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as Claude Code is a multimodal LLM."

---

## 4. 서브에이전트 프롬프트 설계

### 4.1 프롬프트 템플릿

서브에이전트에 전달되는 프롬프트는 기존 `evaluator.py`의 `EVALUATION_PROMPT` (24~64행)를 완전히 포함하며, Read 도구 사용 지시와 배치 처리 형식을 추가한다.

```
당신은 Hair Transfer(헤어 편집) 모델의 학습 데이터셋 품질을 평가하는 전문 검사원입니다.

## 작업
아래 트리플릿 목록의 각 항목에 대해:
1. Read 도구로 input, reference, output 이미지 3장을 순서대로 읽으세요.
2. 3장의 이미지를 시각적으로 비교 분석하세요.
3. 아래 평가 기준에 따라 점수를 매기세요.

## 트리플릿 목록
{triplets_json}

## 평가 기준

You are an expert image quality assessor for hair transfer (hair editing) models.

You are given three images:
1. INPUT: The original photo of a person before hair editing
2. REFERENCE: A photo showing the target hairstyle to be applied
3. OUTPUT: The result image after applying the REFERENCE hairstyle onto the INPUT person

### Key Rules
- The hairstyle in OUTPUT should replicate the REFERENCE hairstyle (color, length, texture/curl pattern, shape, bangs).
- However, the hair ANGLE and POSE in OUTPUT must match the INPUT person's head angle and pose,
  NOT the REFERENCE photo's angle. The REFERENCE may show a completely different person from a
  different angle — only the hairstyle itself (style, color, length, texture, shape, bangs) matters.
- Everything EXCEPT the hair region in OUTPUT must remain identical to INPUT
  (face, skin, eyes, clothing, background, accessories, etc.).
- The face shape and face color must be PIXEL-LEVEL identical between INPUT and OUTPUT.
- The hair-to-face boundary in OUTPUT should look natural with no visible seams, artifacts,
  color bleeding, or unnatural lighting transitions.

### Evaluation Criteria
Evaluate the OUTPUT image and respond ONLY with a valid JSON object (no markdown, no extra text):
{
  "hair_similarity_overall": <0-10>,
  "hair_color": <0-10>,
  "hair_length": <0-10>,
  "hair_texture": <0-10>,
  "hair_shape": <0-10>,
  "bangs_shape": <0-10>,
  "bangs_length": <0-10>,
  "non_hair_preservation": <0-10>,
  "naturalness": <0-10>,
  "face_shape_preservation": <0-10>,
  "face_color_preservation": <0-10>,
  "reason": "<1-2 sentence explanation of key observations>"
}

### Field Descriptions (11 criteria)
1. hair_similarity_overall: How well does the OUTPUT hairstyle match the REFERENCE hairstyle overall?
2. hair_color: Does the hair color in OUTPUT match the REFERENCE? (highlights, roots, gradient, tone)
3. hair_length: Does the hair length in OUTPUT match the REFERENCE?
4. hair_texture: Does the hair texture in OUTPUT match the REFERENCE? (straight/wavy/curly/permed)
5. hair_shape: Does the overall hair shape/silhouette match REFERENCE? (volume, layering, parting, flow direction)
6. bangs_shape: Does OUTPUT's bangs shape match REFERENCE? (straight-across, side-swept, curtain, wispy, blunt, layered, no bangs)
7. bangs_length: Does OUTPUT's bangs length match REFERENCE? (above eyebrows, eye-level, cheek-length, no bangs)
8. non_hair_preservation: Are all non-hair regions (face, body, background) in OUTPUT identical to INPUT?
9. naturalness: Does the hair edit look natural? (no artifacts, seamless blending, consistent lighting)
10. face_shape_preservation: Is face shape EXACTLY identical to INPUT? (jawline, chin, cheekbones, forehead, face outline — pixel-level, ANY distortion = LOW)
11. face_color_preservation: Is face/skin color EXACTLY identical to INPUT? (skin tone, brightness, contrast, shadows, lip color — ANY color shift = LOW)

### Scoring Guide
- 0-3: Major failure — wrong hairstyle, face changed, severe artifacts
- 4-5: Poor — clearly visible problems, obvious mismatch
- 6: Acceptable but flawed
- 7-8: Good — minor issues only, overall convincing result
- 9-10: Near perfect — rare, reserve for truly excellent results

## 출력 형식

모든 트리플릿 평가가 끝나면, 아래 형식의 JSON 배열을 Write 도구로 `{output_file_path}` 파일에 저장하세요.

```json
[
  {
    "filename": "001.jpg",
    "scores": {
      "hair_similarity_overall": 8,
      "hair_color": 9,
      "hair_length": 8,
      "hair_texture": 7,
      "hair_shape": 7,
      "bangs_shape": 8,
      "bangs_length": 8,
      "non_hair_preservation": 9,
      "naturalness": 8,
      "face_shape_preservation": 8,
      "face_color_preservation": 8
    },
    "reason": "Hairstyle matches reference well. Color and length are accurate...",
    "error": false
  },
  ...
]
```

### 오류 처리
- 이미지를 Read할 수 없는 경우: `"error": true`, `"scores": null`, `"reason": "Failed to read image: <path>"`
- 평가가 불확실한 경우에도 최선의 점수를 매기되, reason에 불확실한 이유를 기재하세요.

**중요**: JSON만 파일에 Write하세요. 마크다운이나 다른 텍스트를 포함하지 마세요.
```

### 4.2 트리플릿 목록 형식 (`{triplets_json}`)

```json
[
  {
    "stem": "001",
    "input": "D:/data/input/001.jpg",
    "reference": "D:/data/reference/001.jpg",
    "output": "D:/data/output/001.jpg"
  },
  {
    "stem": "002",
    "input": "D:/data/input/002.jpg",
    "reference": "D:/data/reference/002.jpg",
    "output": "D:/data/output/002.jpg"
  }
]
```

### 4.3 프롬프트 구성 시 주의사항

1. **이미지 경로는 절대 경로**로 전달 (Read 도구가 절대 경로를 요구)
2. **output 파일 경로도 절대 경로**로 전달 (Write 도구가 절대 경로를 요구)
3. 평가 기준 텍스트는 기존 `EVALUATION_PROMPT`의 내용과 **100% 동일**해야 일관성 보장
4. 서브에이전트당 트리플릿 수를 3~5건으로 제한 (이미지 Read가 컨텍스트를 소모하므로)

---

## 5. 평가 항목 (11개 기준)

### 5.1 평가 필드

| # | 필드명 | 점수 범위 | 카테고리 | 설명 |
|---|--------|----------|----------|------|
| 1 | `hair_similarity_overall` | 0~10 | hair | reference 대비 output 헤어스타일 전체적 유사도 |
| 2 | `hair_color` | 0~10 | hair | 헤어 색상 일치도 (하이라이트, 뿌리, 그라데이션, 톤) |
| 3 | `hair_length` | 0~10 | hair | 헤어 길이 일치도 |
| 4 | `hair_texture` | 0~10 | hair | 헤어 질감 일치도 (웨이브/컬/스트레이트/파마 등) |
| 5 | `hair_shape` | 0~10 | hair | 헤어 전체 쉐입/실루엣 일치도 (볼륨, 레이어링, 가르마, 흐름 방향) |
| 6 | `bangs_shape` | 0~10 | hair | 앞머리 모양 일치도 (일자, 사이드, 커튼뱅, 시스루, 뭉뚝, 레이어드, 없음) |
| 7 | `bangs_length` | 0~10 | hair | 앞머리 길이 일치도 (눈썹 위/눈썹/눈 덮는/볼 길이/없음) |
| 8 | `non_hair_preservation` | 0~10 | preservation | 비변경 영역(얼굴, 의상, 배경 등) 보존도 |
| 9 | `naturalness` | 0~10 | naturalness | 합성 자연스러움 (경계, 아티팩트, 조명 일관성) |
| 10 | `face_shape_preservation` | 0~10 | face | 얼굴 형태 보존 (턱선, 턱 모양, 광대뼈, 이마, 얼굴 윤곽, 귀 — 픽셀 수준 동일) |
| 11 | `face_color_preservation` | 0~10 | face | 얼굴 색상 보존 (피부톤, 밝기, 대비, 그림자, 입술색, 안색 — 구분 불가 수준) |
| - | `reason` | string | - | 판단 근거 텍스트 (1-2문장) |

### 5.2 합격/불합격 판정 로직

`evaluator.py`의 `is_pass()` 로직. 4개 카테고리별 임계값 적용:

```python
def is_pass(scores, threshold=7, threshold_hair=None,
            threshold_preservation=None, threshold_naturalness=None,
            threshold_face=None):
    hair_thresh = threshold_hair if threshold_hair is not None else threshold
    pres_thresh = threshold_preservation if threshold_preservation is not None else threshold
    nat_thresh = threshold_naturalness if threshold_naturalness is not None else threshold
    face_thresh = threshold_face if threshold_face is not None else threshold

    hair_fields = [
        "hair_similarity_overall", "hair_color", "hair_length", "hair_texture",
        "hair_shape", "bangs_shape", "bangs_length",
    ]
    for field in hair_fields:
        if scores.get(field, 0) < hair_thresh:
            return False
    if scores.get("non_hair_preservation", 0) < pres_thresh:
        return False
    if scores.get("naturalness", 0) < nat_thresh:
        return False
    face_fields = ["face_shape_preservation", "face_color_preservation"]
    for field in face_fields:
        if scores.get(field, 0) < face_thresh:
            return False
    return True
```

### 5.3 임계값 설정

| 파라미터 | 기본값 | 적용 필드 |
|----------|--------|-----------|
| `threshold` | 7 | 전체 항목 공통 기본값 |
| `threshold_hair` | None (threshold) | hair 관련 7개 항목 |
| `threshold_preservation` | None (threshold) | non_hair_preservation |
| `threshold_naturalness` | None (threshold) | naturalness |
| `threshold_face` | None (threshold) | face_shape_preservation, face_color_preservation |

---

## 6. 병렬 처리 설계

### 6.1 동시성 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| 서브에이전트 동시 실행 수 | 3 | 1~5 | Task(run_in_background=true)로 동시 실행 |
| 서브에이전트당 트리플릿 수 | 3 | 3~5 | 이미지 9~15장/서브에이전트 |
| 라운드당 처리 건수 | 9 | 3~25 | 동시 실행 수 × 트리플릿 수 |

### 6.2 라운드 기반 실행 흐름

```
미처리 항목: [001, 002, ..., 100]

라운드 1 (서브에이전트 3개 × 3건):
  ├─ Task A (background): [001, 002, 003] → workspace/batch_001.json
  ├─ Task B (background): [004, 005, 006] → workspace/batch_002.json
  └─ Task C (background): [007, 008, 009] → workspace/batch_003.json

  메인: TaskOutput 대기 → 결과 수집 → 체크포인트 저장

라운드 2:
  ├─ Task D: [010, 011, 012] → workspace/batch_004.json
  ...

라운드 N: 마지막 남은 항목 처리

→ 전체 결과 병합 → 리포트 생성
```

### 6.3 서브에이전트 완료 대기

메인 에이전트는 `TaskOutput(task_id=..., block=true)` 또는 `Read`로 output 파일을 확인하여 서브에이전트 완료를 감지한다.

### 6.4 예상 처리 시간

```
서브에이전트 1건 (트리플릿 3건, 이미지 9장): ~30-60초
라운드 1회 (서브에이전트 3개 병렬): ~30-60초 (병렬이므로 1건과 동일)
라운드당 처리 건수: 9건

100건 처리: ~12 라운드 × 60초 = ~12분
1,000건 처리: ~112 라운드 × 60초 = ~112분 (~1.9시간)
10,000건 처리: ~1,112 라운드 × 60초 = ~18.5시간
```

> **참고**: 서브에이전트당 트리플릿 수와 동시 실행 수를 늘리면 처리 시간을 단축할 수 있으나, Claude Code API 호출 제한에 주의해야 한다.

---

## 7. 체크포인트/리줌

### 7.1 체크포인트 파일 구조

기존 `checkpoint.py`의 `CheckpointManager` 패턴을 참고하되, 서브에이전트 환경에 맞게 단일 JSON 파일로 관리한다.

```json
{
  "processed_files": ["001.jpg", "002.jpg", "003.jpg"],
  "processed_count": 3,
  "total_count": 10420,
  "results": [
    {
      "filename": "001.jpg",
      "scores": {
        "hair_similarity_overall": 8,
        "hair_color": 9,
        "hair_length": 8,
        "hair_texture": 7,
        "hair_shape": 7,
        "bangs_shape": 8,
        "bangs_length": 8,
        "non_hair_preservation": 9,
        "naturalness": 8,
        "face_shape_preservation": 8,
        "face_color_preservation": 8
      },
      "reason": "Hairstyle matches reference well...",
      "error": false
    }
  ],
  "engine": "claude-opus-4-6",
  "timestamp": "2026-02-08T15:30:00",
  "config": {
    "threshold": 7,
    "threshold_hair": null,
    "threshold_preservation": null,
    "threshold_naturalness": null,
    "threshold_face": null,
    "concurrent_agents": 3,
    "triplets_per_agent": 3
  }
}
```

### 7.2 체크포인트 타이밍

- 매 라운드(서브에이전트 배치) 완료 후 저장
- 서브에이전트 실패 시에도 성공한 결과까지는 저장

### 7.3 리줌 로직

1. 메인 에이전트가 체크포인트 파일을 Read
2. `processed_files`에 포함된 stem 목록 추출
3. 전체 valid_entries에서 이미 처리된 stem 필터링
4. 남은 항목만 서브에이전트에 할당
5. 결과는 기존 체크포인트의 `results`에 추가(append)

---

## 8. 에러 처리

### 8.1 서브에이전트 실패

| 실패 유형 | 처리 방법 |
|-----------|-----------|
| 서브에이전트 자체 실패 (Task 에러) | 해당 배치의 트리플릿을 다음 라운드에서 재시도 (최대 2회) |
| JSON 파일 Write 실패 | 해당 배치 전체 재시도 |
| 부분 실패 (일부 트리플릿만 결과 없음) | 성공 결과는 저장, 실패 항목만 재시도 |

### 8.2 JSON 파싱 실패

서브에이전트가 Write한 JSON 파일이 파싱 불가능한 경우:
1. 해당 배치를 더 작은 단위(트리플릿 1~2건)로 분할하여 재시도
2. 최대 2회 재시도 후에도 실패하면 해당 트리플릿을 `error: true`로 마킹

### 8.3 이미지 Read 실패

서브에이전트가 이미지를 Read할 수 없는 경우:
- 서브에이전트가 해당 트리플릿을 `"error": true`로 기록하고 다음 트리플릿 처리 계속
- 메인 에이전트는 error 항목을 리포트에 포함

### 8.4 재시도 추적

```json
{
  "retry_log": [
    {
      "stem": "053",
      "attempts": 2,
      "last_error": "JSON parse failed",
      "final_status": "error"
    }
  ]
}
```

---

## 9. 리포트 (기존 PRD 100% 유지)

리포트 생성은 기존 `report/generator.py`의 `generate_all_reports()` 함수를 그대로 재활용한다. 서브에이전트 평가 결과의 JSON 구조가 기존 VLM 결과와 동일한 형식이므로 호환된다.

### 9.1 JSON 상세 리포트 (`results.json`)

기존 `generate_json_report()` 재활용. 출력 구조:

```json
{
  "metadata": {
    "engine": "claude-opus-4-6",
    "threshold": 7,
    "total_samples": 10420,
    "passed": 9100,
    "failed": 1320,
    "pass_rate": 87.3,
    "timestamp": "2026-02-08T18:00:00",
    "elapsed_time_sec": 67200
  },
  "statistics": {
    "hair_similarity_overall": {"mean": 7.8, "std": 1.2, "min": 2, "max": 10},
    "hair_color": {"mean": 7.5, ...},
    "hair_length": {"mean": 7.2, ...},
    "hair_texture": {"mean": 7.0, ...},
    "hair_shape": {"mean": 7.1, ...},
    "bangs_shape": {"mean": 7.3, ...},
    "bangs_length": {"mean": 7.2, ...},
    "non_hair_preservation": {"mean": 8.5, ...},
    "naturalness": {"mean": 7.8, ...},
    "face_shape_preservation": {"mean": 8.2, ...},
    "face_color_preservation": {"mean": 8.0, ...},
    ...
  },
  "results": [
    {
      "filename": "001.jpg",
      "pass": true,
      "scores": { ... },
      "reason": "..."
    },
    ...
  ]
}
```

**기존과의 차이점:**
- `metadata.model` → `metadata.engine` (`"claude-opus-4-6"`)
- `metadata.quantization` 필드 불필요 (삭제 가능)
- `metadata.num_gpus` → `metadata.concurrent_agents`
- 나머지 구조는 100% 동일

### 9.2 CSV 요약 리포트 (`results.csv`)

기존 `generate_csv_report()` 재활용. 구조 변경 없음:

```csv
filename,pass,hair_similarity_overall,hair_color,hair_length,hair_texture,hair_shape,bangs_shape,bangs_length,non_hair_preservation,naturalness,face_shape_preservation,face_color_preservation,reason
001.jpg,true,8,9,8,7,7,8,8,9,8,8,8,"Hairstyle matches reference well..."
```

### 9.3 HTML 대시보드 (`summary.html`)

기존 `summary.html` Jinja2 템플릿 + Chart.js 재활용. 포함 내용:

1. **대시보드 요약**
   - 전체 Pass/Fail 비율 (파이 차트)
   - 항목별 평균 점수 (바 차트)
   - 항목별 점수 분포 (히스토그램)

2. **불합격 샘플 갤러리**
   - input | reference | output 3장 가로 배치 (base64 인라인 이미지)
   - 항목별 점수 표시
   - 판단 사유 텍스트
   - 점수 낮은 순 정렬

3. **합격 샘플 갤러리** (상위 N개)
   - 고득점 합격 샘플 예시

4. **필터링/정렬 기능**
   - 항목별 점수 필터
   - Pass/Fail 필터
   - 파일명 검색

### 9.4 불일치 리포트 (`mismatch_report.json`)

```json
{
  "mismatched_files": [
    {"stem": "bad1", "missing_in": ["output"], "present_in": ["input", "reference"]}
  ],
  "corrupted_files": [
    {"stem": "bad2", "corrupted_in": ["reference"]}
  ]
}
```

---

## 10. 불합격 데이터 분리

기존 `separate_failed.py`를 **그대로 재활용**한다. 서브에이전트 시스템이 생성하는 `results.json`의 형식이 기존과 동일하므로 호환된다.

```bash
python dataset_validator/separate_failed.py \
  --report ./reports/results.json \
  --input-dir ./data/input \
  --reference-dir ./data/reference \
  --output-dir ./data/output \
  --failed-dir ./data/failed
```

출력 구조:
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

## 11. 파일 구조

### 11.1 새로 생성할 파일

```
dataset_validator/
├── scripts/                          # 서브에이전트 시스템용 스크립트
│   ├── scan_dataset.py               # 데이터셋 스캔 (image_loader.py 래핑)
│   ├── validate_images.py            # 이미지 무결성 검증 (image_loader.py 래핑)
│   └── generate_reports.py           # 리포트 생성 (generator.py 래핑)
└── (기존 파일 수정 없음)
```

### 11.2 기존 파일 재활용 (수정 없음)

| 파일 | 재활용 부분 |
|------|-------------|
| `core/image_loader.py` | `scan_dataset()`, `filter_corrupted()`, `validate_image()` |
| `core/evaluator.py` | `EVALUATION_PROMPT`, `SCORE_FIELDS`, `is_pass()`, `_parse_json_response()`, `_validate_scores()` |
| `core/checkpoint.py` | `CheckpointManager` 패턴 참고 (직접 사용하지는 않음 — 메인 에이전트가 Write/Read로 직접 관리) |
| `report/generator.py` | `generate_all_reports()`, `_compute_statistics()`, `_compute_histogram()` |
| `report/templates/summary.html` | HTML 대시보드 템플릿 전체 |
| `separate_failed.py` | 불합격 분리 스크립트 전체 |

### 11.3 워크스페이스 디렉토리 (런타임 생성)

```
workspace/                            # 메인 에이전트가 런타임에 생성
├── scan_result.json                  # scan_dataset.py 출력
├── valid_entries.json                # validate_images.py 출력
├── batch_001.json                    # 서브에이전트 출력 (라운드별)
├── batch_002.json
├── ...
└── checkpoint.json                   # 체크포인트 파일
```

---

## 12. 사용 방법

### 12.1 메인 에이전트에 전달할 지시

Claude Code 메인 에이전트에게 아래와 같이 지시한다:

```
Hair Transfer 데이터셋 품질 검사를 실행해주세요.

데이터셋 경로:
- input: D:/data/hair_transfer/input
- reference: D:/data/hair_transfer/reference
- output: D:/data/hair_transfer/output

설정:
- 합격 임계값: 7 (전체 항목)
- 서브에이전트 동시 실행: 3
- 서브에이전트당 트리플릿: 3
- 리포트 출력: ./reports
- 체크포인트: ./workspace/checkpoint.json
- 리줌: false (처음부터 시작)

스펙 문서: .claude/docs/dataset_subagent_validator_spec.md
```

### 12.2 리줌 시

```
이전 체크포인트에서 이어서 실행해주세요.
체크포인트: ./workspace/checkpoint.json
```

---

## 13. 기존 시스템과의 호환성

### 13.1 리포트 포맷 호환

| 항목 | 기존 VLM 시스템 | 서브에이전트 시스템 | 호환 여부 |
|------|----------------|-------------------|-----------|
| `results.json` 구조 | ✓ | ✓ (동일) | 호환 |
| `results.csv` 구조 | ✓ | ✓ (동일) | 호환 |
| `summary.html` 렌더링 | ✓ | ✓ (동일 템플릿) | 호환 |
| `separate_failed.py` 입력 | ✓ | ✓ (동일 JSON) | 호환 |

### 13.2 평가 기준 호환

- 11개 점수 필드: 동일 (`SCORE_FIELDS`) — hair 7개 + preservation 1개 + naturalness 1개 + face 2개
- 프롬프트 텍스트: 동일 (`EVALUATION_PROMPT`)
- 합격 로직: 동일 (`is_pass()`) — 4개 카테고리별 임계값 지원
- 점수 범위: 동일 (0~10)

### 13.3 주요 차이점

| 항목 | 기존 | 서브에이전트 |
|------|------|------------|
| `metadata.model` | `"qwen2.5-vl-7b"` | `"claude-opus-4-6"` (필드명: `engine`) |
| `metadata.quantization` | `"int4"` | 없음 |
| `metadata.num_gpus` | `1~4` | 없음 (`concurrent_agents` 대체) |
| 처리 속도 | 트리플릿당 ~2-3초 | 트리플릿당 ~10-15초 |
| GPU 필요 | 필수 | 불필요 |
| 평가 정확도 | 모델 의존 (7B~12B) | Opus 4.6 (최고 수준) |

---

## 14. 리스크 및 제약사항

| 리스크 | 영향 | 대응 방안 |
|--------|------|-----------|
| Claude Code API 호출 제한 | 대량 처리 시 rate limit | 서브에이전트 동시 실행 수 조절, 라운드 간 대기 |
| 서브에이전트 컨텍스트 한계 | 이미지가 컨텍스트를 많이 소모 | 서브에이전트당 3~5건으로 제한 |
| 처리 속도 | GPU VLM 대비 느림 | 병렬 서브에이전트로 보완, 체크포인트로 중단 후 재개 |
| 평가 일관성 | LLM의 확률적 특성 | 동일 프롬프트 사용, 명확한 채점 기준 제공 |
| 비용 | API 호출 비용 | 트리플릿당 이미지 3장 × 비전 토큰 소모 |
| 네트워크 의존성 | 오프라인 사용 불가 | GPU VLM 시스템을 오프라인 백업으로 유지 |
| JSON 파싱 실패 | 서브에이전트가 잘못된 형식 출력 | 재시도 로직, 배치 분할 후 재시도 |

---

## 15. 향후 확장 가능성

- **평가 기준 커스터마이징**: 프롬프트 텍스트 수정만으로 평가 항목 추가/변경 가능
- **2-Stage 검증**: 서브에이전트(Opus 4.6) 1차 → GPU VLM 2차 (또는 반대)
- **Haiku 4.5 경량 평가**: 비용 절감을 위해 `model="haiku"` 옵션으로 서브에이전트 실행
- **자동화 스크립트**: 메인 에이전트의 오케스트레이션 로직을 Python 스크립트로 구현하여 완전 자동화
- **Batch API 연동**: Anthropic Batch API를 사용하여 대량 처리 비용 절감
