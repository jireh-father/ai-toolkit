# PRD: Face Pixel Comparator — 얼굴 영역 픽셀 유사도 검증 스크립트

## 1. 개요

### 1.1 목적
Hair Transfer 모델의 output 이미지에서 **얼굴 영역이 input과 동일하게 보존**되었는지를, Face Segmentation 기반 **픽셀 단위 비교**로 정량 검증하는 독립 스크립트를 개발한다.

### 1.2 배경
- VLM 기반 검증(`validate_dataset.py`)은 주관적 점수이므로, 얼굴 보존 여부를 **객관적 픽셀 메트릭**으로 이중 검증할 필요가 있다.
- Hair transfer 시 헤어 영역은 변경이 정상이므로, **헤어를 제외한 얼굴 윤곽 영역만** 비교해야 한다.

### 1.3 핵심 아이디어
```
INPUT 이미지                      OUTPUT 이미지
     │                                  │
     ▼                                  ▼
 FARL Face Parsing              FARL Face Parsing
     │                                  │
     ▼                                  ▼
 Face Mask (skin+눈코입눈썹)     Face Mask (skin+눈코입눈썹)
     │                                  │
     └──────────┬───────────────────────┘
                ▼
         Mask Intersection (두 마스크의 겹치는 영역)
                │
                ▼
   INPUT[intersection] vs OUTPUT[intersection]
                │
                ▼
         MAE / MSE 계산 → Pass/Fail 판정
```

---

## 2. Face Segmentation 모델

### 2.1 선택: FaRL (Face Representation Learning)
- **모델**: `FaRL-Base` (Microsoft)
- **학습 데이터**: CelebAMask-HQ
- **출력 클래스**: 11개 (background, skin, nose, eyes, eyebrows, ears, mouth, lip, neck, cloth, hair)
- **정확도**: CelebAMask-HQ face parsing SOTA급
- **HuggingFace**: `microsoft/FaRL` 계열

### 2.2 사용 레이블 (얼굴 영역 마스크 구성)
다음 클래스를 합쳐서 하나의 **얼굴 마스크**를 생성한다:

| 클래스 | 포함 여부 | 비고 |
|--------|-----------|------|
| skin | O | 얼굴 피부 전체 |
| nose | O | 코 |
| left_eye, right_eye | O | 양쪽 눈 |
| left_eyebrow, right_eyebrow | O | 양쪽 눈썹 |
| upper_lip, lower_lip, mouth | O | 입술 + 입 |
| hair | **X** | 헤어 영역 제외 (변경 정상) |
| ears | **X** | 헤어에 가려질 수 있어 불안정 |
| neck | **X** | 의상과 겹칠 수 있어 제외 |
| cloth | **X** | 비교 대상 아님 |
| background | **X** | 비교 대상 아님 |

### 2.3 얼굴 미검출 시 처리
- input 또는 output에서 얼굴 segmentation 결과가 없으면 (마스크 면적 = 0)
- **스킵 (skip)** 처리 — pass/fail 판정하지 않음
- skipped 목록에 별도 기록
- reason: `"Face not detected in {input|output}"`

---

## 3. 비교 파이프라인

### 3.1 전처리
1. input / output 이미지 로드 (동일 파일명 매칭)
2. **두 이미지 크기가 다르면** output을 input 크기로 리사이즈 (LANCZOS)
3. 두 이미지 모두 FARL로 face parsing 수행

### 3.2 마스크 교집합 계산
```python
# input_mask: input 이미지의 얼굴 영역 (skin+nose+eyes+eyebrows+mouth)
# output_mask: output 이미지의 얼굴 영역 (동일 클래스)
intersection_mask = input_mask & output_mask  # 두 마스크가 겹치는 영역만
```

### 3.3 픽셀 비교
```python
# intersection 영역에서만 픽셀 추출
input_pixels = input_image[intersection_mask]    # shape: (N, 3) RGB
output_pixels = output_image[intersection_mask]   # shape: (N, 3) RGB

# MAE (Mean Absolute Error)
mae = np.mean(np.abs(input_pixels.astype(float) - output_pixels.astype(float)))
# 범위: 0 ~ 255, 낮을수록 유사

# MSE (Mean Squared Error)
mse = np.mean((input_pixels.astype(float) - output_pixels.astype(float)) ** 2)
# 범위: 0 ~ 65025, 낮을수록 유사
```

### 3.4 Pass/Fail 판정
```python
def is_pass(mae: float, threshold_mae: float = 10.0) -> bool:
    return mae <= threshold_mae
```
- **기본 임계값**: MAE > 10 → Fail
- CLI에서 `--threshold-mae` 로 조정 가능

### 3.5 추가 메트릭 (참고용 기록)
- **IoU (Intersection over Union)**: input_mask와 output_mask의 겹침 비율
  - IoU가 매우 낮으면 얼굴 위치 자체가 크게 변한 것
- **마스크 면적 비율**: intersection 픽셀 수 / input_mask 픽셀 수
  - 비율이 낮으면 얼굴 형태가 변형된 것

---

## 4. CLI 인터페이스

### 4.1 스크립트: `compare_faces.py`

```bash
python dataset_validator/compare_faces.py \
  --input-dir E:\backup\aihub2025_gpu\output\input_image \
  --output-dir E:\backup\aihub2025_gpu\output \
  --report-dir ./face_reports \
  --threshold-mae 20.0
```

### 4.2 전체 인자 목록

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--input-dir` | str | **필수** | input 이미지 폴더 |
| `--output-dir` | str | **필수** | output 이미지 폴더 |
| `--report-dir` | str | `./face_reports` | 리포트 출력 디렉토리 |
| `--threshold-mae` | float | `10.0` | MAE 실패 임계값 (0~255) |
| `--max-samples` | int | None | 최대 검사 수 (디버깅용) |
| `--batch-size` | int | `8` | Face parsing 배치 크기 |
| `--device` | str | `cuda:0` | 디바이스 (cuda:0 / cpu) |
| `--log-level` | str | `info` | 로깅 레벨 |

---

## 5. 출력 리포트

### 5.1 JSON 리포트 (`face_results.json`)
```json
{
  "metadata": {
    "threshold_mae": 20.0,
    "total_samples": 7906,
    "passed": 7200,
    "failed": 656,
    "skipped": 50,
    "pass_rate": 91.7,
    "timestamp": "2026-02-08T20:00:00",
    "elapsed_time_sec": 3600
  },
  "statistics": {
    "mae": {"mean": 12.3, "std": 8.1, "min": 1.2, "max": 89.5},
    "mse": {"mean": 340.2, "std": 520.1, "min": 3.0, "max": 15200.0},
    "mask_iou": {"mean": 0.92, "std": 0.05, "min": 0.41, "max": 0.99},
    "mask_area_ratio": {"mean": 0.95, "std": 0.04, "min": 0.50, "max": 1.00}
  },
  "results": [
    {
      "filename": "001.jpg",
      "pass": true,
      "mae": 8.5,
      "mse": 120.3,
      "mask_iou": 0.95,
      "mask_area_ratio": 0.97,
      "input_mask_pixels": 45000,
      "output_mask_pixels": 44500,
      "intersection_pixels": 43800
    },
    {
      "filename": "053.jpg",
      "pass": false,
      "mae": 35.2,
      "mse": 2100.5,
      "mask_iou": 0.78,
      "mask_area_ratio": 0.82,
      "input_mask_pixels": 42000,
      "output_mask_pixels": 38000,
      "intersection_pixels": 34000,
      "reason": "MAE 35.2 exceeds threshold 20.0"
    }
  ]
}
```

### 5.2 CSV 리포트 (`face_results.csv`)
```csv
filename,pass,mae,mse,mask_iou,mask_area_ratio,intersection_pixels
001.jpg,true,8.5,120.3,0.95,0.97,43800
053.jpg,false,35.2,2100.5,0.78,0.82,34000
```

### 5.3 HTML 리포트 (`face_summary.html`)
1. **대시보드**
   - Pass/Fail 비율 파이차트
   - MAE 분포 히스토그램
   - MAE vs Mask IoU 산점도

2. **실패 갤러리**
   - input / output 이미지 나란히
   - 차이맵 (difference heatmap) 이미지: 차이가 큰 픽셀을 빨간색으로 시각화
   - MAE, MSE, IoU 점수 표시
   - MAE 높은 순 정렬

3. **필터링/검색**
   - Pass/Fail 필터
   - 파일명 검색
   - MAE 범위 필터

---

## 6. 파일 구조

```
dataset_validator/
├── compare_faces.py              # 메인 엔트리포인트
├── core/
│   ├── face_segmentor.py         # FARL 로드, face parsing, 마스크 생성
│   └── pixel_comparator.py       # 마스크 교집합, MAE/MSE 계산, 차이맵 생성
└── report/
    ├── face_report_generator.py  # JSON/CSV/HTML 리포트 생성
    └── templates/
        └── face_summary.html     # HTML 템플릿
```

---

## 7. 기술 스택

| 구성 요소 | 기술 | 비고 |
|-----------|------|------|
| Face Parsing | FaRL (microsoft/FaRL) | HuggingFace에서 로드 |
| 이미지 처리 | Pillow, numpy, OpenCV | 리사이즈, 마스크 연산, 차이맵 |
| 리포트 | Jinja2, Chart.js | HTML 대시보드 |
| CLI | argparse | 표준 라이브러리 |
| 진행률 | tqdm | 프로그레스 바 |

---

## 8. 성능 예상

```
FARL face parsing: ~0.05초/장 (GPU)
1쌍(input+output) 처리: ~0.15초 (parsing 2회 + 비교)
7,906쌍 전체: ~20분 (단일 GPU)
배치 처리 시: ~10분
```

---

## 9. 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| FARL이 얼굴을 못 찾는 경우 | 자동 fail 처리, reason에 기록 |
| input/output 해상도가 다른 경우 | output을 input 크기로 리사이즈 후 비교 |
| 조명/색온도 차이로 MAE 높아짐 | threshold 조정 가능, MSE/IoU 함께 참고 |
| 얼굴 각도가 약간 달라 마스크 불일치 | intersection(겹치는 부분)만 비교하여 완화 |
