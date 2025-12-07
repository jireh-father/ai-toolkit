# AI 코딩 에이전트용 프롬프트 모음

이 문서는 AI 코딩 에이전트 학습용 프롬프트와 솔루션을 포함합니다.

---

## 📦 Software Engineering (3개)

---

### 🔧 SE-1: REST API 엔드포인트 리팩토링 및 테스트 커버리지 확보

#### Prompt

```
/app/src/ 디렉토리에 Flask 기반 REST API 서버 코드가 있습니다.
현재 코드에는 다음과 같은 문제가 있습니다:

1. 모든 라우트 핸들러가 app.py 한 파일에 800줄 이상 작성되어 있음
2. 데이터베이스 쿼리 로직이 라우트 핸들러에 직접 포함되어 있음
3. 에러 핸들링이 일관성 없이 구현되어 있음
4. 단위 테스트가 전혀 없음

다음 작업을 수행하세요:

1. app.py를 기능별로 Blueprint 패턴을 사용하여 분리하세요
   - /app/src/routes/users.py (사용자 관련)
   - /app/src/routes/products.py (상품 관련)
   - /app/src/routes/orders.py (주문 관련)

2. 데이터베이스 로직을 /app/src/repositories/ 디렉토리로 분리하세요

3. 공통 에러 핸들러를 /app/src/utils/error_handler.py에 구현하세요

4. /app/tests/ 디렉토리에 각 라우트에 대한 단위 테스트를 작성하세요
   - 최소 70% 이상의 코드 커버리지 달성 필요

성공 기준:
- pytest 실행 시 모든 테스트가 통과해야 함
- pytest --cov 실행 시 커버리지가 70% 이상이어야 함
- 리팩토링 후 기존 API 동작이 동일하게 유지되어야 함
- /app/refactoring_report.md 파일에 변경 사항 요약이 포함되어야 함
```

#### Solution

**Step-by-step Plan:**

1. **현재 코드 분석**: app.py 파일을 읽어 기존 라우트 구조, 데이터베이스 쿼리 패턴, 에러 핸들링 방식을 파악합니다.

2. **디렉토리 구조 생성**: routes/, repositories/, utils/ 디렉토리를 생성합니다.

3. **Blueprint 분리**:
   - app.py에서 `/users` 관련 라우트를 추출하여 routes/users.py로 이동
   - `/products` 관련 라우트를 routes/products.py로 이동
   - `/orders` 관련 라우트를 routes/orders.py로 이동
   - 각 파일에서 Flask Blueprint 객체 생성

4. **Repository 패턴 적용**: 각 라우트에서 직접 호출하던 DB 쿼리를 Repository 클래스로 추상화합니다.

5. **에러 핸들러 구현**: 공통 예외 클래스와 Flask의 @app.errorhandler 데코레이터를 활용한 중앙화된 에러 처리 로직을 구현합니다.

6. **테스트 작성**: pytest와 Flask test client를 사용하여 각 엔드포인트에 대한 테스트 케이스를 작성합니다.

**Expected AI Agent Behavior:**
- 파일 구조를 분석하고 의존성 관계를 파악
- 점진적으로 코드를 분리하면서 각 단계마다 동작 확인
- 테스트 코드 작성 시 edge case 포함

**Verification:**
- `pytest` 실행하여 모든 테스트 통과 확인
- `pytest --cov=app --cov-report=term-missing` 으로 커버리지 70% 이상 확인
- 원본 API와 리팩토링된 API의 응답 비교

---

### 🔧 SE-2: 레거시 동기 코드를 비동기로 마이그레이션

#### Prompt

```
# Task
웹 크롤러 파일인 /app/crawler/sync_crawler.py 파일을 참고해서 비동기 방식의 크롤러를 구현하세요.
현재 구현은 requests 라이브러리를 사용한 동기 방식으로, 100개의 URL을 크롤링하는 데 평균 5분 이상 소요됩니다.
기존 크롤링 소스의 병목 구간을 분석하여 aiohttp와 asyncio로 대체하고 동시 연결 수 제한, 재시도 로그 및 타임아웃 처리를 구현하세요.
구현 완료후 두 버전의 성능을 비교하는 벤치마크 스크립트를 작성하고 실행 결과를 json파일로 저장하여 평가하세요.

# 파일 경로
- 동기 크롤러: /app/crawler/sync_crawler.py
- 비동기 크롤러: /app/crawler/async_crawler.py
- 벤치마크 스크립트: /app/crawler/benchmark.py
- 벤치마크 결과: /app/crawler/benchmark_result.json

# 성공 기준
- 비동기 버전이 동기 버전 대비 최소 2배 이상 빠르게 동작해야 함
- /app/crawler/benchmark_result.json에 성능 비교 결과가 저장되어야 함
  (각 버전의 총 소요시간, 성공/실패 URL 수, 평균 응답시간 포함)
- 비동기 버전에서 모든 에러가 적절히 처리되어야 함 (크래시 없이)
```

#### Solution

# Step-by-step Plan
1. 기존 코드의 병목 구간 분석
2. 비동기 크롤러 설계 및 구현
3. 기타 연결 수 제한, 재시도 로직, 타임아웃 기능 구현
4. 벤치마크 스크립트 작성
5. 밴치마크 실행 및 평과 결과 확인

# Expected AI Agent Behavior
- 동기 코드의 구조를 유지하면서 비동기로 변환
- 에러 처리 로직을 빠뜨리지 않고 구현
- 실제 URL로 테스트하여 동작 검증

# Verification
- benchmark.py 실행하여 benchmark_result.json 생성 확인
- JSON 파일에서 비동기 버전이 동기 버전 대비 최소 2배 이상 빠른지 확인
- 실패한 URL에 대한 에러 로그가 적절히 기록되었는지 확인

---

### 🔧 SE-3: CI/CD 파이프라인 설정 및 배포 자동화

#### Prompt

```
/app/project/ 디렉토리에 Node.js 기반 웹 애플리케이션이 있습니다.
현재 배포는 수동으로 이루어지며, 테스트 자동화가 없습니다.

다음 작업을 수행하세요:

1. GitHub Actions 워크플로우 파일을 생성하세요
   - /app/project/.github/workflows/ci.yml
   - /app/project/.github/workflows/deploy.yml

2. CI 워크플로우에 다음을 포함하세요:
   - Node.js 18.x 환경 설정
   - 의존성 설치 (npm ci)
   - ESLint 실행
   - Jest 단위 테스트 실행
   - 코드 커버리지 리포트 생성

3. Deploy 워크플로우에 다음을 포함하세요:
   - main 브랜치 푸시 시 자동 트리거
   - Docker 이미지 빌드
   - 이미지 태깅 (git SHA 기반)
   - /app/project/Dockerfile 생성 (Node.js 애플리케이션용)

4. /app/project/package.json에 필요한 스크립트를 추가하세요
   - "lint", "test", "test:coverage"

성공 기준:
- 로컬에서 `act` 도구를 사용하여 워크플로우 시뮬레이션 시 에러 없이 통과해야 함
- Dockerfile로 이미지 빌드가 성공해야 함 (docker build . 실행)
- /app/project/CI_SETUP.md에 설정 가이드 문서가 포함되어야 함
```

#### Solution

**Step-by-step Plan:**

1. **프로젝트 구조 분석**: package.json, 소스 코드 구조, 기존 테스트 유무를 확인합니다.

2. **package.json 스크립트 추가**:
   ```json
   "scripts": {
     "lint": "eslint src/",
     "test": "jest",
     "test:coverage": "jest --coverage"
   }
   ```

3. **CI 워크플로우 작성**:
   - `on: [push, pull_request]` 트리거 설정
   - Node.js 환경 설정 (actions/setup-node@v3)
   - 캐싱 설정으로 빌드 시간 최적화
   - 순차적 job 정의 (install → lint → test)

4. **Deploy 워크플로우 작성**:
   - `on: push: branches: [main]` 조건 설정
   - Docker buildx 설정
   - 이미지 태그: `${{ github.sha }}`

5. **Dockerfile 작성**: Multi-stage 빌드로 이미지 크기 최적화합니다.

6. **문서 작성**: CI_SETUP.md에 워크플로우 설명과 로컬 테스트 방법 기록합니다.

**Expected AI Agent Behavior:**
- GitHub Actions 문법에 맞는 YAML 파일 생성
- 보안 모범 사례 적용 (secrets 사용, 최소 권한 원칙)
- Dockerfile 최적화 (레이어 캐싱, 불필요한 파일 제외)

**Verification:**
- `act` 도구로 로컬에서 워크플로우 실행 테스트
- `docker build -t test-app .` 명령 성공 확인
- 생성된 문서의 완성도 확인

---

## 🤖 Machine Learning (3개)

---

### 🧠 ML-1: 모델 성능 분석 및 하이퍼파라미터 튜닝 자동화

#### Prompt

```
/app/ml_project/ 디렉토리에 이미지 분류 프로젝트가 있습니다.
현재 모델(ResNet18 기반)의 검증 정확도가 72%로 목표 대비 낮습니다.

다음 작업을 수행하세요:

1. 현재 모델의 문제점을 분석하세요
   - /app/ml_project/training_logs.csv 분석 (loss, accuracy 추이)
   - 과적합/과소적합 여부 판단
   - /app/ml_project/model_analysis.md에 분석 결과 작성

2. Optuna를 활용한 하이퍼파라미터 튜닝 스크립트를 작성하세요
   - /app/ml_project/hyperparameter_tuning.py
   - 탐색할 파라미터: learning_rate, batch_size, weight_decay, dropout_rate
   - 20회 trial 실행
   - SQLite DB에 결과 저장 (/app/ml_project/optuna_study.db)

3. 최적 하이퍼파라미터로 모델을 재학습하고 저장하세요
   - /app/ml_project/best_model.pth

4. 개선된 모델의 성능을 평가하세요
   - Confusion Matrix 생성
   - Classification Report (Precision, Recall, F1-score)
   - 결과를 /app/ml_project/evaluation_report.json에 저장

성공 기준:
- 최적화된 모델의 검증 정확도가 80% 이상이어야 함
- evaluation_report.json에 모든 클래스별 metrics가 포함되어야 함
- model_analysis.md에 개선 전/후 비교가 포함되어야 함
```

#### Solution

**Step-by-step Plan:**

1. **학습 로그 분석**:
   - training_logs.csv를 pandas로 로드
   - train/val loss 곡선을 비교하여 과적합 여부 판단
   - epoch별 accuracy 추이 확인

2. **Optuna 튜닝 스크립트 작성**:
   ```python
   def objective(trial):
       lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
       batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
       weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
       dropout = trial.suggest_float('dropout', 0.1, 0.5)
       # 모델 학습 및 검증 정확도 반환
   ```

3. **최적 파라미터로 재학습**: study.best_params를 사용하여 최종 모델 학습 및 저장합니다.

4. **평가 수행**:
   - sklearn의 confusion_matrix, classification_report 활용
   - 결과를 JSON 형식으로 직렬화하여 저장

**Expected AI Agent Behavior:**
- 로그 분석 시 시각화 또는 수치적 판단 근거 제시
- Optuna의 pruning 기능 활용하여 효율적 탐색
- 재현 가능한 학습을 위해 random seed 고정

**Verification:**
- optuna_study.db에서 최적 trial 결과 확인
- best_model.pth 로드하여 검증 정확도 80% 이상 확인
- evaluation_report.json 구조 및 값 검증

---

### 🧠 ML-2: 데이터 파이프라인 구축 및 특성 엔지니어링

#### Prompt

```
/app/data/ 디렉토리에 전자상거래 거래 데이터가 있습니다:
- transactions.csv (100만 건의 거래 데이터)
- customers.csv (고객 정보)
- products.csv (상품 정보)

고객 이탈 예측 모델을 위한 데이터 파이프라인을 구축하세요.

다음 작업을 수행하세요:

1. 탐색적 데이터 분석(EDA) 수행
   - 결측치 분석
   - 이상치 탐지
   - 주요 통계량 계산
   - /app/data/eda_report.html 생성 (pandas-profiling 또는 수동 HTML)

2. 특성 엔지니어링 파이프라인 구현
   - /app/pipeline/feature_engineering.py
   - RFM(Recency, Frequency, Monetary) 특성 생성
   - 시계열 기반 특성 (최근 30일/90일 구매 패턴)
   - 고객-상품 상호작용 특성

3. 데이터 전처리 파이프라인 구현
   - /app/pipeline/preprocessing.py
   - 결측치 처리 전략 구현
   - 범주형 변수 인코딩
   - 수치형 변수 스케일링
   - sklearn Pipeline 객체로 구현

4. 파이프라인 직렬화 및 저장
   - /app/pipeline/feature_pipeline.pkl
   - /app/pipeline/preprocessing_pipeline.pkl

성공 기준:
- 파이프라인 로드 후 새로운 데이터에 적용 시 에러 없이 동작해야 함
- 생성된 특성이 최소 20개 이상이어야 함
- /app/pipeline/feature_description.json에 각 특성에 대한 설명이 포함되어야 함
- EDA 리포트가 웹 브라우저에서 정상적으로 렌더링되어야 함
```

#### Solution

**Step-by-step Plan:**

1. **데이터 로드 및 EDA**:
   - 각 CSV 파일을 pandas로 로드
   - 결측치: `df.isnull().sum()`, 비율 계산
   - 이상치: IQR 방법 또는 Z-score 방법 적용
   - ydata-profiling (구 pandas-profiling)으로 HTML 리포트 생성

2. **RFM 특성 생성**:
   ```python
   rfm = transactions.groupby('customer_id').agg({
       'transaction_date': lambda x: (today - x.max()).days,  # Recency
       'transaction_id': 'count',  # Frequency
       'amount': 'sum'  # Monetary
   })
   ```

3. **시계열 특성**:
   - 최근 30일/90일 필터링하여 집계 통계 계산
   - 구매 주기, 평균 구매 금액 변화율 등

4. **전처리 파이프라인**:
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.compose import ColumnTransformer
   
   preprocessor = ColumnTransformer([
       ('num', StandardScaler(), numeric_cols),
       ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
   ])
   ```

5. **직렬화**: joblib 또는 pickle로 파이프라인 저장합니다.

**Expected AI Agent Behavior:**
- 대용량 데이터 처리 시 메모리 효율적인 방법 사용 (chunking)
- 특성 간 상관관계 분석으로 중복 특성 제거
- 파이프라인의 fit/transform 분리 이해

**Verification:**
- 테스트 데이터셋으로 파이프라인 로드 및 적용
- 출력 특성 개수 확인 (20개 이상)
- feature_description.json 검증
- eda_report.html 브라우저에서 열어 확인

---

### 🧠 ML-3: 모델 서빙 API 구현 및 A/B 테스트 인프라

#### Prompt

```
/app/models/ 디렉토리에 두 개의 학습된 감성 분석 모델이 있습니다:
- model_v1.pkl (Logistic Regression, 정확도 85%)
- model_v2.pkl (BERT fine-tuned, 정확도 91%)

두 모델을 동시에 서빙하고 A/B 테스트를 수행할 수 있는 API를 구현하세요.

다음 작업을 수행하세요:

1. FastAPI 기반 추론 서버 구현
   - /app/serving/main.py
   - POST /predict 엔드포인트
   - 요청 본문: {"text": "분석할 텍스트", "model_version": "v1" | "v2" | "ab_test"}
   - model_version이 "ab_test"인 경우 50:50 비율로 무작위 라우팅

2. A/B 테스트 로깅 시스템 구현
   - /app/serving/ab_logger.py
   - 각 요청에 대해 기록: timestamp, request_id, model_version, input_text, prediction, latency
   - /app/logs/ab_test_log.jsonl에 저장 (JSON Lines 형식)

3. A/B 테스트 분석 스크립트 작성
   - /app/serving/analyze_ab.py
   - 각 모델의 평균 응답 시간
   - 요청 분포 통계
   - /app/logs/ab_analysis.json에 결과 저장

4. 헬스체크 및 메트릭 엔드포인트 구현
   - GET /health
   - GET /metrics (총 요청 수, 모델별 요청 수, 평균 latency)

성공 기준:
- 서버 시작 후 /health 응답이 {"status": "healthy"}여야 함
- 100개의 테스트 요청 후 ab_test_log.jsonl에 100개의 로그가 기록되어야 함
- A/B 테스트 시 각 모델이 대략 50% (±10%) 비율로 호출되어야 함
- /app/serving/API_DOCUMENTATION.md에 API 사용법이 문서화되어야 함
```

#### Solution

**Step-by-step Plan:**

1. **FastAPI 서버 구조 설계**:
   ```python
   from fastapi import FastAPI
   import random
   
   app = FastAPI()
   
   @app.post("/predict")
   async def predict(request: PredictRequest):
       if request.model_version == "ab_test":
           version = random.choice(["v1", "v2"])
       else:
           version = request.model_version
       # 해당 모델로 추론 수행
   ```

2. **모델 로딩 전략**:
   - 서버 시작 시 두 모델을 메모리에 로드
   - BERT 모델의 경우 GPU 사용 여부 체크

3. **A/B 로깅 구현**:
   ```python
   class ABLogger:
       def log(self, request_id, model_version, input_text, prediction, latency):
           log_entry = {
               "timestamp": datetime.now().isoformat(),
               "request_id": request_id,
               ...
           }
           with open("ab_test_log.jsonl", "a") as f:
               f.write(json.dumps(log_entry) + "\n")
   ```

4. **분석 스크립트**:
   - JSONL 파일 파싱
   - 모델별 그룹화하여 통계 계산
   - 결과 JSON 저장

5. **문서화**: OpenAPI 스펙 기반 자동 문서 + 수동 사용 예제 추가합니다.

**Expected AI Agent Behavior:**
- 비동기 요청 처리 구현
- 스레드 안전한 로깅 구현
- 모델 로딩 에러 처리
- Pydantic을 활용한 요청/응답 스키마 정의

**Verification:**
- `uvicorn main:app` 으로 서버 시작
- `curl http://localhost:8000/health` 로 헬스체크 확인
- 100개 요청 스크립트 실행 후 로그 파일 확인
- analyze_ab.py 실행하여 A/B 비율 검증

---

## 📋 요약

| 번호 | 카테고리 | 난이도 | 핵심 기술 |
|------|----------|--------|-----------|
| SE-1 | Software Engineering | ⭐⭐⭐ | Flask, Blueprint, pytest, TDD |
| SE-2 | Software Engineering | ⭐⭐⭐⭐ | asyncio, aiohttp, 성능 최적화 |
| SE-3 | Software Engineering | ⭐⭐⭐ | GitHub Actions, Docker, CI/CD |
| ML-1 | Machine Learning | ⭐⭐⭐⭐ | Optuna, PyTorch, 하이퍼파라미터 튜닝 |
| ML-2 | Machine Learning | ⭐⭐⭐⭐ | sklearn Pipeline, 특성 엔지니어링, EDA |
| ML-3 | Machine Learning | ⭐⭐⭐⭐⭐ | FastAPI, 모델 서빙, A/B 테스트 |

---

*이 문서는 AI 코딩 에이전트 학습용 프롬프트 모음입니다.*
