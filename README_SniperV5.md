# Hybrid Sniper V5: 지능형 데이터 자동화 패키지 (Phase 1)

이 문서는 Hybrid Sniper V5 시스템의 핵심인 **데이터 자동 수집 및 관리 엔진**에 대한 설명서입니다.  
Gemini 1.5 Flash AI를 활용하여 웹상의 비정형 데이터를 정형 데이터로 변환하고, 구글 시트를 자동으로 업데이트합니다.

---

## 📂 파일 목록 및 역할

### 1. `lotto_updater.py` (핵심 엔진)
- **역할**: 로또 당첨 결과를 웹에서 수집하고, Gemini AI를 통해 JSON으로 파싱하여 구글 시트에 저장합니다.
- **주요 기능**:
  - `Model Explorer`: 사용 가능한 Gemini 모델을 탐색하고 최적의 모델(Gemini 1.5 Flash)을 선택합니다.
  - `fetch_lotto_data_via_gemini(round_no)`: 특정 회차의 검색 결과를 크롤링하고 AI에게 파싱을 요청합니다.
  - `update_sheet(data)`: 파싱된 데이터를 구글 시트('로또 max')에 추가합니다.
  - `check_schedule()`: `schedule_config.json`과 현재 시간을 비교하여 실행 여부를 결정합니다.
  - `log_execution()`: 실행 결과와 상태를 'Log' 시트에 기록합니다.

### 2. `schedule_config.json` (설정 파일)
- **역할**: 사용자가 실행 요일과 시간을 관리하는 파일입니다.
- **기본 설정**: 매주 토요일 21:00 (KST) 실행.
- **활용**: 스크립트 실행 시 이 파일을 참조하여 현재 시간이 실행 허용 범위(Window) 내인지 확인합니다.

### 3. `.github/workflows/lotto_sync.yml` (자동화 설정)
- **역할**: GitHub Actions를 통해 클라우드 환경에서 정해진 시간에 스크립트를 실행합니다.
- **트리거**:
  - `schedule`: 매주 토요일 12:00 UTC (한국 시간 21:00) 자동 실행.
  - `workflow_dispatch`: GitHub 웹에서 수동으로 즉시 실행 가능 (Force Run).

---

## 🛠️ 업데이트 가이드 (유지보수)

### 새로운 데이터를 추가 수집하고 싶을 때
1. **Gemini 프롬프트 수정**: `lotto_updater.py`의 `fetch_lotto_data_via_gemini` 함수 내 `prompt` 변수를 수정하여 원하는 데이터 필드를 요청하세요.
2. **JSON 파싱 로직 확인**: Gemini가 반환하는 JSON 키가 프롬프트 요청과 일치하는지 확인하세요.
3. **시트 저장 로직 수정**: `update_sheet` 함수에서 `ws.append_row`에 전달되는 리스트(`row`)에 새로운 필드를 추가하세요. (구글 시트 헤더도 맞춰야 합니다.)

### 실행 스케줄을 변경하고 싶을 때
1. **GitHub Actions**: `.github/workflows/lotto_sync.yml`의 `cron` 값을 수정하세요. (UTC 기준임에 주의)
2. **로컬/스크립트 검증**: `schedule_config.json`의 `day_of_week`와 `hour`를 수정하세요.

---

## 🗑️ 삭제 가이드 (안전한 제거)

이 기능을 시스템에서 제거하거나 파일을 삭제할 때 다음 체크리스트를 확인하세요.

- [ ] **GitHub Actions 비활성화**: `.github/workflows/lotto_sync.yml` 파일을 삭제하거나 비활성화하여 자동 실행을 중단하세요.
- [ ] **의존성 확인**: 다른 파이썬 파일이 `lotto_updater.py`를 import 하고 있는지 확인하세요. (현재는 독립 실행 모듈임)
- [ ] **구글 시트 권한**: `creds_lotto.json` (Service Account)의 시트 접근 권한을 해제하거나 키를 파기해도 됩니다.
- [ ] **환경 변수 정리**: 로컬의 `.env` 파일이나 GitHub Secrets에서 `GEMINI_API_KEY`, `GOOGLE_CREDS_JSON`을 삭제하세요.

---

## ⚠️ 기술 규격 및 주의사항
- **환경**: Python 3.10+
- **필수 라이브러리**: `google-genai` (New SDK), `gspread`, `oauth2client`, `beautifulsoup4`, `requests`, `python-dotenv`
- **인증**: GitHub Secrets에 `GOOGLE_CREDS_JSON`과 `GEMINI_API_KEY`가 설정되어 있어야 합니다.

<br>

---
---

# 🚀 Hybrid Sniper V5: Dual-Mode & AI Taxonomy (Final Edition)

이 섹션은 시스템의 **최종 완성형**인 **이원화 실행 모드 및 통합 분석 엔진**에 대한 설명입니다.
사용자님의 M5 MacBook 하드웨어를 보호하면서도, 상황에 따라 유연하게 실행할 수 있는 두 가지 모드를 제공합니다.

## 🔀 실행 모드 (Execution Modes)

### 1. Manual Mode (매뉴얼 모드 - Full Cycle)
-   **명령어**: `python lotto_predict.py`
-   **설명**: 사령관님의 직접 명령으로 간주하여, **[데이터 수집 -> 통합 분석(ML+DL) -> 최종 저격]** 전 과정을 논스톱으로 수행합니다.
-   **특징**: 단계별로 충분한 **하드웨어 쿨링 타임(5~10초)**이 자동으로 적용됩니다.

### 2. Scheduled Mode (스케줄 모드 - Distributed)
-   **명령어**: `python lotto_predict.py --scheduled`
-   **설명**: 자동화 스케줄러(cron 등)에 의해 실행될 때 사용됩니다. 오늘 요일에 맞는 미션만 수행하고 종료합니다.
    -   **일**: Sync Data
    -   **월**: Total Analysis (ML/DL)
    -   **수**: Final Strike

## 🧠 AI Taxonomy & Architecture (기술 체계)

### 1. 지도 학습 (Supervised Learning)
-   **역할**: 과거 데이터를 정답지(Label)로 삼아 미래를 예측합니다.
-   **구성**:
    -   **분류(Classification)**: RandomForest, XGBoost 등이 "이 번호가 나올 확률이 높은가?"를 판단합니다.
    -   **특징 추출(Feature Extraction)**: LSTM, CNN (Deep Learning) 모델이 시계열 데이터의 숨겨진 특징을 **인코더-디코더(Encoder-Decoder)** 구조로 추출합니다.

### 2. 비지도 학습 (Unsupervised Learning)
-   **역할**: 정답 없이 데이터 자체의 패턴을 발견합니다.
-   **구성**:
    -   **군집화(Clustering)**: KMeans 알고리즘이 최근 당첨 번호들의 패턴을 그룹화하여 현재 흐름이 어떤 유형인지 파악합니다.
    -   **차원 축소(Dimensionality Reduction)**: PCA 기법을 사용하여 복잡한 고차원 데이터의 핵심 특징만 요약합니다.

### 3. 강화 학습 (Reinforcement Learning)
-   **역할**: 예측 결과에 대한 보상(Reward)을 통해 모델을 진화시킵니다.
-   **구성**:
    -   **PPO 가중치**: 최근 5회차 성적을 바탕으로, 잘 맞춘 모델에게 더 높은 발언권(가중치)을 부여하는 보상 체계를 적용합니다.

### 4. 생성형 AI (Generative AI)
-   **역할**: 수치 데이터를 기반으로 인간 수준의 전략적 판단을 내립니다.
-   **구성**:
    -   **LLM 필터링**: Gemini 1.5 Pro (또는 가용한 모델)가 유전 알고리즘이 만든 후보군을 검토하고, 최종 10개 조합을 생성합니다.
    -   **Dynamic Discovery**: API 연결 시 사용 가능한 모델(Pro/Flash)을 자동으로 탐색하여 연결합니다.

---

## 🛡️ 안전 제1수칙 (Safety Protocols)

1.  **자원 제한**: 전체 CPU 코어 중 2개를 시스템용으로 남겨두어 쾌적함을 유지합니다.
2.  **Safety Pause**: ML과 DL 분석 사이(5초), 전체 사이클 단계 간(10초), 유전 알고리즘 세대 간(1.5초)에 **휴식 시간**을 두어 M5 칩의 과열을 방지합니다.
3.  **Memory Clean**: 단계별로 메모리를 강제 회수(GC)하여 안정성을 확보합니다.
4.  **Sync Protection**: 데이터 수집 3회 실패 시 자동으로 중단하여 무한 루프를 방지합니다.
