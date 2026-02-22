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
- **필수 라이브러리**: `google-generativeai`, `gspread`, `oauth2client`, `beautifulsoup4`, `requests`, `python-dotenv`
- **인증**: GitHub Secrets에 `GOOGLE_CREDS_JSON`과 `GEMINI_API_KEY`가 설정되어 있어야 합니다.

<br>

---
---

# 🚀 Hybrid Sniper V5: 자율 진화 오케스트레이션 (Evolutionary Orchestration)

이 섹션은 시스템의 **최종 완성형**인 **자율 진화 및 통합 오케스트레이션 엔진**에 대한 설명입니다.
데이터 수집부터 모델 학습, 예측, 그리고 코드 자체의 개선 제안까지 하나의 흐름으로 자동화되었습니다.

## 🌀 핵심 사이클 (The Cycle)

### 1. Unified Orchestration (통합 지휘)
-   기존의 `Updater`와 `Predictor`를 `HybridSniperOrchestrator` 클래스로 통합했습니다.
-   **단 한 번의 실행**으로 [데이터 업데이트 -> PPO 가중치 계산 -> 앙상블 학습 -> 유전 진화 -> Gemini 필터링 -> 리포트 작성]이 순차적으로 수행됩니다.

### 2. PPO Reinforcement Learning (성과 기반 보상)
-   최근 5회차 데이터를 검증 세트(Validation Set)로 활용하여 각 모델(RF, XGB, LSTM 등)의 성능을 평가합니다.
-   **PPO(Proximal Policy Optimization)** 개념을 차용하여, 성적이 좋은 모델에게 더 높은 **투표 가중치(Weight)**를 부여하고 실전 예측에 반영합니다.

### 3. Self-Evolution (자기 개량)
-   모든 분석이 끝나면, Gemini 1.5 Pro가 **자신의 코드(`lotto_predict.py`)를 스스로 읽고 리뷰**합니다.
-   알고리즘 최적화 방안이나 버그 수정 제안을 'Log' 시트에 자동으로 기록하여, 사령관님이 다음 버전에 반영할 수 있도록 합니다.

## 🛡️ M5 하드웨어 세이프가드 (Hardware Integrity)

1.  **CPU Core Limit**: 전체 코어 중 2개를 시스템용으로 남겨두어 백그라운드 작업 시에도 맥북이 쾌적하게 유지됩니다.
2.  **Cooling Pause**: 유전 알고리즘 진화 50세대마다 1.5초간 휴식하여 칩셋 발열을 제어합니다.
3.  **Memory GC**: 각 단계가 끝날 때마다 메모리를 강제로 회수하여 장시간 실행에도 튕기지 않습니다.

## 💻 실행 가이드 (How to Run)

터미널에서 아래 명령어를 입력하면 오케스트레이션이 시작됩니다.

```bash
python lotto_predict.py
```
> **자동화 시나리오:**
> 1. 최신 당첨 결과가 있는지 확인하고 자동 업데이트.
> 2. 모델별 최근 성적표를 매겨 가중치 산정 (PPO).
> 3. 가중치를 적용한 앙상블 & 유전 알고리즘 가동.
> 4. 최종 결과 리포트 및 **코드 개선 제안서** 작성.
