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

# 🧠 Hybrid Sniper V5: 지능형 학습 엔진 (Phase 2)

이 섹션은 **Phase 2**에서 추가된 **지능형 분석 및 예측 시스템**에 대한 설명입니다.
사용자님의 M5 MacBook 성능을 활용하여 로또 데이터를 심층 학습하고, 최적의 번호를 추천합니다.

## 📂 핵심 파일
### `lotto_predict.py` (The Brain)
- **역할**: 수집된 데이터를 학습하고, 미래의 당첨 번호를 예측합니다.
- **실행 위치**: 사용자님의 로컬 PC (MacBook Pro M5)

## 💡 쉬운 설명 (Logic Explanation)
이 시스템은 마치 **3명의 전문가와 1명의 연습생**이 협력하는 것과 같습니다.

1.  **LSTM (Time Machine) - 역사학자**
    -   과거의 데이터를 시간 순서대로 쭉 살펴봅니다. "최근 10주 동안 이런 흐름이었으니, 다음엔 이게 나오겠네?"라고 과거의 흐름(Sequence)을 읽습니다. 10주, 50주... 900주(데이터 양에 맞춰 조정됨)까지 다양한 렌즈로 봅니다.
2.  **TabNet (Logician) - 통계학자**
    -   숫자의 특징을 분석합니다. "홀수가 너무 많이 나왔어", "합계가 너무 높아" 같은 논리적 패턴(Feature)을 찾아냅니다.
3.  **GNN (Chemist) - 관계 전문가**
    -   숫자들 사이의 '궁합'을 봅니다. "7번이랑 12번은 자주 같이 나오네?"라는 관계(Relation)를 파악합니다.
4.  **cGAN (Imagination Engine) - 상상력이 풍부한 연습생**
    -   과거 데이터가 부족할 때, "만약 이런 상황이라면 어떨까?" 하고 가상의 데이터를 만들어냅니다. 이를 통해 AI가 더 많은 연습 문제(Data Augmentation)를 풀어보고 실력을 키우게 합니다. (데이터 10배 증폭 효과)

## 🚀 하드웨어 가속 (Hardware: MacBook Pro M5)
사용자님의 맥북은 **슈퍼카**와 같은 엔진(GPU)을 가지고 있습니다. 이 코드는 그 엔진을 100% 활용합니다.

-   **MPS (Metal Performance Shaders)**: 코드는 자동으로 `mps` 모드를 감지하여 실행됩니다. 일반 CPU보다 수십 배 빠른 속도로 계산합니다.
-   **FP16 (반정밀도)**: 계산할 때 소수점 아래 너무 긴 자리까지 계산하지 않고, 적당히 잘라내어(16비트) 속도는 높이고 메모리는 아낍니다. (M5 칩에 최적화된 방식)

## 💻 실행 가이드 (How to Run)

터미널을 열고 프로젝트 폴더로 이동한 뒤, 아래 명령어를 순서대로 입력하세요.

**1. 필수 도구 설치 (최초 1회만)**
```bash
pip install -r requirements.txt
```

**2. 지능형 학습 및 예측 시작**
```bash
python lotto_predict.py
```
> **실행하면 벌어지는 일:**
> 1. 구글 시트에서 최신 데이터를 가져옵니다.
> 2. cGAN이 가상 데이터를 만들어 연습량을 늘립니다.
> 3. 3명의 전문가 AI(LSTM, TabNet, GNN)가 머리를 맞대고 학습합니다.
> 4. 학습이 끝나면 '추천번호' 시트에 **10개의 게임**과 **분석 리포트**를 작성합니다.
