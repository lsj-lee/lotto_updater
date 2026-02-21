# 로또 AI 통합 시스템 (Hybrid Sniper V5) 코드 설명서

이 문서는 로또 당첨 번호 예측을 위한 AI 시스템의 전체 코드 구조와 기능별 상세 설명을 담고 있습니다. 본 시스템은 최신 딥러닝 기술(TabNet, LSTM, cGAN)과 생성형 AI(Gemini)를 결합한 **'Hybrid Sniper V5'** 아키텍처를 기반으로 합니다.

---

## 1. 시스템 개요 (System Overview)

본 프로젝트는 과거 로또 데이터를 학습하여 다음 회차의 당첨 번호를 예측하고, 구글 시트에 결과를 리포트하며, 스스로 코드를 개선(Self-Evolution)하는 자율형 AI 시스템입니다.

- **핵심 기술**: Tabular Feature Attention (TabNet 응용), LSTM (시계열 예측), cGAN (데이터 증강), PPO (강화학습 기반 가중치 최적화).
- **전략 엔진**: Google Gemini 1.5 Pro/Flash (최적 조합 생성 및 R&D 분석).
- **하드웨어 최적화**: Apple Silicon (M5) `mps` 가속 지원.

---

## 2. 파일별 기능 상세 (Detailed Functions)

### 📁 1. 데이터 동기화 (`lotto_updater.py`)
> **목적**: 최신 로또 당첨 결과를 가져와 구글 시트('로또 max')에 업데이트합니다.

1.  **구글 시트 연결**: `gspread`를 사용하여 '로또 max' 시트에 접근하고, 마지막으로 기록된 회차를 확인합니다.
2.  **데이터 수집 (이중화 전략)**:
    -   **1차 시도 (API)**: 동행복권 공식 API(`dhlottery.co.kr`)를 호출하여 당첨 번호, 보너스 번호, 당첨자 수, 당첨금을 가져옵니다.
    -   **2차 시도 (크롤링)**: API 차단 시, 네이버 검색 결과(`search.naver.com`)를 `BeautifulSoup`으로 크롤링하여 데이터를 확보합니다 (Fallback).
3.  **데이터 정제 및 저장**:
    -   문자열(예: "1,234원")에서 콤마(`,`)와 단위(`원`, `명`)를 제거하고 정수형(`int`)으로 변환합니다.
    -   시트의 최상단(2행)에 새로운 데이터를 삽입합니다.

### 📁 2. 라이트급 사전 학습 (`lotto_brain.py`)
> **목적**: 9차원 데이터(번호 6개 + 보너스 + 당첨자 수 + 금액)를 기반으로 가볍고 빠르게 모델을 학습시킵니다. (`자동시작.sh`의 1단계)

1.  **환경 설정**: M5 칩(`mps`) 가속을 활성화하고 구글 시트 데이터를 로드합니다.
2.  **모델 구조 (`LottoBrain`)**:
    -   입력: 9개 특징 (번호, 보너스, 당첨자 수, 당첨금).
    -   구조: LSTM (3 layers) -> Fully Connected Layer.
3.  **멀티 스케일 학습**:
    -   8가지 시야(10, 50, 100, 200, 300, 400, 500, 1000주)에 대해 순차적으로 학습합니다.
    -   각 시야별로 학습된 모델을 `.pth` 파일로 저장합니다.
    -   *참고*: 이 스크립트는 `lotto_predict.py`와 독립적으로 실행되며, 주로 모델의 초기 패턴 학습용으로 사용됩니다.

### 📁 3. 핵심 예측 엔진 (`lotto_predict.py`)
> **목적**: **Hybrid Sniper V5**의 핵심 로직. 심층 학습, 예측, 전략 수립, 리포팅, 자율 진화를 모두 수행합니다.

#### [단계 1] 환경 및 모델 정의
-   **하드웨어**: `torch.device("mps")`로 설정.
-   **모델 아키텍처**:
    -   **TabularFeatureAttention**: TabNet의 개념을 차용하여 입력 특징(Feature) 간의 중요도를 학습하고 마스킹합니다.
    -   **LottoBrain (Main)**: Feature Attention -> LSTM -> Self-Attention -> 출력으로 이어지는 파이프라인.
    -   **cGAN (LottoGenerator/Discriminator)**: 10만 개의 가상 당첨 패턴을 생성하여 '당첨 확률 가중치'를 계산합니다.

#### [단계 2] 데이터 파이프라인 (`load_data`)
-   '시트1'에서 데이터를 로드하고 역순(과거->최신)으로 정렬합니다.
-   **특성 공학(Feature Engineering)**: Gap(미출현 기간), Odd/Even Ratio(홀짝 비율), Sum(총합) 등 파생 변수를 추가합니다.
-   MinMax 스케일링으로 데이터를 0~1 사이로 정규화합니다.

#### [단계 3] 학습 및 예측 루프 (`run_pipeline`)
-   **cGAN 학습**: 생성적 적대 신경망을 통해 데이터 증강을 수행하고, 번호별 가중치(`cgan_weights`)를 산출합니다.
-   **8단계 시야(Scale) 학습**: 10주~1000주 데이터를 각각 학습합니다.
-   **PPO Inspired Weighting**: 최근 5주간의 예측 성과(Reward)를 측정하여, 잘 맞춘 모델(Scale)에 더 높은 가중치(Policy)를 부여합니다.

#### [단계 4] 전략 수립 및 게임 생성 (`analyze_and_generate`)
-   **점수 통합**: 최신성(Recency), LSTM 앙상블 점수, cGAN 가중치를 합산하여 45개 번호의 최종 점수를 매깁니다.
-   **Gemini AI 전략가**:
    -   상위 45개 확률 데이터를 Gemini(1.5 Pro/Flash)에게 전송합니다.
    -   Gemini는 **'정예 번호 15~20개'**를 선별하고, 이를 조합하여 **'최종 10게임'**을 생성합니다.
    -   **R&D Insight**: 이번 예측의 기술적 근거(TabNet, cGAN 효과 등)를 분석하여 코멘트를 작성합니다.
    -   *Fallback*: API 오류 시, 자체 알고리즘(Elite-20)으로 게임을 생성합니다.

#### [단계 5] 리포트 작성 (`update_report`)
-   구글 시트 '추천번호' 탭에 접속하여 기존 데이터를 초기화합니다.
-   타이틀, 전략 요약, 생성된 10게임, R&D Insight를 포맷에 맞춰 기록합니다 (셀 병합 포함).

#### [단계 6] 자율 진화 (`generate_evolution_proposal`)
-   현재 실행 중인 코드(`lotto_predict.py`)를 읽어와 Gemini에게 분석을 요청합니다.
-   TabNet, cGAN, PPO 등을 개선한 **'진화된 전체 코드'**를 생성하여 `proposals/` 폴더에 저장합니다.

### 📁 4. 진화 관리자 (`evolution_manager.py`)
> **목적**: AI가 제안한 코드 변경 사항을 검토하고 시스템에 적용합니다.

1.  **제안서 스캔**: `proposals/` 폴더에서 가장 최신 파이썬 파일을 찾습니다.
2.  **헤더 분석**: 제안서의 `[Evolution Proposal]` 헤더(변경 사항, 기대 효과)를 추출하여 보여줍니다.
3.  **사용자 결정**:
    -   `[A]pply`: 현재 코드를 백업(`lotto_predict_bak.py`)하고, 제안된 코드로 덮어씁니다.
    -   `[D]elete`: 제안서를 삭제합니다.
    -   `[C]ancel`: 작업을 취소합니다.

### 📁 5. 유틸리티 (`api_check.py`)
> **목적**: 현재 사용 가능한 Gemini 모델 리스트를 확인하고 API 연결 상태를 점검합니다.

### 📁 6. 자동화 스크립트 (`자동시작.sh`)
> **목적**: 전체 프로세스를 순차적으로 실행하는 쉘 스크립트입니다.

1.  **중복 방지**: 오늘 날짜를 확인하여 하루에 한 번만 실행되도록 합니다.
2.  **Step 1**: `lotto_brain.py` 실행 (라이트급 학습).
3.  **Step 2**: `lotto_predict.py` 실행 (메인 예측 및 리포팅).
4.  **Step 3**: `로또_분석.py` 실행 (심화 분석 - *현재 파일 누락됨*).

### 📁 7. 깃허브 액션 (`.github/workflows/lotto_sync.yml`)
> **목적**: 매주 정기적으로 데이터 동기화 작업을 수행합니다.

-   **스케줄**: 매주 월, 화, 수 오전 10시 (UTC) 또는 수동 실행(`workflow_dispatch`).
-   **작업 흐름**:
    1.  리포지토리 체크아웃 및 Python 3.10 설정.
    2.  필요 라이브러리 설치 (`google-genai`, `gspread` 등).
    3.  시크릿 변수(`GOOGLE_CREDS_JSON`)를 `creds_lotto.json` 파일로 생성.
    4.  `lotto_updater.py` 실행하여 데이터 최신화.

---

## 3. 실행 워크플로우 (Workflow)

1.  **준비**: `.env` 파일에 `GEMINI_API_KEY_1` 등 필수 키가 설정되어 있어야 합니다.
2.  **데이터 업데이트**: (수동 또는 별도 스케줄러로) `python lotto_updater.py` 실행.
3.  **예측 및 리포트**:
    -   자동: `./자동시작.sh` 실행.
    -   수동: `python lotto_predict.py` 실행.
4.  **진화 확인**: 실행 후 `python evolution_manager.py`를 통해 AI가 제안한 코드 개선안을 검토 및 적용.
