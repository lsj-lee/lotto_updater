# Hybrid Sniper V5: 지능형 데이터 자동화 패키지 (Phase 1)

## 1. 개요
이 문서는 Hybrid Sniper V5 시스템의 핵심인 **지능형 데이터 자동화 패키지**에 대한 설명서입니다.
본 패키지는 Google Gemini 1.5 Flash를 활용하여 웹에서 로또 당첨 데이터를 수집하고, 무결성을 검증하며 구글 시트를 자동으로 업데이트합니다.

## 2. 파일 목록 및 역할

### `lotto_updater.py` (핵심 엔진)
- **역할**: 웹 크롤링 및 데이터 파싱, 구글 시트 업데이트를 수행하는 메인 스크립트입니다.
- **주요 함수**:
  - `setup_credentials()`: GitHub Secrets 또는 환경 변수로부터 `creds_lotto.json` 인증 파일을 생성합니다.
  - `check_schedule()`: `schedule_config.json`을 읽어 현재 시간에 실행해야 할지 결정합니다.
  - `get_best_model()`: 사용 가능한 Gemini 모델 중 최적 모델(예: gemini-1.5-flash)을 자동으로 선택합니다.
  - `scrape_lotto_data(draw_no)`: 네이버/구글 검색 결과에서 해당 회차의 로또 당첨 정보를 크롤링합니다.
  - `parse_with_gemini(model_name, raw_text, draw_no)`: Gemini에게 불규칙한 텍스트를 전달하여 정형화된 JSON 데이터로 변환합니다.
  - `update_google_sheet(sheet, data_list)`: 수집된 데이터를 구글 시트에 추가합니다.
  - `log_execution(doc, model_name, status, updated_count)`: 실행 결과(성공/실패, 모델명 등)를 'Log' 시트에 기록합니다.

### `.github/workflows/lotto_sync.yml` (자동 실행 설정)
- **역할**: GitHub Actions에서 정해진 시간에 `lotto_updater.py`를 실행하도록 스케줄링합니다.
- **설정**:
  - Cron 스케줄: 매일 한국 시간 09:00 ~ 23:00 사이 매 시간 실행 (`0 0-14 * * *` UTC)
  - 환경 변수: `GEMINI_API_KEY`, `GOOGLE_CREDS_JSON` Secrets 사용

### `schedule_config.json` (사용자 설정)
- **역할**: 사용자가 직접 실행 요일과 시간을 제어할 수 있는 설정 파일입니다.
- **주요 항목**:
  - `active_days`: 실행할 요일 (예: ["Sat", "Sun", "Mon"])
  - `active_hours`: 실행할 시간대 (0~23)
  - `force_run`: `true`로 설정 시 스케줄 무시하고 즉시 실행

## 3. 업데이트 가이드
새로운 데이터 소스를 추가하거나 로직을 변경할 때 참고하세요.

1.  **크롤링 대상 변경**:
    - `lotto_updater.py` 내 `scrape_lotto_data` 함수에서 `url` 변수를 수정하세요.
    - `BeautifulSoup` 파싱 로직을 대상 사이트 구조에 맞게 변경하세요.

2.  **데이터 파싱 형식 변경**:
    - `parse_with_gemini` 함수 내의 `prompt` 내용을 수정하여 Gemini에게 새로운 지시를 내리세요.
    - JSON 키 값을 변경할 경우 `update_google_sheet` 함수의 매핑 로직도 함께 수정해야 합니다.

3.  **스케줄 변경**:
    - GitHub Actions 실행 빈도는 `.github/workflows/lotto_sync.yml`의 `cron` 값을 수정하세요.
    - 실제 실행 허용 시간은 `schedule_config.json`을 수정하세요.

## 4. 삭제 가이드
특정 기능을 제거할 때 다음 사항을 확인하세요.

-   **Gemini 모델 탐색 기능 제거**: `get_best_model` 함수를 제거하고, `model_name` 변수에 고정된 모델명 문자열을 할당하세요.
-   **스케줄 필터 제거**: `check_schedule` 함수 호출부를 주석 처리하세요.
-   **전체 삭제**: `lotto_updater.py`, `schedule_config.json`, `.github/workflows/lotto_sync.yml` 파일을 삭제하고, GitHub Repository의 Secrets(`GEMINI_API_KEY`, `GOOGLE_CREDS_JSON`)를 제거하세요.

## 5. 기술 규격
-   **환경**: Python 3.10+, MacBook Pro M5 (로컬 테스트 시), Ubuntu-latest (GitHub Actions)
-   **라이브러리**:
    -   `google-generativeai`: Gemini API 연동
    -   `gspread`, `oauth2client`: 구글 시트 연동
    -   `requests`, `beautifulsoup4`: 웹 크롤링
    -   `python-dotenv`: 환경 변수 관리
-   **보안**: 구글 인증 파일(`creds_lotto.json`)은 저장소에 업로드하지 않으며, 환경 변수를 통해 동적으로 생성합니다.
