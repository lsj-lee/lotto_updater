# lotto_updater.py
# Hybrid Sniper V5: 지능형 데이터 자동화 패키지 (Phase 1)
#
# 이 스크립트는 로또 당첨 번호를 자동으로 수집하고 구글 시트를 업데이트합니다.
# Gemini 1.5 Flash를 사용하여 웹 검색 결과에서 데이터를 파싱하며,
# GitHub Actions와 연동하여 정해진 스케줄에 따라 실행됩니다.

import os
import json
import datetime
import time
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

# 환경 변수 로드 (.env 파일이 있을 경우)
load_dotenv()

# ==========================================
# 설정 (Configuration)
# ==========================================
CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'
SCHEDULE_CONFIG_FILE = 'schedule_config.json'
LOG_SHEET_NAME = 'Log'

# User-Agent 설정 (크롤링 차단 방지)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def setup_credentials():
    """
    보안 및 로깅: 인증 파일(creds_lotto.json) 자동 생성
    GitHub Secrets(GOOGLE_CREDS_JSON)가 환경 변수로 존재하면 파일로 생성합니다.
    """
    if not os.path.exists(CREDS_FILE):
        creds_json = os.getenv('GOOGLE_CREDS_JSON')
        if creds_json:
            print(f"🔑 [보안] {CREDS_FILE} 파일이 없어 환경 변수에서 생성합니다.")
            with open(CREDS_FILE, 'w', encoding='utf-8') as f:
                f.write(creds_json)
        else:
            print(f"⚠️ [경고] {CREDS_FILE} 파일도 없고 GOOGLE_CREDS_JSON 환경 변수도 없습니다.")

def check_schedule():
    """
    지능형 실행 필터: schedule_config.json 설정과 현재 시간을 대조하여 실행 여부 결정
    """
    if not os.path.exists(SCHEDULE_CONFIG_FILE):
        print(f"ℹ️ [스케줄] {SCHEDULE_CONFIG_FILE} 파일이 없습니다. 기본적으로 실행합니다.")
        return True

    try:
        with open(SCHEDULE_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 강제 실행 모드 확인
        if config.get('force_run', False):
            print("🚀 [스케줄] 강제 실행 모드가 활성화되었습니다.")
            return True

        now = datetime.datetime.now()
        # 요일 확인 (Mon, Tue, ...)
        current_day = now.strftime('%a')
        if current_day not in config.get('active_days', []):
            print(f"zzz [스케줄] 오늘은 실행 요일이 아닙니다. ({current_day})")
            return False

        # 시간 확인 (0~23)
        current_hour = now.hour
        if current_hour not in config.get('active_hours', []):
            print(f"zzz [스케줄] 현재 시간({current_hour}시)은 실행 시간이 아닙니다.")
            return False

        print("✅ [스케줄] 실행 조건이 충족되었습니다.")
        return True

    except Exception as e:
        print(f"⚠️ [스케줄] 설정 파일 읽기 오류: {e}. 안전하게 실행을 진행합니다.")
        return True

def get_best_model():
    """
    지능형 모델 탐색 (Model Explorer): 사용 가능한 Gemini 모델 중 최적 모델 선택
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

    genai.configure(api_key=api_key)

    preferred_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    available_models = []

    print("🔍 [Model Explorer] 사용 가능한 모델 검색 중...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)

        # 선호 모델 순서대로 확인
        for pref in preferred_models:
            for model in available_models:
                if pref in model:
                    print(f"✨ [Model Explorer] 최적 모델 선택됨: {model}")
                    return model

        # 선호 모델이 없으면 첫 번째 사용 가능 모델 반환
        if available_models:
            print(f"⚠️ [Model Explorer] 선호 모델을 찾지 못해 대체 모델 선택: {available_models[0]}")
            return available_models[0]

    except Exception as e:
        print(f"⚠️ [Model Explorer] 모델 목록 조회 실패: {e}. 기본값 사용.")

    return 'gemini-1.5-flash' # Fallback

def scrape_lotto_data(draw_no):
    """
    Gemini 기반 데이터 파싱 (No API) 1단계: 웹 검색 결과 크롤링
    """
    query = f"로또 {draw_no}회 당첨번호"
    url = f"https://search.naver.com/search.naver?query={query}"

    print(f"🌐 [크롤링] {draw_no}회차 데이터 검색 중... ({url})")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 텍스트만 추출하여 Gemini에게 전달 (불필요한 태그 제거)
        # 네이버 로또 결과 영역이나 전체 텍스트를 가져옴
        text_content = soup.get_text(separator=' ', strip=True)
        # 너무 긴 텍스트는 잘라서 전달 (토큰 절약 및 정확도 향상)
        return text_content[:5000]
    except Exception as e:
        print(f"❌ [크롤링] 검색 실패: {e}")
        return None

def parse_with_gemini(model_name, raw_text, draw_no):
    """
    Gemini 기반 데이터 파싱 (No API) 2단계: 비정형 텍스트 -> JSON 변환
    """
    prompt = f"""
    아래 텍스트는 로또 {draw_no}회 당첨 결과 검색 내용입니다.
    이 텍스트에서 다음 정보를 추출하여 정확한 JSON 형식으로 출력하세요.
    JSON 키: "drwNo" (회차, 정수), "drwtNo1", "drwtNo2", "drwtNo3", "drwtNo4", "drwtNo5", "drwtNo6" (당첨번호 6개, 정수), "bnusNo" (보너스번호, 정수), "firstWinamnt" (1등 당첨금, 숫자만, 원 단위), "firstPrzwnerCo" (1등 당첨자 수, 정수), "firstAccumamnt" (1등 총 당첨금, 숫자만, 없으면 0).

    텍스트:
    {raw_text}

    오직 JSON만 출력하세요. 마크다운 코드 블록 없이.
    """

    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # 마크다운 제거 (```json ... ```)
        if text.startswith('```'):
            text = text.split('\n', 1)[1]
            if text.endswith('```'):
                text = text.rsplit('\n', 1)[0]

        data = json.loads(text)
        return data
    except Exception as e:
        print(f"❌ [Gemini] 파싱 실패: {e}")
        return None

def update_google_sheet(sheet, data_list):
    """
    구글 시트 업데이트
    """
    if not data_list:
        return

    # 데이터 포맷팅 (시트 컬럼 순서에 맞게)
    # 가정: 회차, 날짜(오늘), 번호1, 번호2, 번호3, 번호4, 번호5, 번호6, 보너스, 1등당첨금, 당첨자수
    rows_to_add = []
    today_str = datetime.date.today().strftime('%Y-%m-%d')

    for data in data_list:
        row = [
            data.get('drwNo'),
            today_str, # 추첨일 대신 수집일 기록 (혹은 Gemini가 추첨일도 파싱하게 할 수 있음)
            data.get('drwtNo1'),
            data.get('drwtNo2'),
            data.get('drwtNo3'),
            data.get('drwtNo4'),
            data.get('drwtNo5'),
            data.get('drwtNo6'),
            data.get('bnusNo'),
            data.get('firstWinamnt'),
            data.get('firstPrzwnerCo')
        ]
        rows_to_add.append(row)

    try:
        # 마지막 행에 추가
        sheet.append_rows(rows_to_add)
        print(f"💾 [시트] {len(rows_to_add)}개 회차 데이터 업데이트 완료.")
    except Exception as e:
        print(f"❌ [시트] 업데이트 실패: {e}")

def log_execution(doc, model_name, status, updated_count):
    """
    로그 기록: 'Log' 탭에 실행 정보 저장
    """
    try:
        worksheet = doc.worksheet(LOG_SHEET_NAME)
    except:
        worksheet = doc.add_worksheet(title=LOG_SHEET_NAME, rows=1000, cols=10)
        worksheet.append_row(['실행시간', '사용모델', '상태', '업데이트 수'])

    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    worksheet.append_row([now_str, model_name, status, updated_count])
    print(f"📝 [로그] 실행 기록 저장 완료.")

def main():
    # 1. 인증 파일 준비
    setup_credentials()

    # 2. 스케줄 확인
    if not check_schedule():
        return

    # 3. 구글 시트 연결
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
        client = gspread.authorize(creds)
        doc = client.open(SHEET_NAME)
        sheet = doc.sheet1 # 첫 번째 시트 사용 가정 ('로또 max'의 메인 시트)
    except Exception as e:
        print(f"❌ [초기화] 구글 시트 연결 실패: {e}")
        return

    # 4. 마지막 회차 확인 (무결성 검증)
    try:
        # A열(회차)의 모든 값 가져오기
        col_values = sheet.col_values(1)
        # 헤더 제외하고 정수로 변환하여 최대값 찾기
        valid_values = []
        for v in col_values:
            if v.isdigit():
                valid_values.append(int(v))

        last_draw = max(valid_values) if valid_values else 0
        print(f"📊 [무결성] 현재 시트 마지막 회차: {last_draw}")
    except Exception as e:
        print(f"❌ [무결성] 마지막 회차 확인 실패: {e}")
        return

    # 5. 모델 선택
    model_name = get_best_model()

    # 6. 누락 데이터 수집 (최신 회차까지)
    # 최신 회차 추정 (매주 토요일 추첨)
    # 기준일: 1회차(2002-12-07)
    base_date = datetime.date(2002, 12, 7)
    today = datetime.date.today()
    days_diff = (today - base_date).days
    estimated_latest_draw = (days_diff // 7) + 1

    # 오늘이 토요일이고 20시 40분 이전이면 아직 추첨 전일 수 있음 (안전하게 -1 처리 혹은 시간 체크)
    # 여기서는 단순하게 추정값 사용하고 데이터 없으면 스킵

    print(f"🎯 [목표] 예상 최신 회차: {estimated_latest_draw}")

    new_data = []

    for draw_no in range(last_draw + 1, estimated_latest_draw + 1):
        print(f"🚀 [수집] {draw_no}회차 데이터 수집 시작...")

        raw_text = scrape_lotto_data(draw_no)
        if not raw_text:
            print(f"⚠️ [수집] {draw_no}회차 크롤링 실패. 건너뜀.")
            continue

        parsed_data = parse_with_gemini(model_name, raw_text, draw_no)
        if parsed_data:
            # 검증: 회차가 맞는지 확인
            if str(parsed_data.get('drwNo')) == str(draw_no):
                new_data.append(parsed_data)
                print(f"✅ [수집] {draw_no}회차 파싱 성공: {parsed_data.get('drwtNo1')}~{parsed_data.get('drwtNo6')}")
            else:
                print(f"⚠️ [검증] 파싱된 회차({parsed_data.get('drwNo')})가 요청 회차({draw_no})와 다름.")
        else:
            print(f"⚠️ [수집] {draw_no}회차 파싱 실패.")

        # API 및 서버 부하 방지를 위한 딜레이
        time.sleep(2)

    # 7. 시트 업데이트
    if new_data:
        update_google_sheet(sheet, new_data)
        log_execution(doc, model_name, "Success", len(new_data))
    else:
        print("ℹ️ [완료] 업데이트할 새로운 데이터가 없습니다.")
        log_execution(doc, model_name, "No New Data", 0)

if __name__ == "__main__":
    main()
