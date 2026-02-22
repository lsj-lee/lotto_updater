import os
import json
import time
import datetime
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

# .env 파일 로드 (로컬 개발 환경용)
load_dotenv()

class LottoUpdater:
    """
    Hybrid Sniper V5: 지능형 로또 데이터 자동 업데이트 엔진
    - Gemini 1.5 Flash를 활용한 비정형 데이터 파싱
    - 구글 시트('로또 max') 자동 동기화
    - 실행 스케줄링 및 로깅 기능 포함
    """

    def __init__(self):
        self.creds_file = 'creds_lotto.json'
        self.sheet_name = '로또 max'
        self.log_sheet_name = 'Log'  # 로그 기록용 시트 탭 이름 (없으면 생성 고려)
        self.model_name = 'gemini-1.5-flash'

        # 1. Google Sheets 인증
        self.gc = self._authenticate_google_sheets()

        # 2. Gemini API 설정
        self._setup_gemini()

    def _authenticate_google_sheets(self):
        """구글 시트 API 인증을 처리합니다."""
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        if not os.path.exists(self.creds_file):
            raise FileNotFoundError(f"인증 파일({self.creds_file})을 찾을 수 없습니다. GitHub Secrets 또는 로컬 설정을 확인하세요.")

        creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, scope)
        return gspread.authorize(creds)

    def _setup_gemini(self):
        """Gemini 모델을 설정하고 사용 가능한지 확인합니다."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

        genai.configure(api_key=api_key)

        # 모델 탐색 및 설정 (Model Explorer)
        available_models = [m.name for m in genai.list_models()]
        target_model = f"models/{self.model_name}"

        if target_model in available_models:
            print(f"✅ [Model Explorer] {self.model_name} 모델이 사용 가능합니다.")
            self.model = genai.GenerativeModel(self.model_name)
        else:
            print(f"⚠️ [Model Explorer] {self.model_name}를 찾을 수 없습니다. 기본 모델로 대체합니다.")
            self.model = genai.GenerativeModel('gemini-pro') # Fallback

    def check_schedule(self):
        """
        schedule_config.json을 확인하여 현재 실행해야 할 타이밍인지 검증합니다.
        GitHub Actions에서는 보통 Cron으로 제어되지만, 이중 검증을 위해 유지합니다.
        """
        config_path = 'schedule_config.json'
        if not os.path.exists(config_path):
            print("⚠️ 설정 파일이 없습니다. 스케줄 검사를 건너뜁니다.")
            return True # 파일이 없으면 강제 실행 허용 (또는 False)

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if not config.get('active', True):
                print("⛔ 스케줄러가 비활성화되어 있습니다.")
                return False

            # 현재 시간 확인
            now = datetime.datetime.now()
            # 요일 확인 (0:월, ... 5:토, 6:일)
            # 설정 파일의 'Saturday' 등을 매핑
            target_day = config['schedule']['day_of_week']
            target_hour = config['schedule']['hour']
            window = config['schedule'].get('window_minutes', 60)

            days_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

            if now.weekday() != days_map.get(target_day, 5):
                print(f"⏳ 오늘은 {target_day}가 아닙니다. 실행을 건너뜁니다.")
                return False # 요일 불일치

            # 시간 범위 확인 (target_hour:00 ~ target_hour:window)
            start_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            end_time = start_time + datetime.timedelta(minutes=window)

            if start_time <= now <= end_time:
                print("✅ 스케줄 실행 시간 내에 있습니다.")
                return True
            else:
                print(f"⏳ 실행 시간이 아닙니다. (설정: {target_hour}시, 현재: {now.hour}시)")
                return False # 시간 불일치

        except Exception as e:
            print(f"⚠️ 스케줄 확인 중 오류 발생: {e}")
            return True # 오류 시 안전하게 실행 시도

    def get_latest_recorded_round(self):
        """구글 시트에서 마지막으로 저장된 회차 정보를 가져옵니다."""
        try:
            sh = self.gc.open(self.sheet_name)
            ws = sh.get_worksheet(0) # 첫 번째 시트 가정

            # A열(회차)의 값들을 가져옴
            col_values = ws.col_values(1)

            if not col_values or len(col_values) <= 1:
                return 0 # 헤더만 있거나 비어있음

            # 마지막 값이 숫자인지 확인하고 반환
            last_val = col_values[-1]
            try:
                return int(last_val.replace('회', '').strip())
            except:
                return 0
        except Exception as e:
            print(f"❌ 시트 데이터 조회 실패: {e}")
            return 0

    def get_current_expected_round(self):
        """
        오늘 날짜 기준으로 예상되는 최신 회차를 계산합니다.
        로또 1회: 2002-12-07
        """
        start_date = datetime.datetime(2002, 12, 7, 21, 0, 0) # 1회 추첨일
        now = datetime.datetime.now()

        diff = now - start_date
        weeks = diff.days // 7
        return weeks + 1

    def fetch_lotto_data_via_gemini(self, round_no):
        """
        Google/Naver 검색 결과를 크롤링하고, Gemini에게 파싱을 요청합니다.
        API를 사용하지 않고 비정형 텍스트에서 데이터를 추출하는 핵심 로직입니다.
        """
        print(f"🔍 {round_no}회 데이터 수집 시도 중 (Gemini Powered)...")

        # 1. 검색 결과 크롤링 (Naver 활용이 봇 탐지에 조금 더 유연할 수 있음, 또는 동행복권 사이트 직접 접근)
        # 여기서는 네이버 검색 결과를 타겟으로 함
        url = f"https://search.naver.com/search.naver?query=로또+{round_no}회+당첨번호"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # 페이지의 텍스트 콘텐츠 추출 (너무 길면 Gemini 토큰 낭비이므로 적당히 자름)
            # 로또 번호가 있는 영역 위주로 텍스트 추출하면 좋지만, 범용성을 위해 body 텍스트 사용
            page_text = soup.get_text()[:10000]

            # 2. Gemini에게 파싱 요청
            prompt = f"""
            아래 텍스트는 로또 {round_no}회 당첨 결과 검색 페이지의 내용이다.
            이 텍스트에서 다음 정보를 추출하여 정확한 JSON 형식으로만 응답해라.
            다른 말은 하지 말고 오직 JSON만 출력해라.

            필요한 필드:
            - drwNo: 회차 (정수)
            - drwtNo1: 번호1 (정수)
            - drwtNo2: 번호2 (정수)
            - drwtNo3: 번호3 (정수)
            - drwtNo4: 번호4 (정수)
            - drwtNo5: 번호5 (정수)
            - drwtNo6: 번호6 (정수)
            - bnusNo: 보너스번호 (정수)
            - firstAccumamnt: 1등 총 당첨금 (정수, 원 단위, '원'이나 콤마 제거)
            - firstPrzwnerCo: 1등 당첨자 수 (정수)
            - firstWinamnt: 1등 1인당 당첨금 (정수)
            - firstPrzwnerStore: 1등 당첨점 (문자열, 여러 곳일 경우 쉼표로 구분)
            - drwNoDate: 추첨일 (YYYY-MM-DD 형식 문자열)

            [텍스트 데이터]
            {page_text}
            """

            response = self.model.generate_content(prompt)

            # 응답 전처리 (Markdown 코드 블록 제거 등)
            result_text = response.text.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            data = json.loads(result_text)

            # 데이터 검증
            if int(data['drwNo']) != round_no:
                print(f"⚠️ 추출된 회차({data['drwNo']})가 요청 회차({round_no})와 다릅니다.")
                return None

            return data

        except Exception as e:
            print(f"❌ Gemini 파싱 실패 ({round_no}회): {e}")
            return None

    def update_sheet(self, data):
        """수집된 데이터를 구글 시트에 추가합니다."""
        try:
            sh = self.gc.open(self.sheet_name)
            ws = sh.get_worksheet(0)

            # 행 데이터 구성 (시트의 열 순서에 맞춰야 함)
            # 순서: 회차, 날짜, 번호1~6, 보너스, 1등당첨자수, 1등금액, 1등당첨점
            row = [
                data['drwNo'],
                data['drwNoDate'],
                data['drwtNo1'], data['drwtNo2'], data['drwtNo3'],
                data['drwtNo4'], data['drwtNo5'], data['drwtNo6'],
                data['bnusNo'],
                data['firstPrzwnerCo'],
                data['firstAccumamnt'],
                data.get('firstPrzwnerStore', '')
            ]

            ws.append_row(row)
            print(f"💾 {data['drwNo']}회 데이터 시트 저장 완료.")
            return True
        except Exception as e:
            print(f"❌ 시트 저장 실패: {e}")
            return False

    def log_execution(self, status, message):
        """실행 로그를 시트의 'Log' 탭에 기록합니다."""
        try:
            sh = self.gc.open(self.sheet_name)
            try:
                log_ws = sh.worksheet(self.log_sheet_name)
            except:
                # 로그 시트가 없으면 생성
                log_ws = sh.add_worksheet(title=self.log_sheet_name, rows=1000, cols=5)
                log_ws.append_row(["Timestamp", "Model", "Status", "Message"])

            log_ws.append_row([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.model_name,
                status,
                message
            ])
        except Exception as e:
            print(f"⚠️ 로그 기록 실패: {e}")

    def run(self, force=False):
        """전체 프로세스를 실행합니다."""
        print("🚀 Hybrid Sniper V5 데이터 엔진 시작...")

        # 1. 스케줄 확인
        if not force and not self.check_schedule():
            self.log_execution("SKIPPED", "스케줄 시간이 아니므로 실행을 건너뜁니다.")
            return

        # 2. 업데이트 필요 여부 확인
        last_round = self.get_latest_recorded_round()
        expected_round = self.get_current_expected_round()

        print(f"ℹ️ 마지막 저장 회차: {last_round}, 현재 예상 회차: {expected_round}")

        if last_round >= expected_round:
            print("✅ 모든 데이터가 최신 상태입니다.")
            self.log_execution("SUCCESS", "업데이트 할 데이터가 없습니다.")
            return

        # 3. 누락 데이터 순차 업데이트
        updated_count = 0
        for r in range(last_round + 1, expected_round + 1):
            data = self.fetch_lotto_data_via_gemini(r)

            if data:
                if self.update_sheet(data):
                    updated_count += 1
                else:
                    self.log_execution("ERROR", f"{r}회 시트 저장 실패")
            else:
                print(f"⚠️ {r}회 데이터를 가져오지 못했습니다. 다음 실행 때 재시도합니다.")
                self.log_execution("FAIL", f"{r}회 데이터 파싱 실패")
                break # 연속 실패 방지를 위해 중단할지, 계속할지 결정 (여기선 중단)

            # API 과부하 방지 및 사람처럼 보이기 위한 대기
            time.sleep(2)

        if updated_count > 0:
            self.log_execution("SUCCESS", f"총 {updated_count}개 회차 업데이트 완료 ({last_round+1} ~ {last_round+updated_count})")
        else:
            self.log_execution("INFO", "업데이트 시도했으나 성공한 건수 없음")

if __name__ == "__main__":
    # GitHub Actions 등에서 실행될 때 인자 처리 가능 (현재는 기본 실행)
    updater = LottoUpdater()

    # 기본적으로 스케줄 설정을 따름 (force=False)
    # 수동 실행이나 테스트 시에는 force=True로 변경하여 실행 가능
    is_manual_run = os.getenv("MANUAL_RUN", "false").lower() == "true"
    updater.run(force=is_manual_run)
