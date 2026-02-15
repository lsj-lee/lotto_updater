import os
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google import genai
from google.genai import types
from dotenv import load_dotenv

# .env 파일에 숨겨진 보안 정보 로드
load_dotenv()

# 환경 변수에서 보안 정보 가져오기
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDS_PATH = os.getenv("GOOGLE_CREDS_PATH")
SHEET_NAME = "로또 max"

# 제미나이 클라이언트 설정
ai_client = genai.Client(api_key=GEMINI_API_KEY)

class LottoDataPipeline:
    def __init__(self):
        print("🚀 [파이프라인] 데이터 수집 엔진 가동...")
        self.scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        self.client = self.connect_google_sheet()
        self.spreadsheet = self.client.open(SHEET_NAME)
        self.sheet = self.spreadsheet.worksheet("시트1")

    def connect_google_sheet(self):
        # 보안을 위해 로컬 경로에서 인증키 로드
        creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_PATH, self.scope)
        return gspread.authorize(creds)

    def get_target_draw(self):
        # A2 셀(최신 회차)을 읽어서 다음 회차 번호 계산
        last_draw_val = self.sheet.acell('A2').value
        return int(re.sub(r'[^0-9]', '', str(last_draw_val))) + 1

    def fetch_and_update(self):
        target_draw = self.get_target_draw()
        print(f"📡 [AI 검색] {target_draw}회차 정보를 구글 실시간 검색으로 수집합니다...")

        prompt = f"한국 로또 {target_draw}회 당첨 번호 6개와 보너스 번호를 '회차,번1,번2,번3,번4,번5,번6,보너스' 형식으로 숫자만 콤마로 구분해서 알려줘. 아직 발표 전이면 'WAIT'라고 답해."

        try:
            # 실시간 검색 도구 활성화
            response = ai_client.models.generate_content(
                model='gemini-2.0-flash', # 혹은 상진님이 사용 가능한 최신 모델
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
                )
            )
            result = response.text.strip()

            if "WAIT" in result:
                print(f"⏳ {target_draw}회차는 아직 발표 전입니다.")
                return

            numbers = [int(s) for s in re.findall(r'\d+', result)]
            if len(numbers) >= 8:
                # 2행에 삽입하여 최신 데이터가 위로 오게 함
                self.sheet.insert_row(numbers[:8], 2)
                print(f"✅ {target_draw}회차 업데이트 성공!")
            else:
                print("⚠️ 데이터 형식이 올바르지 않습니다.")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    pipeline = LottoDataPipeline()
    pipeline.fetch_and_update()