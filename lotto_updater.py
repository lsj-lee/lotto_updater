import os
import re
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDS_PATH = os.getenv("GOOGLE_CREDS_PATH", "creds.json")
SHEET_NAME = "로또 max"

ai_client = genai.Client(api_key=GEMINI_API_KEY)

class LottoDataPipeline:
    def __init__(self):
        print("🚀 [파이프라인] 데이터 동기화 엔진 가동...")
        self.scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        self.client = self.connect_google_sheet()
        self.spreadsheet = self.client.open(SHEET_NAME)
        self.sheet = self.spreadsheet.worksheet("시트1")

    def connect_google_sheet(self):
        creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_PATH, self.scope)
        return gspread.authorize(creds)

    def fetch_and_update(self):
        # 최신 회차까지 무한 반복 (WAIT가 나올 때까지)
        while True:
            last_draw_val = self.sheet.acell('A2').value
            target_draw = int(re.sub(r'[^0-9]', '', str(last_draw_val))) + 1
            
            print(f"📡 [AI 검색] {target_draw}회차 수집 시도 중...")
            
            # 429 에러 방지를 위한 필수 지연 (무료 쿼터 보호)
            time.sleep(10) 

            prompt = f"한국 로또 {target_draw}회 당첨 번호 6개와 보너스 번호를 '회차,번1,번2,번3,번4,번5,번6,보너스' 형식으로 숫자만 알려줘. 발표 전이면 'WAIT'라고 답해."

            try:
                response = ai_client.models.generate_content(
                    model='models/gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(tools=[{"google_search": {}}])
                )
                
                result = response.text.strip()
                if "WAIT" in result:
                    print(f"🏁 최신 회차({target_draw-1}회)까지 동기화가 완료되었습니다.")
                    break

                numbers = [int(s) for s in re.findall(r'\d+', result)]
                if len(numbers) >= 8:
                    self.sheet.insert_row(numbers[:8], 2)
                    print(f"✅ {target_draw}회차 동기화 성공: {numbers[1:7]}")
                else:
                    print(f"⚠️ 형식 오류로 중단합니다: {result}")
                    break
            except Exception as e:
                if "429" in str(e):
                    print("🛑 API 할당량이 소진되었습니다. 나머지는 다음에 자동으로 업데이트됩니다.")
                else:
                    print(f"❌ 오류 발생: {e}")
                break

if __name__ == "__main__":
    LottoDataPipeline().fetch_and_update()
