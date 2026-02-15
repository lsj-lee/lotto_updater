import os
import re
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. 환경 변수 로드 (로컬 테스트용, GitHub에서는 Secrets가 우선 적용됨)
load_dotenv()

# 주요 설정값
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDS_PATH = os.getenv("GOOGLE_CREDS_PATH", "creds.json")
SHEET_NAME = "로또 max"

# 제미나이 클라이언트 초기화
ai_client = genai.Client(api_key=GEMINI_API_KEY)

class LottoDataPipeline:
    def __init__(self):
        print("🚀 [파이프라인] 데이터 수집 엔진 가동...")
        self.scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        self.client = self.connect_google_sheet()
        self.spreadsheet = self.client.open(SHEET_NAME)
        self.sheet = self.spreadsheet.worksheet("시트1")

    def connect_google_sheet(self):
        """구글 시트 인증 및 연결"""
        try:
            # GitHub Secrets에서 생성된 creds.json 파일을 읽습니다.
            creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_PATH, self.scope)
            return gspread.authorize(creds)
        except Exception as e:
            print(f"❌ 구글 시트 인증 실패: {e}")
            raise

    def fetch_and_update(self):
        """최신 회차를 감지하고 AI 검색을 통해 시트를 업데이트합니다."""
        try:
            # A2 셀에서 마지막 회차 정보를 가져옵니다.
            last_draw_val = self.sheet.acell('A2').value
            # 숫자만 추출하여 다음 회차 계산
            target_draw = int(re.sub(r'[^0-9]', '', str(last_draw_val))) + 1
        except Exception as e:
            print(f"❌ 마지막 회차 읽기 실패: {e}")
            return

        print(f"📡 [AI 검색] {target_draw}회차 정보를 수집합니다...")

        # 429 에러(Quota Exhausted) 방지를 위한 5초 대기
        time.sleep(5)

        prompt = (
            f"한국 로또 {target_draw}회 당첨 번호 6개와 보너스 번호를 "
            f"'회차,번1,번2,번3,번4,번5,번6,보너스' 형식으로 숫자만 알려줘. "
            f"만약 아직 발표 전이라면 정확히 'WAIT'라고만 답해."
        )

        try:
            # 진단 결과 가용성이 확인된 2.5-flash 모델 사용
            response = ai_client.models.generate_content(
                model='models/gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
                )
            )
            
            result = response.text.strip()
            
            if "WAIT" in result:
                print(f"⏳ {target_draw}회차는 아직 발표 전입니다. 작업을 안전하게 종료합니다.")
                return

            # 정규표현식으로 숫자만 리스트로 추출
            numbers = [int(s) for s in re.findall(r'\d+', result)]
            
            if len(numbers) >= 8:
                # 2행에 새로운 데이터 삽입 (기존 데이터는 아래로 자동 밀림)
                self.sheet.insert_row(numbers[:8], 2)
                print(f"✅ {target_draw}회차 업데이트 성공: {numbers[1:7]} + 보너스 {numbers[7]}")
            else:
                print(f"⚠️ 데이터 형식 오류: {result}")
                
        except Exception as e:
            print(f"❌ AI 검색 중 오류 발생: {e}")

if __name__ == "__main__":
    try:
        pipeline = LottoDataPipeline()
        pipeline.fetch_and_update()
    except Exception as fatal_e:
        print(f"🚨 치명적 오류 발생: {fatal_e}")