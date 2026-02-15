import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google import genai
from google.genai import types
import pandas as pd
import numpy as np
import re
import datetime

# ==========================================
# 1. 핵심 설정 
# ==========================================
GEMINI_API_KEY = "AIzaSyCOX9mBuPBkcX_sL61mtaI1ZmgbB5Mo3rU" 
GOOGLE_CREDS_PATH = "/Users/lsj/Desktop/구글 연결 키/creds.json"
SHEET_NAME = "로또 max"

ai_client = genai.Client(api_key=GEMINI_API_KEY)

class LottoMaxAIV2:
    def __init__(self):
        print("🚀 [시스템] 로또 MAX 2.5 엔진 부팅 중 (구글 검색 + 최상단 기록 모드)...")
        self.scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        self.client = self.connect_google_sheet()
        self.spreadsheet = self.client.open(SHEET_NAME)
        
        self.df = self.get_data_from_sheet()
        self.weights = {"NDA": 0.8, "TE": 0.2}
        self.past_winners = [set(map(int, row)) for row in self.df.iloc[:, 1:7].values]

    def connect_google_sheet(self):
        creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_PATH, self.scope)
        return gspread.authorize(creds)

    def get_data_from_sheet(self):
        sheet = self.spreadsheet.worksheet("시트1")
        data = sheet.get_all_values()
        
        if len(data) < 2:
            raise ValueError("❌ 시트에 데이터가 없습니다.")

        cleaned_rows = []
        for row in data[1:]:
            new_row = []
            for item in row[:8]:
                num_str = re.sub(r'[^0-9]', '', str(item)) 
                new_row.append(int(num_str) if num_str else np.nan)
            cleaned_rows.append(new_row)
            
        columns = ['회차', '번1', '번2', '번3', '번4', '번5', '번6', '보너스']
        df = pd.DataFrame(cleaned_rows, columns=columns)
        
        # 내부 계산을 위해 판다스 데이터프레임은 오름차순(과거->최신)으로 정렬해 둠
        df = df.dropna(subset=['회차'])
        return df.sort_values(by='회차', ascending=True).reset_index(drop=True)

    # ------------------------------------------
    # 2. 제미나이 실시간 구글 검색(Grounding) 수집
    # ------------------------------------------
    def auto_fetch_latest(self):
        last_draw = int(self.df.iloc[-1, 0])
        target_draw = last_draw + 1
        print(f"📡 [AI 검색] 제미나이가 구글 실시간 검색으로 {target_draw}회차 결과를 찾아옵니다...")

        prompt = f"지금 당장 구글 인터넷을 검색해서 한국 로또 {target_draw}회 당첨 번호 6개와 보너스 번호를 찾아줘. 반드시 '회차,번1,번2,번3,번4,번5,번6,보너스' 형식으로 숫자만 콤마로 구분해서 알려줘. 만약 검색해도 아직 발표 전이라면 'WAIT'라고 답해."
        
        try:
            response = ai_client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}] 
                )
            )
            result = response.text.strip()

            if "WAIT" in result or len(result) < 5:
                print(f"⏳ {target_draw}회차는 아직 추첨 전이거나 AI가 확신하지 못했습니다.")
                return False

            numbers = [int(s) for s in re.findall(r'\d+', result)]
            if len(numbers) >= 8:
                sheet = self.spreadsheet.worksheet("시트1")
                # 💡 핵심 변경점: append_row 대신 insert_row(데이터, 삽입할 행 번호) 사용
                # 1행은 열 제목이므로, 2행에 새 데이터를 밀어 넣고 기존 데이터는 아래로 내림
                sheet.insert_row(numbers[:8], 2)
                print(f"✅ [업데이트 완료] 제미나이가 {target_draw}회차를 찾아 시트 최상단(2행)에 기록했습니다: {numbers[1:7]}")
                self.df = self.get_data_from_sheet() 
                return True
        except Exception as e:
            if "429" in str(e):
                print("⚠️ [안내] 1분당 요청 횟수 제한(쿨타임)입니다. 1분 뒤에 다시 실행해 주세요.")
            else:
                print(f"⚠️ AI 검색 스킵 (사유: {e})")
        return False

    # ------------------------------------------
    # 3. 전수 백테스팅 & 마스터 번호 추출
    # ------------------------------------------
    def run_process(self):
        print(f"🕵️ [전수 백테스팅] 1회부터 {int(self.df.iloc[-1,0])}회까지의 역사를 복습 중입니다...")
        for i in range(100, len(self.df)):
            past = self.df.iloc[:i]
            actual = set(self.df.iloc[i, 1:7].astype(int).values)
            gaps = []
            for num in actual:
                found = past[past.iloc[:, 1:7].eq(num).any(axis=1)].index
                gap = i - found[-1] if len(found) > 0 else 50
                gaps.append(gap)
            
            avg_gap = np.mean(gaps)
            lr = 0.005
            if avg_gap > 12:
                self.weights["TE"] = min(0.8, self.weights["TE"] + lr)
                self.weights["NDA"] = max(0.2, self.weights["NDA"] - lr)
            else:
                self.weights["NDA"] = min(0.8, self.weights["NDA"] + lr)
                self.weights["TE"] = max(0.2, self.weights["TE"] - lr)
        
        scores = np.zeros(46)
        recent_15 = self.df.tail(15)
        for n in range(1, 46):
            m = recent_15.iloc[:, 1:7].eq(n).sum().sum() / 15
            found = self.df[self.df.iloc[:, 1:7].eq(n).any(axis=1)].index
            t = (len(self.df) - found[-1]) if len(found) > 0 else 50
            scores[n] = (m * self.weights["NDA"]) + ((t/20) * self.weights["TE"]) + 0.1

        master_pool = np.argsort(scores)[-20:].tolist()
        master_pool.sort()
        print(f"🎯 1차 도출 마스터 번호(20개): {master_pool}")
        
        final_sets = []
        while len(final_sets) < 10:
            candidate = sorted(np.random.choice(master_pool, 6, replace=False).tolist())
            if set(candidate) not in self.past_winners:
                if 2 <= sum(1 for n in candidate if n % 2 != 0) <= 4:
                    final_sets.append(candidate)

        self.save_to_sheet(final_sets, master_pool)
        return master_pool, final_sets

    def save_to_sheet(self, sets, pool):
        target = self.spreadsheet.worksheet("추천번호")
        target.clear()
        target.append_row(["💎 로또 MAX 2.5 (제미나이 실시간 검색 탑재)"])
        target.append_row(["최종 가중치", f"NDA: {self.weights['NDA']:.2f}", f"TE: {self.weights['TE']:.2f}"])
        target.append_row(["마스터 번호 (20개)", str(pool)])
        target.append_row([])
        target.append_row(["세트", "번1", "번2", "번3", "번4", "번5", "번6", "분석완료시간"])
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        for i, s in enumerate(sets, 1):
            target.append_row([f"{i}세트"] + s + [now])
        print("✅ 구글 시트 '추천번호' 탭에 결과가 완벽하게 전송되었습니다!")

if __name__ == "__main__":
    try:
        engine = LottoMaxAIV2()
        engine.auto_fetch_latest()  
        pool, sets = engine.run_process() 
        print("\n🚀 [로또 MAX 분석 리포트]")
        for i, s in enumerate(sets, 1):
            print(f"세트 {i:02d}: {s}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")