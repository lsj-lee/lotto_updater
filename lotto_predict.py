import os
import time
import gc
import random
import json
import datetime
import re
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
import requests
import joblib
import sys

# [라이브러리 확인] Google GenAI SDK (v1.0+)
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ Critical Dependency Missing: 'google-genai'")
    print("💡 Run: pip install google-genai")
    sys.exit(1)

from bs4 import BeautifulSoup

# 환경 변수 로드
load_dotenv()

# ==========================================
# ⚙️ [Configuration] 사용자 설정
# ==========================================
# ⚠️ [중요] 스프레드시트 ID를 여기에 입력하거나 .env 파일에 'SPREADSHEET_ID'로 설정하세요.
# 브라우저 주소창의 https://docs.google.com/spreadsheets/d/THIS_LONG_STRING/edit 에서 복사
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "ENTER_YOUR_SPREADSHEET_ID_HERE")

CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'  # (백업용 이름)
REC_SHEET_NAME = '추천번호'
STATE_FILE = 'hybrid_sniper_v5_state.pth'

# [M5 Hardware Protection]
# Apple Silicon (MPS) 가속 사용, 코어 과열 방지
TOTAL_CORES = multiprocessing.cpu_count()
USED_CORES = 6
torch.set_num_threads(USED_CORES)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"🚀 [System] M5 Neural Engine Activated (MPS/Metal). Cores: {USED_CORES}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ [System] Running on CPU (MPS not found).")

# [Network] 위장 헤더
REAL_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.naver.com/"
}

# ==========================================
# 🧠 [Phase 2] The Brain: NDA & Hybrid Network
# ==========================================

class NDA_FeatureEngine:
    """
    [NDA] 다차원 데이터 분석 엔진
    - 시계열(Time-Series) + 통계적(Statistical) + 관계적(Relational) 데이터 생성
    """
    @staticmethod
    def calculate_derived_features(numbers_list):
        """논리 특성 레이어: 합계, 홀짝비, 고저비, AC지수"""
        features = []
        for nums in numbers_list:
            # 1. Sum (총합) -> 정규화 (보통 100~200 사이)
            s = sum(nums)
            # 2. Odd/Even (홀짝) -> 홀수 개수 (0~6)
            odd = sum(1 for n in nums if n % 2 != 0)
            # 3. High/Low (고저: 23이상) -> 고번호 개수 (0~6)
            high = sum(1 for n in nums if n >= 23)
            # 4. AC Index (복잡도)
            diffs = set()
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    diffs.add(nums[j] - nums[i])
            ac = len(diffs) - (6 - 1)

            features.append([s/255.0, odd/6.0, high/6.0, ac/10.0])
        return np.array(features)

    @staticmethod
    def create_multimodal_dataset(data, lookback=10):
        """
        데이터를 3가지 브랜치(Branch A, B, C) 입력 형태로 변환
        """
        X_seq, X_stat, y = [], [], []
        if len(data) <= lookback: return None, None, None

        # 기본 숫자 데이터 (1~45)
        raw_nums = np.array(data)
        # 파생 특성 데이터
        derived = NDA_FeatureEngine.calculate_derived_features(data)

        for i in range(lookback, len(data)):
            # Branch A input: 시계열 (Lookback 주차의 번호 흐름)
            # (Batch, Lookback, 6)
            seq = raw_nums[i-lookback:i]
            # 정규화 (1~45 -> 0~1)
            X_seq.append(seq / 45.0)

            # Branch B input: 통계적 특성 (직전 회차의 파생 변수)
            # (Batch, 4)
            stat = derived[i-1]
            X_stat.append(stat)

            # Target: 이번 회차 번호 (Multi-hot encoding for Classification)
            target = np.zeros(45)
            for n in raw_nums[i]:
                target[n-1] = 1
            y.append(target)

        return (
            torch.tensor(np.array(X_seq), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(X_stat), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(y), dtype=torch.float32).to(DEVICE)
        )

class CreativeConnectionModel(nn.Module):
    """
    [CC] 멀티-헤드 하이브리드 신경망 (Phase 2 Core)
    - Branch A (LSTM): 시계열 패턴 학습
    - Branch B (Dense): 통계적/논리적 특성 학습
    - Decision Head: 통합 추론
    """
    def __init__(self):
        super(CreativeConnectionModel, self).__init__()

        # Branch A: Time-Series (LSTM)
        # Input: (Batch, Lookback, 6)
        self.lstm = nn.LSTM(input_size=6, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.ln_a = nn.LayerNorm(128)

        # Branch B: Statistical Features (TabNet-style Dense)
        # Input: (Batch, 4)
        self.stat_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32)
        )

        # Decision Head (Fusion)
        # LSTM(128) + Stat(32) = 160
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 45), # 45개 번호에 대한 Logits
            nn.Sigmoid()        # 확률 (0~1)
        )

    def forward(self, x_seq, x_stat):
        # Branch A
        # LSTM output: (Batch, Lookback, Hidden)
        out_seq, _ = self.lstm(x_seq)
        # Take the last time step's hidden state
        out_seq = self.ln_a(out_seq[:, -1, :])

        # Branch B
        out_stat = self.stat_net(x_stat)

        # Fusion
        combined = torch.cat([out_seq, out_stat], dim=1)
        output = self.head(combined)
        return output

# ==========================================
# 🛰️ [System] Scout & Interface
# ==========================================

def get_verified_model(api_key):
    """Scout: Finds the best available Gemini model"""
    print("🛰️ [Scout] Scanning for Gemini Models...")
    if not api_key: return None

    candidates = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]
    best_model = None

    for model in candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        try:
            # Simple Ping
            payload = {"contents": [{"parts": [{"text": "Ping"}]}]}
            resp = requests.post(url, json=payload, timeout=3)
            if resp.status_code == 200:
                print(f"   ✅ Active: {model}")
                return model # Return first active one
        except: continue

    return "gemini-1.5-flash" # Fallback

class LottoOrchestrator:
    def __init__(self):
        self.creds_file = CREDS_FILE
        self.gc, self.docs = self._auth()

        api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = get_verified_model(api_key)
        try:
            self.client = genai.Client(api_key=api_key)
        except:
            self.client = None
            print("⚠️ GenAI Client Init Failed (Manual Mode)")

    def _auth(self):
        # [Phase 3] Scope Update for Drive/Docs/Sheets
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/spreadsheets"
        ]
        if not os.path.exists(self.creds_file):
             print(f"❌ Credential file '{self.creds_file}' not found.")
             sys.exit(1)

        creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, scope)
        gc = gspread.authorize(creds)
        try:
            docs = build('docs', 'v1', credentials=creds)
        except:
            docs = None
        return gc, docs

    def get_sheet(self):
        """Open sheet by ID (Priority) or Name (Fallback)"""
        try:
            if SPREADSHEET_ID and "ENTER" not in SPREADSHEET_ID:
                return self.gc.open_by_key(SPREADSHEET_ID)
            else:
                print(f"⚠️ Warning: SPREADSHEET_ID not set. Trying name '{SHEET_NAME}'...")
                return self.gc.open(SHEET_NAME)
        except Exception as e:
            print(f"❌ Spreadsheet Connection Failed: {e}")
            print("💡 Solution: Set 'SPREADSHEET_ID' in the code to your file's ID.")
            sys.exit(1)

    # --- [Phase 1] Intelligent Sync ---
    def sync_data(self):
        print("\n🔄 [Phase 1] Executing Intelligent Naver Sync...")
        try:
            sh = self.get_sheet()
            ws = sh.get_worksheet(0)

            # Local Last Round
            try:
                col1 = ws.col_values(1)
                if len(col1) > 1:
                    local_last = int(str(col1[-1]).replace(',', '').replace('회', ''))
                else:
                    local_last = 0
            except: local_last = 0

            # Portal Last Round
            portal_last = self._get_naver_latest_round()
            print(f"   📊 Status: Local({local_last}) vs Portal({portal_last})")

            if portal_last > local_last:
                for r in range(local_last + 1, portal_last + 1):
                    data = self._scrape_round_detail(r)
                    if data:
                        row = [
                            data['drwNo'], data['drwNoDate'],
                            data['drwtNo1'], data['drwtNo2'], data['drwtNo3'],
                            data['drwtNo4'], data['drwtNo5'], data['drwtNo6'],
                            data['bnusNo'], data.get('firstPrzwnerCo',0), data.get('firstAccumamnt',0), ""
                        ]
                        ws.append_row(row)
                        print(f"   ✅ Synced Round {r}")
                        time.sleep(2)
            else:
                print("   ✅ Already Up-to-Date.")

        except Exception as e:
            print(f"❌ Sync Error: {e}")

    def _get_naver_latest_round(self):
        url = "https://search.naver.com/search.naver?query=로또"
        try:
            res = requests.get(url, headers=REAL_BROWSER_HEADERS, timeout=5)
            # Regex for "1234회차"
            m = re.search(r'(\d+)회차', res.text)
            if m: return int(m.group(1))
            return 0
        except: return 0

    def _scrape_round_detail(self, round_no):
        """Naver Search -> Gemini Parse -> Regex Fallback"""
        url = f"https://search.naver.com/search.naver?query=로또+{round_no}회+당첨번호"
        text = ""
        try:
            res = requests.get(url, headers=REAL_BROWSER_HEADERS, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            text = soup.get_text()[:5000]

            # 1. AI Parsing
            if self.client:
                prompt = f"Extract Lotto {round_no} numbers from text. JSON format: {{'drwNo': {round_no}, 'drwNoDate': 'YYYY-MM-DD', 'drwtNo1':.., 'bnusNo':.., 'firstPrzwnerCo': 0, 'firstAccumamnt': 0}}. Text: {text}"
                try:
                    resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
                    js_str = resp.text.replace('```json','').replace('```','')
                    js = json.loads(js_str)
                    if js.get('drwtNo1') and js['drwtNo1'] > 0: return js
                except: pass

            # 2. Regex Fallback
            print(f"   ⚠️ AI Parsing Failed for {round_no}. Engaging Regex...")
            # Pattern: Try to find sequence of 6 numbers + 1 bonus
            # This is a basic fallback for '1, 2, 3, 4, 5, 6 + 7' patterns often found in text
            nums = re.findall(r'\b(\d{1,2})\b', text)
            nums = [int(n) for n in nums if 1 <= int(n) <= 45]

            # 휴리스틱: 1~45 사이 숫자가 7개 이상 연속으로 나오거나 근처에 모여있으면 로또 번호로 추정
            # 정확도가 낮을 수 있으므로 실패 처리하거나 아주 엄격하게 체크
            # 여기서는 안전을 위해 None 반환 (잘못된 데이터 입력 방지)
            return None

        except: return None

    # --- [Phase 2] Training ---
    def train_brain(self):
        print("\n🧠 [Phase 2] Training Hybrid Neural Network (M5/MPS)...")
        sh = self.get_sheet()
        ws = sh.get_worksheet(0)
        rows = ws.get_all_values()[1:]

        data = []
        for r in rows:
            try:
                # 번호가 있는 열만 추출 (보통 C~H열, 즉 인덱스 2~7 + 보너스 8)
                # 여기서는 1~6번 공만 사용 (인덱스 2~7)
                nums = [int(x.replace(',','')) for x in r[2:8]]
                data.append(nums)
            except: pass

        if len(data) < 50:
            print("❌ Not enough data to train.")
            return None, None

        # Prepare Data
        X_seq, X_stat, y = NDA_FeatureEngine.create_multimodal_dataset(data, lookback=10)
        if X_seq is None: return None, None

        # Model Setup
        model = CreativeConnectionModel().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.BCELoss()

        # Training Loop (with Progress)
        model.train()
        epochs = 100
        print(f"   🔥 Ignite: Training {epochs} epochs on {DEVICE}...")

        for e in range(epochs):
            opt.zero_grad()
            out = model(X_seq, X_stat)
            loss = crit(out, y)
            loss.backward()
            opt.step()

            if (e+1) % 20 == 0:
                print(f"   Epoch {e+1}/{epochs} | Loss: {loss.item():.4f}")

        # Save Weights (IW)
        torch.save(model.state_dict(), STATE_FILE)
        print("   ✅ Brain Saved.")

        return model, data

    # --- [Phase 3] Prediction & Report ---
    def generate_report(self, model, data):
        if not model: return
        print("\n📝 [Phase 3] Generative Strategy Reporting...")
        model.eval()

        # Predict Next
        # Input for prediction is the LAST 10 weeks of data
        input_raw = data[-10:]
        input_seq = torch.tensor(np.array(input_raw) / 45.0, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Stat input is based on the VERY LAST week
        last_stat = NDA_FeatureEngine.calculate_derived_features([data[-1]])
        input_stat = torch.tensor(last_stat, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            probs = model(input_seq, input_stat).cpu().numpy()[0]

        # Select Top 15
        top_indices = probs.argsort()[::-1][:15]
        top_nums = [int(n+1) for n in top_indices] # 1-based, int conversion
        print(f"   🎯 Target Lock: {top_nums}")

        # Generate 10 Games (Intelligent Combination)
        games = []
        for _ in range(10):
            # Weighted random choice from top 15
            # Or just simple random for variety
            g = sorted(random.sample(top_nums, 6))
            games.append(g)

        # Write to Docs & Sheet
        self._write_docs(games, top_nums)
        self._write_sheet(games)

    def _write_docs(self, games, candidates):
        if not self.docs: return
        print("   📄 Writing to Google Docs...")

        doc_title = f"Sniper V5 Report - {datetime.date.today()}"
        try:
            body = {'title': doc_title}
            doc = self.docs.documents().create(body=body).execute()
            doc_id = doc['documentId']

            content = f"[Sniper V5 Strategic Report]\nDate: {datetime.date.today()}\n"
            content += f"Target Candidates (Top 15): {candidates}\n\n"
            content += "[Tactical Combinations]\n"
            for i, g in enumerate(games):
                content += f"Scenario {i+1}: {g}\n"
            content += "\n[End of Report]"

            reqs = [{'insertText': {'location': {'index': 1}, 'text': content}}]
            self.docs.documents().batchUpdate(documentId=doc_id, body={'requests': reqs}).execute()
            print(f"   ✅ Report URL: https://docs.google.com/document/d/{doc_id}")
        except Exception as e:
            print(f"   ⚠️ Docs Error: {e}")

    def _write_sheet(self, games):
        try:
            sh = self.get_sheet()
            # Try to get or create recommendation sheet
            try:
                ws = sh.worksheet(REC_SHEET_NAME)
            except:
                ws = sh.add_worksheet(title=REC_SHEET_NAME, rows=100, cols=20)

            ws.clear()
            ws.update(range_name='A1', values=[['🏆 Sniper V5 Generated Games']])

            rows = []
            for i, g in enumerate(games):
                rows.append([f"Scenario {i+1}"] + g)

            ws.update(range_name='A3', values=rows)
            print("   ✅ Google Sheet Updated.")
        except Exception as e:
            print(f"⚠️ Sheet Write Error: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    app = LottoOrchestrator()

    # Check for Scheduled Mode (GitHub Actions / Cron)
    if "--scheduled" in sys.argv:
        day = datetime.datetime.now().strftime("%a")
        print(f"🗓️ Scheduled Mode: Today is {day}")

        if day == "Sun":
            # Sunday: Only Sync Data
            app.sync_data()
        elif day == "Mon":
            # Monday: Weekly Analysis (Training)
            app.train_brain()
        elif day == "Wed":
            # Wednesday: Final Prediction & Report
            # (Need to train first to get model)
            model, data = app.train_brain()
            if model and data:
                app.generate_report(model, data)
        else:
            print("💤 No scheduled mission for today.")

    else:
        # Default: Full Cycle (Manual Execution)
        print("🚀 Manual Mode: Executing Full Strategy...")

        # 1. Sync
        app.sync_data()

        # 2. Train
        model, data = app.train_brain()

        # 3. Report
        if model and data:
            app.generate_report(model, data)

    print("\n✅ Mission Accomplished.")
