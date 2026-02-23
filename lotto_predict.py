# -*- coding: utf-8 -*-
import os
import time
import gc
import random
import json
import datetime
import re
import multiprocessing
import sys
import traceback

# [필수 라이브러리]
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import gspread
from oauth2client.service_account import ServiceAccountCredentials

try:
    from google import genai
except ImportError:
    print("❌ Critical Dependency Missing: 'google-genai'")
    sys.exit(1)

load_dotenv()

# ==========================================
# ⚙️ [Configuration] 기지 좌표 및 설정
# ==========================================

SPREADSHEET_ID = '1lOifE_xRUocAY_Av-P67uBMKOV1BAb4mMwg_wde_tyA'
CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'
REC_SHEET_NAME = '추천번호'
STATE_FILE = 'hybrid_sniper_v5_state.pth'

# M5 하드웨어 안전장치 (6코어 제한)
USED_CORES = 6
torch.set_num_threads(USED_CORES)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"🚀 [System] M5 Neural Engine Activated (MPS/Metal). Cores: {USED_CORES}")
else:
    DEVICE = torch.device("cpu")

REAL_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Referer": "https://www.naver.com/"
}

# ==========================================
# 🧠 [Phase 2] The Brain Engine
# ==========================================

class NDA_FeatureEngine:
    @staticmethod
    def calculate_derived_features(numbers_list):
        features = []
        for nums in numbers_list:
            if len(nums) < 6:
                features.append([0,0,0,0])
                continue
            s = sum(nums)
            odd = sum(1 for n in nums if n % 2 != 0)
            high = sum(1 for n in nums if n >= 23)
            diffs = set()
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    diffs.add(nums[j] - nums[i])
            ac = len(diffs) - 5
            features.append([s/255.0, odd/6.0, high/6.0, ac/10.0])
        return np.array(features)

    @staticmethod
    def create_multimodal_dataset(data, lookback=10):
        X_seq, X_stat, y = [], [], []
        if len(data) <= lookback: return None, None, None
        raw_nums = np.array(data)
        derived = NDA_FeatureEngine.calculate_derived_features(data)
        for i in range(lookback, len(data)):
            X_seq.append(raw_nums[i-lookback:i] / 45.0)
            X_stat.append(derived[i-1])
            target = np.zeros(45)
            for n in raw_nums[i]: target[n-1] = 1
            y.append(target)
        return (torch.tensor(np.array(X_seq), dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(X_stat), dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(y), dtype=torch.float32).to(DEVICE))

class CreativeConnectionModel(nn.Module):
    def __init__(self):
        super(CreativeConnectionModel, self).__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.ln_a = nn.LayerNorm(128)
        self.stat_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.BatchNorm1d(32))
        self.head = nn.Sequential(nn.Linear(128 + 32, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 45), nn.Sigmoid())

    def forward(self, x_seq, x_stat):
        out_seq, _ = self.lstm(x_seq)
        out_seq = self.ln_a(out_seq[:, -1, :])
        out_stat = self.stat_net(x_stat)
        combined = torch.cat([out_seq, out_stat], dim=1)
        return self.head(combined)

# ==========================================
# 🛰️ [System] Orchestrator
# ==========================================

def get_verified_model(api_key):
    print("🛰️ [Scout] Scanning for Gemini Models...")
    if not api_key: return "gemini-1.5-flash"
    candidates = ["gemini-3-flash-preview", "gemini-2.0-flash-exp", "gemini-1.5-flash"]
    for model in candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        try:
            payload = {"contents": [{"parts": [{"text": "Ping"}]}]}
            resp = requests.post(url, json=payload, timeout=3)
            if resp.status_code == 200:
                print(f"   ✅ Active: {model}")
                return model
        except: continue
    return "gemini-1.5-flash"

class LottoOrchestrator:
    def __init__(self):
        self.gc = self._auth()
        api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = get_verified_model(api_key)
        try:
            self.client = genai.Client(api_key=api_key)
        except:
            self.client = None

    def _auth(self):
        # Docs 권한을 깔끔하게 제거하고 스프레드시트와 드라이브 권한만 유지합니다.
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive",
                 "https://www.googleapis.com/auth/spreadsheets"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
        return gspread.authorize(creds)

    def get_sheet(self):
        try:
            return self.gc.open_by_key(SPREADSHEET_ID)
        except:
            return self.gc.open(SHEET_NAME)

    def sync_data(self):
        print("\n🔄 [Phase 1] 지능형 네이버 동기화 시작...")
        try:
            sh = self.get_sheet()
            ws = sh.get_worksheet(0)
            
            # 내림차순 시트에서도 정확한 마지막 회차 찾기
            col1 = ws.col_values(1)
            rounds = []
            for val in col1:
                clean = str(val).replace(',', '').replace('회', '').replace('차', '').strip()
                if clean.isdigit(): rounds.append(int(clean))
            
            local_last = max(rounds) if rounds else 0
            portal_last = self._get_naver_latest_round()
            print(f"   📊 상태: 내 파일({local_last}회) vs 네이버({portal_last}회)")

            if portal_last > local_last:
                for r in range(local_last + 1, portal_last + 1):
                    print(f"   🔍 {r}회차 데이터 수집 중...")
                    data = self._scrape_round_detail(r)
                    if data:
                        row = [data['drwNo'], data['drwNoDate'], data['drwtNo1'], data['drwtNo2'], data['drwtNo3'],
                               data['drwtNo4'], data['drwtNo5'], data['drwtNo6'], data['bnusNo'],
                               data.get('firstPrzwnerCo', 0), data.get('firstAccumamnt', 0), ""]
                        # 내림차순이므로 2행(헤더 바로 아래)에 삽입
                        ws.insert_row(row, 2)
                        print(f"   ✅ {r}회차 저장 완료.")
                        time.sleep(2)
            else:
                print("   ✅ 이미 최신 상태입니다.")
        except Exception as e:
            print(f"❌ 동기화 중 오류: {e}")

    def _get_naver_latest_round(self):
        try:
            res = requests.get("https://search.naver.com/search.naver?query=로또", headers=REAL_BROWSER_HEADERS, timeout=5)
            m = re.search(r'(\d+)회차', res.text)
            return int(m.group(1)) if m else 0
        except: return 0

    def _scrape_round_detail(self, round_no):
        url = f"https://search.naver.com/search.naver?query=로또+{round_no}회+당첨번호"
        try:
            res = requests.get(url, headers=REAL_BROWSER_HEADERS, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            text = soup.get_text()[:5000]
            
            if self.client:
                prompt = f"Extract Lotto Round {round_no} data from text as JSON: {text}"
                try:
                    resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
                    return json.loads(resp.text.strip().replace('```json','').replace('```',''))
                except: pass

            # Regex Fallback
            nums = re.findall(r'\b(\d{1,2})\b', text)
            valid = [int(n) for n in nums if 1 <= int(n) <= 45]
            if len(valid) >= 7:
                return {"drwNo": round_no, "drwNoDate": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "drwtNo1": valid[0], "drwtNo2": valid[1], "drwtNo3": valid[2],
                        "drwtNo4": valid[3], "drwtNo5": valid[4], "drwtNo6": valid[5], "bnusNo": valid[6]}
            return None
        except: return None

    def train_brain(self):
        print("\n🧠 [Phase 2] 하이브리드 신경망 학습 (M5 가속)...")
        sh = self.get_sheet()
        ws = sh.get_worksheet(0)
        rows = ws.get_all_values()[1:]
        data = []
        for r in rows:
            try:
                nums = [int(str(x).replace(',', '')) for x in r[2:8]]
                data.append(nums)
            except: pass
        if len(data) < 50: return None, None
        X_seq, X_stat, y = NDA_FeatureEngine.create_multimodal_dataset(data, 10)
        model = CreativeConnectionModel().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.BCELoss()
        model.train()
        for e in range(100):
            opt.zero_grad()
            loss = crit(model(X_seq, X_stat), y)
            loss.backward()
            opt.step()
        torch.save(model.state_dict(), STATE_FILE)
        return model, data

    def generate_report(self, model, data):
        print("\n📝 [Phase 3] 전략 보고서 생성 (구글 시트 전용)...")
        model.eval()
        input_seq = torch.tensor(np.array(data[-10:]) / 45.0, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        input_stat = torch.tensor(NDA_FeatureEngine.calculate_derived_features([data[-1]]), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = model(input_seq, input_stat).cpu().numpy()[0]
        top_nums = [int(n+1) for n in probs.argsort()[::-1][:15]]
        games = [sorted(random.sample(top_nums, 6)) for _ in range(10)]
        self._write_sheet(games)

    def _write_sheet(self, games):
        sh = self.get_sheet()
        try: ws = sh.worksheet(REC_SHEET_NAME)
        except: ws = sh.add_worksheet(title=REC_SHEET_NAME, rows=100, cols=20)
        ws.clear()
        ws.update(range_name='A1', values=[['🏆 Sniper V5 최종 추천 번호']])
        ws.update(range_name='A3', values=[[f"시나리오 {i+1}"] + g for i, g in enumerate(games)])
        print("   ✅ 구글 시트 '추천번호' 탭에 작전 결과가 하달되었습니다.")

if __name__ == "__main__":
    app = LottoOrchestrator()
    print("🚀 Manual Mode: Executing Full Strategy...")
    app.sync_data()
    model, data = app.train_brain()
    if model and data: app.generate_report(model, data)
    print("\n✅ 작전 완료 (Mission Accomplished).")