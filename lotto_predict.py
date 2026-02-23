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
LOG_SHEET_NAME = 'Log'
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
# 🧠 [Phase 2] The Brain Engine (Model Architecture)
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
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive",
                 "https://www.googleapis.com/auth/spreadsheets"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
        return gspread.authorize(creds)

    def get_sheet(self):
        try:
            return self.gc.open_by_key(SPREADSHEET_ID)
        except:
            return self.gc.open(SHEET_NAME)

    # -------------------------------------------------------------------------
    # 🔄 [Phase 1] Data Sync (일요일 02:00)
    # -------------------------------------------------------------------------
    def sync_data(self):
        print("\n🔄 [Phase 1] 지능형 네이버 동기화 시작...")
        try:
            sh = self.get_sheet()
            ws = sh.get_worksheet(0)
            
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

            nums = re.findall(r'\b(\d{1,2})\b', text)
            valid = [int(n) for n in nums if 1 <= int(n) <= 45]
            if len(valid) >= 7:
                return {"drwNo": round_no, "drwNoDate": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "drwtNo1": valid[0], "drwtNo2": valid[1], "drwtNo3": valid[2],
                        "drwtNo4": valid[3], "drwtNo5": valid[4], "drwtNo6": valid[5], "bnusNo": valid[6]}
            return None
        except: return None

    # -------------------------------------------------------------------------
    # 📥 [Helper] Data Loading
    # -------------------------------------------------------------------------
    def load_data(self):
        """구글 시트에서 전체 데이터를 로드하고 시간 순서(과거->현재)로 정렬하여 반환"""
        try:
            sh = self.get_sheet()
            ws = sh.get_worksheet(0)
            rows = ws.get_all_values()[1:] # 헤더 제외
            data = []
            for r in rows:
                try:
                    nums = [int(str(x).replace(',', '')) for x in r[2:8]]
                    data.append(nums)
                except: pass

            # 시트가 최신순(내림차순)이면, 학습을 위해 과거->현재로 뒤집음
            data.reverse()
            return data
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return []

    # -------------------------------------------------------------------------
    # 🧠 [Phase 2] Model Training (월요일 02:00)
    # -------------------------------------------------------------------------
    def train_brain(self):
        print("\n🧠 [Phase 2] 하이브리드 신경망 학습 (M5 가속)...")
        data = self.load_data()

        if len(data) < 50:
            print("⚠️ 학습 데이터가 부족합니다 (최소 50회차).")
            return None

        # 데이터셋 생성
        X_seq, X_stat, y = NDA_FeatureEngine.create_multimodal_dataset(data, 10)

        # 모델 초기화
        model = CreativeConnectionModel().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.BCELoss()

        # 학습 루프
        model.train()
        for e in range(100): # 100 Epochs
            opt.zero_grad()
            loss = crit(model(X_seq, X_stat), y)
            loss.backward()
            opt.step()
            if (e+1) % 20 == 0:
                print(f"   Epoch {e+1}/100 - Loss: {loss.item():.4f}")

        # 모델 저장 (가중치 파일 생성)
        torch.save(model.state_dict(), STATE_FILE)
        print(f"💾 모델 학습 완료 및 저장됨: {STATE_FILE}")
        return model # 학습된 모델 반환 (즉시 사용 시)

    # -------------------------------------------------------------------------
    # 🔮 [Phase 3] Prediction Only (수요일 02:00)
    # -------------------------------------------------------------------------
    def load_and_predict(self):
        """
        [분리된 예측 기능]
        학습 없이 저장된 가중치(pth)를 불러와 예측만 수행합니다.
        """
        print("\n🔮 [Phase 3] 저장된 두뇌를 깨워 예측을 시작합니다...")

        # 1. 데이터 로드 (최신 데이터를 입력값으로 사용)
        data = self.load_data()
        if not data:
            print("❌ 예측할 데이터가 없습니다.")
            return

        # 2. 가중치 파일 존재 여부 확인
        if not os.path.exists(STATE_FILE):
            print(f"❌ 학습된 모델 파일({STATE_FILE})이 없습니다. Phase 2(학습)를 먼저 실행하세요.")
            return

        # 3. 모델 구조 생성 및 가중치 로드
        try:
            model = CreativeConnectionModel().to(DEVICE)
            # map_location을 사용하여 저장된 장치와 무관하게 로드 (안전성 확보)
            model.load_state_dict(torch.load(STATE_FILE, map_location=DEVICE))
            model.eval() # 평가 모드 전환
            print("   ✅ 모델 로드 성공.")
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}")
            return

        # 4. 예측 수행 및 보고서 생성
        self.generate_report(model, data)

    def generate_report(self, model, data):
        print("📝 전략 보고서 작성 중...")
        # 최근 10회차 데이터를 입력 시퀀스로 사용
        last_seq = data[-10:]
        if len(last_seq) < 10:
            print("⚠️ 예측을 위한 최근 데이터가 부족합니다.")
            return

        input_seq = torch.tensor(np.array(last_seq) / 45.0, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        # 마지막 회차의 통계적 특징 추출
        input_stat = torch.tensor(NDA_FeatureEngine.calculate_derived_features([data[-1]]), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            probs = model(input_seq, input_stat).cpu().numpy()[0]

        # 확률 상위 15개 번호 추출 (후보군)
        top_nums = [int(n+1) for n in probs.argsort()[::-1][:15]]

        # 10개 게임 생성 (랜덤 조합)
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

    # -------------------------------------------------------------------------
    # 🏅 [Phase 4] Reward System (목요일 02:00)
    # -------------------------------------------------------------------------
    def evaluate_performance(self):
        print("\n🏅 [Phase 4] 지난 작전 성과 평가 (Reward Check)...")
        try:
            sh = self.get_sheet()
            ws_main = sh.get_worksheet(0)
            latest_row = ws_main.row_values(2)
            real_round = int(latest_row[0].replace('회', ''))
            real_nums = set([int(x) for x in latest_row[2:8]])
            bonus_num = int(latest_row[8])
            print(f"   🎯 실제 결과 ({real_round}회): {sorted(list(real_nums))} + {bonus_num}")

            try: ws_rec = sh.worksheet(REC_SHEET_NAME)
            except:
                print("   ⚠️ 추천 번호 시트가 없습니다. 평가 건너뜀.")
                return

            rec_rows = ws_rec.get_all_values()
            predictions = []
            for r in rec_rows:
                if r and "시나리오" in r[0]:
                    try:
                        nums = set([int(x) for x in r[1:7] if x])
                        predictions.append(nums)
                    except: pass

            if not predictions:
                print("   ⚠️ 평가할 추천 번호가 없습니다.")
                return

            total_hits = 0
            max_hit = 0
            results = []

            for idx, pred in enumerate(predictions):
                hit_cnt = len(real_nums.intersection(pred))
                is_bonus = bonus_num in pred
                rank = "낙첨"

                if hit_cnt == 6: rank = "1등"
                elif hit_cnt == 5 and is_bonus: rank = "2등"
                elif hit_cnt == 5: rank = "3등"
                elif hit_cnt == 4: rank = "4등"
                elif hit_cnt == 3: rank = "5등"

                total_hits += hit_cnt
                if hit_cnt > max_hit: max_hit = hit_cnt
                results.append(f"시나리오 {idx+1}: {hit_cnt}개 일치 ({rank})")

            avg_hit = total_hits / len(predictions)
            self._log_reward(real_round, max_hit, avg_hit, results)
            print(f"   📊 평가 완료: 최고 {max_hit}개 일치, 평균 {avg_hit:.1f}개")

        except Exception as e:
            print(f"❌ 성과 평가 중 오류: {e}")
            traceback.print_exc()

    def _log_reward(self, round_no, max_hit, avg_hit, details):
        try:
            sh = self.get_sheet()
            try: ws_log = sh.worksheet(LOG_SHEET_NAME)
            except:
                ws_log = sh.add_worksheet(title=LOG_SHEET_NAME, rows=1000, cols=10)
                ws_log.append_row(["Timestamp", "Round", "Max Hit", "Avg Hit", "Details"])

            ws_log.append_row([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round_no,
                max_hit,
                f"{avg_hit:.2f}",
                str(details)
            ])
            print("   💾 로그 저장 완료.")
        except Exception as e:
            print(f"⚠️ 로그 저장 실패: {e}")

if __name__ == "__main__":
    app = LottoOrchestrator()
    print("🚀 Manual Mode: Executing Full Strategy (Sequential)...")
    app.sync_data()       # Phase 1
    app.train_brain()     # Phase 2
    app.load_and_predict()# Phase 3
    # app.evaluate_performance() # Phase 4 (Optional)
    print("\n✅ 작전 완료 (Mission Accomplished).")
