# -*- coding: utf-8 -*-
import os
import time
import gc
import random
import json
import datetime
import re
import sys
import traceback
import itertools
import psutil
from collections import deque

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
    print("❌ 'google-genai' 라이브러리가 필요합니다. pip install google-genai를 실행하세요.")
    sys.exit(1)

load_dotenv()

# ==========================================
# ⚙️ [Configuration] 기지 좌표 및 설정
# ==========================================

SPREADSHEET_ID = '1lOifE_xRUocAY_Av-P67uBMKOV1BAb4mMwg_wde_tyA'
CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'
REC_SHEET_NAME = '추천번호'
LOG_SHEET_NAME = '작전로그'
STATE_FILE = 'hybrid_sniper_v5_state.pth'
SNIPER_STATE_JSON = 'sniper_state.json'

# 🚀 M5 하드웨어 가속 설정
USED_CORES = 6
torch.set_num_threads(USED_CORES)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"🚀 [System] M5 Neural Engine (MPS/Metal) 가속 활성화. (Core: {USED_CORES})")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ [System] MPS 가속 불가. CPU 모드로 실행합니다.")

REAL_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Referer": "https://www.naver.com/"
}

# ==========================================
# 🧠 [Core Engine] 신경망 모델 및 특징 추출
# ==========================================

class NDA_FeatureEngine:
    """
    [데이터 특징 공학 엔진]
    로또 번호의 통계적 특징(합계, 홀짝, 고저, AC값)을 계산합니다.
    """
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
    """
    [하이브리드 신경망 모델]
    LSTM(시계열) + Dense(통계) 결합 구조
    """
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
# 🛰️ [System] Orchestrator (Main Logic)
# ==========================================

class SniperState:
    """
    [지능형 상태 관리자]
    sniper_state.json을 통해 작전 상태, 학습 지표, 동적 프롬프트를 관리합니다.
    """
    def __init__(self):
        self.state_file = SNIPER_STATE_JSON
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass

        # 기본 상태값
        return {
            "last_sync_date": None,
            "last_train_date": None,
            "last_predict_date": None,
            "last_evolution_date": None,
            "last_loss": 0.0,
            "active_strategy_prompt": {
                "version": "v1.0 (Default)",
                "content": """
                당신은 로또 분석 전문가입니다. 아래 50개의 유력 조합 중, 당첨 확률이 가장 높아 보이는 5~10개를 골라주세요.
                번호가 골고루 분포되어 있고, 너무 뻔한 패턴이 아닌 것을 선호합니다.
                """
            },
            "recent_hit_rates": []
        }

    def save_state(self):
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=4)

    def update_phase(self, phase_key, value=None):
        if value is None:
            value = datetime.datetime.now().strftime("%Y-%m-%d")
        self.state[phase_key] = value
        self.save_state()

    def update_metric(self, key, value):
        self.state[key] = value
        self.save_state()

    def add_hit_rate(self, hit_rate):
        rates = self.state.get("recent_hit_rates", [])
        rates.append(hit_rate)
        if len(rates) > 5: rates.pop(0)
        self.state["recent_hit_rates"] = rates
        self.save_state()

class LottoOrchestrator:
    def __init__(self):
        self.gc_client = self._auth()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = self._init_gemini(api_key)
        self.state_manager = SniperState()

        # [지휘관 모델 고정] gemini-2.5-flash
        self.model_name = "gemini-2.5-flash"
        print(f"🛰️ [System] 지휘관 모델 설정: {self.model_name}")

    def _auth(self):
        """
        [하이브리드 인증] 로컬 파일 우선, 부재 시 환경 변수 사용
        """
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive",
                 "https://www.googleapis.com/auth/spreadsheets"]
        try:
            if os.path.exists(CREDS_FILE):
                print("🔑 [Auth] 로컬 인증 파일 사용")
                creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
            elif os.getenv("GOOGLE_CREDS_JSON"):
                print("🔑 [Auth] GitHub Secrets 인증 사용")
                creds_dict = json.loads(os.getenv("GOOGLE_CREDS_JSON"))
                creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            else:
                raise FileNotFoundError("❌ 인증 파일을 찾을 수 없습니다.")
            return gspread.authorize(creds)
        except Exception as e:
            print(f"❌ 인증 실패: {e}")
            sys.exit(1)

    def _init_gemini(self, api_key):
        if not api_key: return None
        try: return genai.Client(api_key=api_key)
        except: return None

    def get_sheet(self):
        try: return self.gc_client.open_by_key(SPREADSHEET_ID)
        except: return self.gc_client.open(SHEET_NAME)

    def cleanup_memory(self):
        """[M5 최적화] 메모리 강제 정화"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def log_operation(self, phase, status, detail=""):
        try:
            sh = self.get_sheet()
            try: ws = sh.worksheet(LOG_SHEET_NAME)
            except:
                ws = sh.add_worksheet(title=LOG_SHEET_NAME, rows=1000, cols=10)
                ws.append_row(["Timestamp", "Day", "Phase", "Status", "CPU/MEM", "Detail"])

            now = datetime.datetime.now()
            icon = "✅" if status == "SUCCESS" else "❌" if status == "FAIL" else "💤"
            ws.insert_row([
                now.strftime("%Y-%m-%d %H:%M:%S"),
                now.strftime("%A"),
                phase,
                f"{icon} {status}",
                f"{psutil.cpu_percent()}% / {psutil.virtual_memory().percent}%",
                detail
            ], 2)
            print(f"📝 [Log] {phase} - {status}")
        except: pass

    # --- Phase 1: Data Sync ---
    def sync_data(self):
        print("\n🔄 [Phase 1] 데이터 동기화 (Naver + Gemini)...")
        self.cleanup_memory()
        try:
            sh = self.get_sheet()
            ws = sh.get_worksheet(0)
            col1 = ws.col_values(1)
            rounds = [int(str(v).replace('회','').replace(',','').strip()) for v in col1 if str(v).replace('회','').replace(',','').strip().isdigit()]
            local_last = max(rounds) if rounds else 0
            portal_last = self._get_naver_latest_round()
            print(f"   📊 상태: 로컬({local_last}) vs 네이버({portal_last})")

            cnt = 0
            if portal_last > local_last:
                for r in range(local_last + 1, portal_last + 1):
                    data = self._scrape_round_detail(r)
                    if data:
                        row = [data['drwNo'], data['drwNoDate'], data['drwtNo1'], data['drwtNo2'], data['drwtNo3'],
                               data['drwtNo4'], data['drwtNo5'], data['drwtNo6'], data['bnusNo'],
                               data.get('firstPrzwnerCo', 0), data.get('firstAccumamnt', 0), ""]
                        ws.insert_row(row, 2)
                        cnt += 1
                        time.sleep(2)
            else:
                print("   ✅ 최신 상태임.")

            self.state_manager.update_phase("last_sync_date")
            self.log_operation("Phase 1", "SUCCESS", f"Updated {cnt}")
        except Exception as e:
            print(f"❌ 동기화 실패: {e}")
            self.log_operation("Phase 1", "FAIL", str(e))
        finally:
            self.cleanup_memory()

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
            text = soup.get_text()[:3000]
            if self.client:
                prompt = f"JSON for Lotto {round_no} from: {text}"
                try:
                    resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
                    return json.loads(resp.text.strip().replace('```json','').replace('```',''))
                except: pass

            # Fallback
            nums = re.findall(r'\b(\d{1,2})\b', text)
            valid = [int(n) for n in nums if 1 <= int(n) <= 45]
            if len(valid) >= 7:
                return {"drwNo": round_no, "drwNoDate": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "drwtNo1": valid[0], "drwtNo2": valid[1], "drwtNo3": valid[2],
                        "drwtNo4": valid[3], "drwtNo5": valid[4], "drwtNo6": valid[5], "bnusNo": valid[6]}
            return None
        except: return None

    # --- Phase 2: Train ---
    def load_data(self):
        try:
            sh = self.get_sheet()
            ws = sh.get_worksheet(0)
            rows = ws.get_all_values()[1:]
            data = []
            for r in rows:
                try:
                    nums = [int(str(x).replace(',', '')) for x in r[2:8]]
                    data.append(nums)
                except: pass
            data.reverse()
            return data
        except: return []

    def train_brain(self):
        print("\n🧠 [Phase 2] 모델 학습 (M5)...")
        self.cleanup_memory()
        try:
            data = self.load_data()
            if len(data) < 50: return

            X_seq, X_stat, y = NDA_FeatureEngine.create_multimodal_dataset(data, 10)
            model = CreativeConnectionModel().to(DEVICE)
            opt = optim.Adam(model.parameters(), lr=0.001)
            crit = nn.BCELoss()

            model.train()
            loss_val = 0
            for e in range(100):
                opt.zero_grad()
                loss = crit(model(X_seq, X_stat), y)
                loss.backward()
                opt.step()
                loss_val = loss.item()
                if (e+1)%20 == 0: print(f"   Epoch {e+1}: {loss_val:.4f}")

            torch.save(model.state_dict(), STATE_FILE)
            self.state_manager.update_phase("last_train_date")
            self.state_manager.update_metric("last_loss", loss_val)
            self.log_operation("Phase 2", "SUCCESS", f"Loss: {loss_val:.4f}")
            del model, X_seq, X_stat, y
        except Exception as e:
            self.log_operation("Phase 2", "FAIL", str(e))
        finally:
            self.cleanup_memory()

    # --- Phase 3: Predict ---
    def load_and_predict(self):
        print("\n🔮 [Phase 3] 지능형 예측 (동적 프롬프트)...")
        self.cleanup_memory()
        try:
            data = self.load_data()
            if not data or not os.path.exists(STATE_FILE): return

            model = CreativeConnectionModel().to(DEVICE)
            model.load_state_dict(torch.load(STATE_FILE, map_location=DEVICE))
            model.eval()

            # 1. Top 20 Extraction
            last_seq = data[-10:]
            input_seq = torch.tensor(np.array(last_seq)/45.0, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            input_stat = torch.tensor(NDA_FeatureEngine.calculate_derived_features([data[-1]]), dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                probs = model(input_seq, input_stat).cpu().numpy()[0]

            top_20 = [int(n+1) for n in probs.argsort()[::-1][:20]]
            print(f"   🎯 Top 20: {sorted(top_20)}")

            # 2. Simulation & Filtering
            combos = list(itertools.combinations(top_20, 6))
            if len(combos) > 10000: combos = random.sample(combos, 10000)

            filtered = []
            for c in combos:
                if 100 <= sum(c) <= 170 and 2 <= sum(1 for n in c if n%2!=0) <= 4:
                    filtered.append(sorted(list(c)))

            candidates = random.sample(filtered, 50) if len(filtered) > 50 else filtered
            print(f"   ✅ 후보 압축: {len(candidates)}개")

            # 3. LLM Selection (Dynamic Prompt)
            final = self._ask_gemini(candidates)
            self._write_sheet(final if final else candidates[:10])

            self.state_manager.update_phase("last_predict_date")
            self.log_operation("Phase 3", "SUCCESS", f"Generated {len(final) if final else 10}")

        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            self.log_operation("Phase 3", "FAIL", str(e))
        finally:
            self.cleanup_memory()

    def _ask_gemini(self, candidates):
        if not self.client: return None

        # [동적 프롬프트 로드]
        state_prompt = self.state_manager.state.get("active_strategy_prompt", {})
        prompt_content = state_prompt.get("content", "기본 프롬프트: 골고루 분포된 번호를 고르세요.")
        version = state_prompt.get("version", "Default")

        print(f"   🧬 적용된 전략: {version}")

        c_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
        full_prompt = f"{prompt_content}\n\n[후보]\n{c_str}\n\n[출력]\n오직 JSON 배열만 출력."

        try:
            resp = self.client.models.generate_content(model=self.model_name, contents=full_prompt)
            return json.loads(resp.text.strip().replace('```json','').replace('```',''))
        except: return None

    def _write_sheet(self, games):
        try:
            sh = self.get_sheet()
            try: ws = sh.worksheet(REC_SHEET_NAME)
            except: ws = sh.add_worksheet(REC_SHEET_NAME, 100, 20)
            ws.clear()
            ws.update(range_name='A1', values=[['🏆 Sniper V5 추천 번호']])
            ws.update(range_name='A3', values=[[f"시나리오 {i+1}"] + g for i, g in enumerate(games)])
            print("   ✅ 시트 저장 완료.")
        except: pass

    # --- Phase 4: Evaluate ---
    def evaluate_performance(self):
        print("\n🏅 [Phase 4] 성과 평가...")
        try:
            sh = self.get_sheet()
            main_ws = sh.get_worksheet(0)
            row = main_ws.row_values(2)
            real = set([int(x) for x in row[2:8]])
            bonus = int(row[8])

            try: rec_ws = sh.worksheet(REC_SHEET_NAME)
            except: return

            preds = []
            for r in rec_ws.get_all_values():
                if "시나리오" in r[0]:
                    preds.append(set([int(x) for x in r[1:7] if x]))

            if not preds: return

            total_hits = 0
            max_hit = 0
            for p in preds:
                cnt = len(real.intersection(p))
                total_hits += cnt
                if cnt > max_hit: max_hit = cnt

            avg = total_hits / len(preds)
            self.state_manager.add_hit_rate(avg)
            self.log_operation("Phase 4", "SUCCESS", f"Max: {max_hit}, Avg: {avg:.2f}")
            print(f"   📊 결과: 최고 {max_hit}개, 평균 {avg:.2f}개")

        except Exception as e:
            self.log_operation("Phase 4", "FAIL", str(e))

if __name__ == "__main__":
    app = LottoOrchestrator()
    print("🚀 Manual Run...")
    app.sync_data()
    app.train_brain()
    app.load_and_predict()
    # app.evaluate_performance()
