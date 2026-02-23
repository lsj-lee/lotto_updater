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
# ⚙️ [Configuration] 기지 좌표 및 M5 최적화 설정
# ==========================================

SPREADSHEET_ID = '1lOifE_xRUocAY_Av-P67uBMKOV1BAb4mMwg_wde_tyA'
CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'
REC_SHEET_NAME = '추천번호'
LOG_SHEET_NAME = '작전로그'
STATE_FILE = 'hybrid_sniper_v5_state.pth'
SNIPER_STATE_JSON = 'sniper_state.json'

# 🚀 MacBook Pro M5 하드웨어 안전장치 (발열 관리 및 성능 최적화)
USED_CORES = 6
torch.set_num_threads(USED_CORES)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"🚀 [System] M5 Neural Engine (MPS/Metal) 가속 활성화. (Core: {USED_CORES})")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ [System] MPS 가속을 사용할 수 없습니다. 일반 CPU 모드로 실행합니다.")

REAL_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Referer": "https://www.naver.com/"
}

# ==========================================
# 🧠 [Core Engine] 신경망 모델 및 특징 추출기
# ==========================================

class NDA_FeatureEngine:
    """
    [데이터 특징 공학 엔진]
    로또 번호의 통계적 특징(합계, 홀짝, 고저, AC값)을 계산하여
    AI가 숫자 패턴을 더 잘 이해하도록 돕는 전처리 클래스입니다.
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
    - 작전 수행 상태, 학습 지표, 그리고 '진화된 프롬프트(Strategy)'를 관리합니다.
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

        # 기본 상태 초기화
        return {
            "last_sync_date": None,
            "last_train_date": None,
            "last_predict_date": None,
            "last_evolution_date": None,
            "last_loss": 0.0,
            "active_strategy_prompt": {
                "version": "v1.0",
                "content": """
                당신은 로또 분석 전문가입니다. 아래 50개의 유력 조합 중, 당첨 확률이 가장 높아 보이는 5~10개를 골라주세요.
                번호가 골고루 분포되어 있고, 너무 뻔한 패턴이 아닌 것을 선호합니다.
                """
            },
            "recent_hit_rates": [] # 최근 5주간 성과 (진화용 피드백 데이터)
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

    def update_strategy_prompt(self, new_prompt, version):
        self.state["active_strategy_prompt"] = {
            "version": version,
            "content": new_prompt
        }
        self.save_state()

    def add_hit_rate(self, hit_rate):
        """최근 성과를 기록 (최대 5개 유지)"""
        rates = self.state.get("recent_hit_rates", [])
        rates.append(hit_rate)
        if len(rates) > 5:
            rates.pop(0)
        self.state["recent_hit_rates"] = rates
        self.save_state()

class SystemMonitor:
    @staticmethod
    def check_health():
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        mem_percent = memory.percent
        print(f"🩺 [System Check] CPU: {cpu_usage}% | RAM: {mem_percent}%")
        is_healthy = (cpu_usage < 50) and (mem_percent < 70)
        return is_healthy, cpu_usage, mem_percent

class LottoOrchestrator:
    def __init__(self):
        self.gc_client = self._auth()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = self._init_gemini(api_key)
        self.state_manager = SniperState()
        self.model_name = "gemini-2.5-flash"
        print(f"🛰️ [System] 지휘관 모델 설정: {self.model_name}")

    def _auth(self):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive",
                 "https://www.googleapis.com/auth/spreadsheets"]
        try:
            if os.path.exists(CREDS_FILE):
                creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
            elif os.getenv("GOOGLE_CREDS_JSON"):
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

    def _optimize_memory(self):
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
            day_str = now.strftime("%A")
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            icon = "✅" if status == "SUCCESS" else "❌" if status == "FAIL" else "💤"

            ws.insert_row([
                now.strftime("%Y-%m-%d %H:%M:%S"), day_str, phase,
                f"{icon} {status}", f"{cpu}% / {mem}%", detail
            ], 2)
            print(f"📝 [Log] 작전로그 기록 완료: {phase} - {status}")
        except Exception as e:
            print(f"⚠️ 로그 기록 실패: {e}")

    # --- Phase 1: Sync ---
    def sync_data(self):
        print("\n🔄 [Phase 1] 데이터 동기화 시작 (Naver + Gemini)...")
        self._optimize_memory()
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

            updated_count = 0
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
                        updated_count += 1
                        time.sleep(2)
            else:
                print("   ✅ 이미 최신 상태입니다.")

            self.state_manager.update_phase("last_sync_date")
            self.log_operation("Phase 1: Sync", "SUCCESS", f"Updated {updated_count} rounds")

        except Exception as e:
            print(f"❌ 동기화 중 오류: {e}")
            self.log_operation("Phase 1: Sync", "FAIL", str(e))
        finally:
            self._optimize_memory()

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
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return []

    def train_brain(self):
        print("\n🧠 [Phase 2] AI 모델 학습 시작 (M5 Neural Engine)...")
        self._optimize_memory()
        try:
            data = self.load_data()
            if len(data) < 50: return None

            X_seq, X_stat, y = NDA_FeatureEngine.create_multimodal_dataset(data, 10)
            model = CreativeConnectionModel().to(DEVICE)
            opt = optim.Adam(model.parameters(), lr=0.001)
            crit = nn.BCELoss()

            model.train()
            final_loss = 0.0
            for e in range(100):
                opt.zero_grad()
                loss = crit(model(X_seq, X_stat), y)
                loss.backward()
                opt.step()
                final_loss = loss.item()
                if (e+1) % 20 == 0:
                    print(f"   Epoch {e+1}/100 - Loss: {final_loss:.4f}")

            torch.save(model.state_dict(), STATE_FILE)
            print(f"💾 학습 완료. 가중치 파일 저장됨: {STATE_FILE}")

            self.state_manager.update_phase("last_train_date")
            self.state_manager.update_metric("last_loss", final_loss)
            self.log_operation("Phase 2: Train", "SUCCESS", f"Loss: {final_loss:.4f}")
            del model, X_seq, X_stat, y
        except Exception as e:
            print(f"❌ 학습 중 오류: {e}")
            self.log_operation("Phase 2: Train", "FAIL", str(e))
        finally:
            self._optimize_memory()

    # --- Phase 3: Predict ---
    def load_and_predict(self):
        print("\n🔮 [Phase 3] 하이브리드 예측 전략 가동 (Top 20 + LLM)...")
        self._optimize_memory()
        try:
            data = self.load_data()
            if not data or not os.path.exists(STATE_FILE):
                print("❌ 데이터 또는 가중치 파일 없음.")
                return

            print("1️⃣ [AI 분석] 상위 20개 유력 번호(Top 20) 추출 중...")
            model = CreativeConnectionModel().to(DEVICE)
            model.load_state_dict(torch.load(STATE_FILE, map_location=DEVICE))
            model.eval()

            last_seq = data[-10:]
            input_seq = torch.tensor(np.array(last_seq) / 45.0, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            input_stat = torch.tensor(NDA_FeatureEngine.calculate_derived_features([data[-1]]), dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                probs = model(input_seq, input_stat).cpu().numpy()[0]

            top_20_indices = probs.argsort()[::-1][:20]
            top_20_nums = [int(n+1) for n in top_20_indices]
            print(f"   🎯 Top 20 후보 번호: {sorted(top_20_nums)}")

            print("2️⃣ [시뮬레이션] Top 20 기반 10,000개 조합 생성 중...")
            generated_games = []
            all_combinations = list(itertools.combinations(top_20_nums, 6))
            if len(all_combinations) > 10000:
                generated_games = random.sample(all_combinations, 10000)
            else:
                generated_games = all_combinations

            del all_combinations
            self._optimize_memory()

            print("3️⃣ [필터링] 통계적 기준(합계, 홀짝)으로 50개 압축 중...")
            filtered_games = []
            for game in generated_games:
                total = sum(game)
                if not (100 <= total <= 170): continue
                odd_count = sum(1 for n in game if n % 2 != 0)
                if not (2 <= odd_count <= 4): continue
                filtered_games.append(sorted(list(game)))

            final_candidates = random.sample(filtered_games, 50) if len(filtered_games) > 50 else filtered_games
            print(f"   ✅ 필터링 통과: {len(filtered_games)}개 -> 최종 후보 50개 선정.")

            print("4️⃣ [LLM 전략] Gemini에게 최종 5~10개 추천 요청 중...")
            # [진화된 프롬프트 사용]
            final_selection = self._ask_gemini_to_select(final_candidates)

            if final_selection:
                self._write_sheet(final_selection)
            else:
                print("   ⚠️ LLM 응답 실패로 랜덤 10개를 저장합니다.")
                self._write_sheet(final_candidates[:10])

            self.state_manager.update_phase("last_predict_date")
            self.log_operation("Phase 3: Predict", "SUCCESS", f"Generated {len(final_selection) if final_selection else 10} games")

        except Exception as e:
            print(f"❌ 예측 프로세스 중 오류: {e}")
            traceback.print_exc()
            self.log_operation("Phase 3: Predict", "FAIL", str(e))
        finally:
            self._optimize_memory()

    def _ask_gemini_to_select(self, candidates):
        """
        [지능형 프롬프트 호출]
        상태 파일에 저장된 '최신 진화 프롬프트'를 불러와 Gemini에게 전달합니다.
        """
        if not self.client: return None

        candidates_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

        # [핵심] 진화된 프롬프트 로드 (Fallback: Default)
        strategy = self.state_manager.state.get("active_strategy_prompt", {})
        strategy_content = strategy.get("content", """
        당신은 로또 분석 전문가입니다. 아래 50개의 유력 조합 중, 당첨 확률이 가장 높아 보이는 5~10개를 골라주세요.
        번호가 골고루 분포되어 있고, 너무 뻔한 패턴이 아닌 것을 선호합니다.
        """)
        version = strategy.get("version", "v1.0 (Default)")

        print(f"   🧬 [Meta-Prompting] 전략 버전 사용: {version}")

        full_prompt = f"""
        {strategy_content}

        [후보 목록]
        {candidates_str}

        [출력]
        오직 JSON 배열만 출력하세요. 예: [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        """
        try:
            resp = self.client.models.generate_content(model=self.model_name, contents=full_prompt)
            text = resp.text.strip().replace('```json', '').replace('```', '')
            selected = json.loads(text)
            if isinstance(selected, list) and len(selected) > 0:
                print(f"   ✨ Gemini가 {len(selected)}개의 조합을 엄선했습니다.")
                return selected
            return None
        except Exception as e:
            print(f"   ⚠️ Gemini 요청 실패: {e}")
            return None

    def _write_sheet(self, games):
        sh = self.get_sheet()
        try: ws = sh.worksheet(REC_SHEET_NAME)
        except: ws = sh.add_worksheet(title=REC_SHEET_NAME, rows=100, cols=20)

        ws.clear()
        ws.update(range_name='A1', values=[['🏆 Sniper V5 최종 추천 번호 (Top 20 Hybrid)']])
        ws.update(range_name='A3', values=[[f"시나리오 {i+1}"] + g for i, g in enumerate(games)])
        print("   ✅ 구글 시트 '추천번호' 탭에 결과가 기록되었습니다.")

    # --- Phase 4: Reward ---
    def evaluate_performance(self):
        print("\n🏅 [Phase 4] 지난 작전 성과 평가 (Reward Check)...")
        try:
            sh = self.get_sheet()
            ws_main = sh.get_worksheet(0)
            latest_row = ws_main.row_values(2)
            real_round = int(latest_row[0].replace('회', ''))
            real_nums = set([int(x) for x in latest_row[2:8]])
            bonus_num = int(latest_row[8])
            print(f"   🎯 실제 결과 ({real_round}회): {sorted(list(real_nums))} + 보너스 {bonus_num}")

            try: ws_rec = sh.worksheet(REC_SHEET_NAME)
            except:
                print("   ⚠️ 추천 번호 시트가 없습니다.")
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
                print("   ⚠️ 평가할 예측 데이터가 없습니다.")
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

            # [피드백 데이터 축적]
            self.state_manager.add_hit_rate(avg_hit)

            self.log_operation("Phase 4: Reward", "SUCCESS", f"Max Hit: {max_hit}, Avg: {avg_hit:.2f}")
            print(f"   📊 평가 완료: 최고 {max_hit}개 일치, 평균 {avg_hit:.1f}개")

        except Exception as e:
            print(f"❌ 성과 평가 중 오류: {e}")
            traceback.print_exc()
            self.log_operation("Phase 4: Reward", "FAIL", str(e))

    def _log_reward(self, round_no, max_hit, avg_hit, details):
        try:
            sh = self.get_sheet()
            try: ws_log = sh.worksheet(LOG_SHEET_NAME)
            except:
                ws_log = sh.add_worksheet(title=LOG_SHEET_NAME, rows=1000, cols=10)
                ws_log.append_row(["Timestamp", "Day", "Phase", "Status", "CPU/MEM", "Detail"])

            try:
                ws_detail_log = sh.worksheet("Log")
            except:
                ws_detail_log = sh.add_worksheet(title="Log", rows=1000, cols=10)
                ws_detail_log.append_row(["Timestamp", "Round", "Max Hit", "Avg Hit", "Details"])

            ws_detail_log.append_row([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round_no,
                max_hit,
                f"{avg_hit:.2f}",
                str(details)
            ])
            print("   💾 상세 로그 저장 완료.")
        except Exception as e:
            print(f"⚠️ 상세 로그 저장 실패: {e}")

if __name__ == "__main__":
    app = LottoOrchestrator()
    print("🚀 수동 모드: 전체 파이프라인 순차 실행...")
    app.sync_data()
    app.train_brain()
    app.load_and_predict()
    # app.evaluate_performance()
    print("\n✅ 작전 완료.")
