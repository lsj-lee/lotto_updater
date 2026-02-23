import os
import time
import gc
import random
import json
import datetime
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import requests
import joblib
import sys

# [Library Migration] Use google-genai
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ Critical Dependency Missing: 'google-genai'")
    print("💡 Please run: pip install google-genai")
    sys.exit(1)

# Check for XGBoost/CatBoost (Optional)
try:
    import xgboost as xgb
except ImportError:
    print("⚠️ Missing XGBoost. Run: pip install xgboost")
    xgb = None

try:
    import catboost as cb
except ImportError:
    print("⚠️ Missing CatBoost. Run: pip install catboost")
    cb = None

from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# --- Configuration & Constants ---
CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'
LOG_SHEET_NAME = 'Log'
REC_SHEET_NAME = '추천번호'
STATE_A_FILE = 'state_A.pkl'
STATE_B_FILE = 'state_B.pkl'
STATE_TOTAL_FILE = 'state_total.pkl'

# Hardware Safety: Device Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Deep Learning: Running on Mac M-Series GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Deep Learning: Running on CPU")

# Hardware Safety: Core Limitation
TOTAL_CORES = multiprocessing.cpu_count()
USED_CORES = max(1, TOTAL_CORES - 2)
torch.set_num_threads(USED_CORES)


# --- [Scout Logic] Intelligent Model Discovery ---
def get_verified_model(api_key):
    """
    [Scout Function] Directly probes Google's REST API to find the most capable active model.
    Prioritizes: 3.x > 2.x > 1.5 Pro > 1.5 Flash
    """
    print("\n🛰️ [Scout] Initiating Deep Space Scan for Intelligence Models...")

    if not api_key:
        print("❌ API Key is missing.")
        return None

    # 1. Fetch available models via REST (Bypassing SDK limitations)
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(list_url)
        if response.status_code != 200:
            print(f"⚠️ Model List Scan Failed: HTTP {response.status_code}")
            return None

        models_data = response.json().get('models', [])
        candidates = []

        # Filter for 'generateContent' capable models
        for m in models_data:
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                candidates.append(m['name'].replace('models/', ''))

        if not candidates:
            print("⚠️ No generation-capable models found.")
            return None

    except Exception as e:
        print(f"⚠️ Network Error during Scan: {e}")
        return None

    # 2. Smart Sorting (Intelligence Hierarchy)
    # We define a score for each model to sort them by capability
    def model_intelligence_score(name):
        score = 0
        name = name.lower()

        # Generation Score
        if 'gemini-3' in name: score += 5000
        elif 'gemini-2' in name: score += 4000
        elif 'gemini-1.5' in name: score += 3000
        elif 'gemini-1.0' in name or 'gemini-pro' in name: score += 1000

        # Variant Score
        if 'ultra' in name: score += 500
        elif 'pro' in name: score += 300
        elif 'flash' in name: score += 100

        # Recency Score
        if 'latest' in name: score += 50
        if 'exp' in name: score += 20  # Experimental models often powerful but unstable

        return score

    candidates.sort(key=model_intelligence_score, reverse=True)
    print(f"📋 Candidate List (Top 5): {candidates[:5]}")

    # 3. Real-World Firing Test (Ping)
    for model_name in candidates:
        print(f"   👉 Testing connection to [{model_name}]...", end="")
        test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": "Hello"}]}]}

        try:
            # Short timeout, we need speed
            start_t = time.time()
            ping = requests.post(test_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=5)
            elapsed = time.time() - start_t

            if ping.status_code == 200:
                print(f" ✅ ONLINE (Latency: {elapsed:.2f}s)")
                return model_name
            elif ping.status_code == 429:
                print(f" ⚠️ BUSY (Rate Limit). Skipping.")
                time.sleep(1) # Backoff
            else:
                print(f" ❌ FAILED (HTTP {ping.status_code})")

        except Exception:
            print(" ❌ ERROR (Timeout/Network)")

    return None


# --- Integrated Orchestrator Class ---
class HybridSniperOrchestrator:
    def __init__(self):
        self.creds_file = CREDS_FILE
        self.sheet_name = SHEET_NAME
        self.gc = self._authenticate_google_sheets()

        # [Dynamic Integration] Use the Scout Function
        api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = get_verified_model(api_key)

        if self.model_name:
            print(f"\n🎯 [Target Locked] System will use: {self.model_name}")
            try:
                self.client = genai.Client(api_key=api_key)
            except:
                print("⚠️ Client Init Failed despite verified model.")
                self.client = None
        else:
             print("\n⚠️ [Critical] All AI Models Unresponsive. Switching to Manual Fallback.")
             self.client = None

        self.data_manager = LottoDataManager(self.gc, self.sheet_name)
        self.ensemble = EnsemblePredictor()

    def _authenticate_google_sheets(self):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        if not os.path.exists(self.creds_file):
            raise FileNotFoundError(f"Credential file {self.creds_file} not found.")
        creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, scope)
        return gspread.authorize(creds)

    # --- Execution Modes (Dual-Mode) ---
    def run_full_cycle(self):
        print("\n" + "="*60)
        print("🚀 사령관 직접 명령: 전 과정 통합 저격을 시작합니다 (Full-Cycle Mode)")
        print("="*60 + "\n")

        self.log_to_sheet("SYSTEM", "MANUAL_START", "Full Cycle Initiated by Commander.")

        # 1. Update Data
        print("\n[Phase 1] Data Synchronization (Anti-GIGO)")
        self.mission_sunday_sync()

        # Safety Pause
        print("❄️ [Safety] Cooling M5 (5s)...")
        time.sleep(5)

        # 2. Total Analysis
        print("\n[Phase 2] Unified Analysis (Supervised ML/DL + Unsupervised)")
        self.mission_monday_total_analysis()

        # Safety Pause
        print("❄️ [Safety] Cooling M5 (10s) before Final Strike...")
        time.sleep(10)

        # 3. Final Strike
        print("\n[Phase 3] Final Strike (Reinforcement PPO + Evolutionary GA + Generative AI)")
        self.mission_wednesday_final_strike()

        print("\n✅ All Missions Accomplished successfully.")

    def dispatch_mission(self, force_day=None):
        day = force_day if force_day else datetime.datetime.now().strftime("%a")
        print(f"🗓️ Mission Control (Scheduled Mode): Today is {day}. Initiating protocols...")
        self.log_to_sheet("SYSTEM", "SCHEDULED", f"Mission started for {day}")

        if day == 'Sun': self.mission_sunday_sync()
        elif day == 'Mon': self.mission_monday_total_analysis()
        elif day == 'Wed': self.mission_wednesday_final_strike()
        else:
            print("💤 No scheduled mission for today. Resting M5.")
            self.log_to_sheet("SYSTEM", "SLEEP", "No mission scheduled.")

    # --- Missions ---
    def mission_sunday_sync(self):
        print("☀️ Sunday Mission: Data Synchronization")
        self.update_data()
        print("✅ Sync Process Finished.")
        self.log_to_sheet("DataSync", "COMPLETE", "Sync check finished.")

    def mission_monday_total_analysis(self):
        print("🌙 Monday Mission: Total Analysis (Unified ML & DL)")
        full_data = self.data_manager.fetch_data()
        if len(full_data) < 100: return

        # [AI Taxonomy: Unsupervised Learning]
        print("🔍 [Unsupervised] Analyzing Data Patterns (Clustering & Dimensionality Reduction)...")
        cluster_info = self.data_manager.analyze_patterns_unsupervised(full_data)
        self.log_to_sheet("Unsupervised", "INFO", f"Pattern Analysis: {cluster_info}")

        split_idx = len(full_data) - 5
        train_data = full_data[:split_idx]
        val_data = full_data[split_idx:]
        val_history = full_data[split_idx-5:split_idx]

        # 1. Supervised Learning: ML Models (Classification/Regression)
        print("📚 [Supervised] Training Group A (ML/Classification)...")
        X_train, y_train = self.data_manager.prepare_training_data(train_data)
        self.ensemble.train_group_a(X_train, y_train)

        # Validation inputs needs to be sequence
        X_val, _ = self.data_manager.prepare_training_data(val_history + val_data, lookback=5)
        val_preds_a = self.ensemble.predict_group_a(X_val)

        X_full, y_full = self.data_manager.prepare_training_data(full_data)
        self.ensemble.train_group_a(X_full, y_full)
        last_seq = full_data[-5:]
        X_next = np.array(last_seq).flatten().reshape(1, -1)
        next_preds_a = self.ensemble.predict_group_a(X_next, is_single=True)

        # [Safety] Cooling
        print("❄️ [Hardware Safety] Cooling Pause (5s)...")
        time.sleep(5)
        gc.collect()

        # 2. Supervised Learning: DL Models (Encoder-Decoder / Feature Extraction)
        print("🧠 [Supervised] Training Group B (DL/Encoder-Decoder)...")
        X_train_dl, y_train_dl = self.data_manager.prepare_training_data(train_data)
        self.ensemble.train_group_b(X_train_dl, y_train_dl)

        val_preds_b = self.ensemble.predict_group_b(X_val)

        self.ensemble.train_group_b(X_full, y_full)
        X_next_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        next_preds_b = self.ensemble.predict_group_b(X_next_tensor, is_single=True)

        state = {
            'val_preds': {**val_preds_a, **val_preds_b},
            'next_preds': {**next_preds_a, **next_preds_b},
            'val_targets': val_data
        }
        joblib.dump(state, STATE_TOTAL_FILE)
        print(f"✅ Total Analysis Saved to {STATE_TOTAL_FILE}")
        self.log_to_sheet("TotalAnalysis", "SAVED", "Unified ML/DL Models Processed.")

        gc.collect()
        if DEVICE.type == 'mps': torch.mps.empty_cache()

    def mission_wednesday_final_strike(self):
        print("🚀 Wednesday Mission: Final Strike (RL + GA + GenAI)")

        if not os.path.exists(STATE_TOTAL_FILE):
            print("❌ Missing State File! Run Analysis first.")
            return

        state = joblib.load(STATE_TOTAL_FILE)

        # [AI Taxonomy: Reinforcement Learning] PPO-based Weighting
        print("⚖️ [Reinforcement Learning] Calculating PPO Reward Weights...")
        val_targets = state['val_targets']
        all_val_preds = state['val_preds']
        weights = self.calculate_ppo_weights(all_val_preds, val_targets)
        print(f"📊 Model Weights: {weights}")

        all_next_preds = state['next_preds']
        final_probs = np.zeros(45)
        for name, pred_probs in all_next_preds.items():
            w = weights.get(name, 1.0)
            final_probs += pred_probs * w
        final_probs /= len(all_next_preds)

        # [AI Taxonomy: Evolutionary Computing] Genetic Algorithm
        print("🧬 [Evolutionary] Running Genetic Algorithm (Optimization)...")
        ga = GeneticEvolution(final_probs, population_size=1000, generations=500)
        elite_candidates = ga.evolve()

        # [AI Taxonomy: Generative AI] LLM Filtering
        model_display = self.model_name if self.model_name else "Manual Mode"
        print(f"🤖 [Generative AI] {model_display}: Strategic Filtering...")
        full_data = self.data_manager.fetch_data()
        last_seq = [d['nums'] for d in full_data[-5:]]

        gemini_filter = GeminiStrategyFilter(self.client, self.model_name)
        final_games = gemini_filter.filter_candidates(elite_candidates, last_seq)

        # [Rate Limit Defense] Post-Prediction Cooling
        time.sleep(3)

        self.update_report(final_games)
        print("✅ Final Strike Complete.")

        if os.path.exists(STATE_TOTAL_FILE): os.remove(STATE_TOTAL_FILE)

    def calculate_ppo_weights(self, all_preds, targets):
        weights = {}
        total_score = 0
        for name, preds in all_preds.items():
            score = 0
            for i in range(len(targets)):
                target_set = set(targets[i])
                if isinstance(preds, list): p = preds[i]
                elif len(preds.shape) > 1: p = preds[i]
                else: p = preds

                # [Fix] Align indices (0-44) to Lotto numbers (1-45)
                top_15 = p.argsort()[::-1][:15] + 1
                hits = len(target_set & set(top_15))
                score += hits
            weights[name] = max(0.1, score)
            total_score += weights[name]
        for k in weights: weights[k] /= total_score
        return weights

    def update_data(self):
        print("📡 Checking for Data Updates...")
        last_round = self.data_manager.get_latest_recorded_round()
        expected_round = self.data_manager.get_current_expected_round()
        if last_round >= expected_round:
            print("✅ Data is up to date.")
            return

        # [Sync Safety] Retry Logic
        failures = 0
        for r in range(last_round + 1, expected_round + 1):
            if failures >= 3:
                print("❌ Too many fetch failures. Aborting Sync to prevent hanging.")
                break

            print(f"🔍 Fetching Round {r}...")
            data = self.fetch_lotto_data_via_gemini(r)
            if data:
                self.data_manager.update_sheet_row(data)
                print(f"💾 Updated Round {r}")
                failures = 0 # Reset on success
            else:
                failures += 1
                print(f"⚠️ Failed to fetch round {r}. (Attempts: {failures}/3)")

            # [Rate Limit Defense]
            time.sleep(3)

    def fetch_lotto_data_via_gemini(self, round_no):
        """
        [Library Migration] Uses google-genai to parse lottery data.
        """
        if not self.client or not self.model_name:
            print("❌ Gemini Client not initialized. Cannot parse.")
            return None

        url = f"https://search.naver.com/search.naver?query=로또+{round_no}회+당첨번호"
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()[:5000]

            prompt = f"Extract Lotto data for round {round_no} from text into JSON. Fields: drwNo(int), drwtNo1..6(int), bnusNo(int), firstAccumamnt(int), firstPrzwnerCo(int), drwNoDate(str YYYY-MM-DD). Text: {text}"

            # [SDK v1.0+] Correct usage
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            json_str = response.text.strip().replace('```json', '').replace('```', '')
            data = json.loads(json_str)
            return data if int(data['drwNo']) == round_no else None
        except Exception as e:
            print(f"❌ Fetch Error ({self.model_name}): {e}")
            return None

    def update_report(self, games):
        """
        [Report Neutralization] Uses neutral terms like 'Scenario' and 'Target Candidate'.
        Groups results into Primary and Secondary categories.
        """
        try:
            ws = self.gc.open(self.sheet_name).worksheet(REC_SHEET_NAME)
            ws.clear()
            model_display = self.model_name if self.model_name else "Fallback Strategy"

            # [Report] Disclaimer and Neutral Title
            ws.update(range_name='A1', values=[['⚠️ 본 리포트는 AI의 적합도 기반 분석 결과이며, 절대적인 당첨 순위를 의미하지 않습니다.']])
            ws.format("A1", {"textFormat": {"foregroundColor": {"red": 1.0, "green": 0.0, "blue": 0.0}, "bold": True}})

            ws.update(range_name='A3', values=[['🏆 Hybrid Sniper V5: AI Suitability Scenarios']])
            ws.update(range_name='A5', values=[[f'🔥 Strategy: {model_display} Selected Candidates']])

            # [Report] Grouping
            report_rows = []

            # Primary Group
            report_rows.append(["[Primary Group] High Probability Candidates"])
            for i in range(5):
                if i < len(games):
                    report_rows.append([f"Scenario {i+1}"] + games[i])

            # Secondary Group
            report_rows.append([]) # Empty row
            report_rows.append(["[Secondary Group] Strategic Alternatives"])
            for i in range(5, 10):
                if i < len(games):
                    report_rows.append([f"Scenario {i+1}"] + games[i])

            ws.update(range_name='A7', values=report_rows)

            # Insight Section
            ws.update(range_name='A25', values=[['🚀 AI Future Technology Lab (R&D Insight)']])
            ws.update(range_name='A26', values=[
                ["Analysis", "Supervised (Classification/Regression) + Unsupervised (PCA)"],
                ["Optimization", "Reinforcement (PPO) + Evolutionary (GA)"],
                ["Filter", f"Generative AI ({model_display})"],
                ["Hardware", "M5 Safety Mode (Active Cooling)"]
            ])
        except Exception as e:
            print(f"Report Error: {e}")

    def log_to_sheet(self, agent, status, msg):
        try:
            ws = self.gc.open(self.sheet_name).worksheet(LOG_SHEET_NAME)
            ws.append_row([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), agent, status, msg])
        except: pass

# --- Data Manager with Unsupervised Learning ---
class LottoDataManager:
    def __init__(self, gc, sheet_name):
        self.gc = gc
        self.sheet_name = sheet_name
        self.numbers = []
        self.raw_data = []

    def fetch_data(self):
        ws = self.gc.open(self.sheet_name).get_worksheet(0)
        records = ws.get_all_values()[1:]
        self.numbers = []
        self.raw_data = []
        for r in records:
            if not r[0]: continue
            try:
                nums = [int(r[i].replace(',', '')) for i in range(1, 7)]
                self.numbers.append(nums)
                self.raw_data.append({'round': int(r[0].replace(',', '')), 'nums': nums})
            except: continue
        return self.numbers

    def analyze_patterns_unsupervised(self, full_data):
        try:
            data = np.array(full_data)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            # [AI Taxonomy: Unsupervised] Clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(scaled_data)
            last_draw = scaled_data[-1].reshape(1, -1)
            cluster_id = kmeans.predict(last_draw)[0]

            # [AI Taxonomy: Unsupervised] Dimensionality Reduction
            pca = PCA(n_components=2)
            pca.fit(scaled_data)
            variance = pca.explained_variance_ratio_

            print(f"   > [Clustering] Identified Pattern Group: {cluster_id}")
            print(f"   > [PCA] Explained Variance Ratio: {variance}")

            return f"Cluster {cluster_id} | PCA Variance: {variance}"
        except Exception as e:
            return f"Unsupervised Analysis failed: {e}"

    def prepare_training_data(self, data_source, lookback=5):
        X, y = [], []
        if len(data_source) <= lookback: return np.array([]), np.array([])
        # Handle list of dicts OR list of lists
        if isinstance(data_source[0], dict):
            numbers = [d['nums'] for d in data_source]
        else:
            numbers = data_source

        for i in range(lookback, len(numbers)):
            seq = numbers[i-lookback:i]
            target = numbers[i]
            X.append(np.array(seq).flatten())
            t_vec = np.zeros(45)
            for n in target: t_vec[n-1] = 1
            y.append(t_vec)
        return np.array(X), np.array(y)

    def get_latest_recorded_round(self):
        try:
            ws = self.gc.open(self.sheet_name).get_worksheet(0)
            val = ws.col_values(1)[-1]
            return int(val.replace('회','').replace(',','').strip())
        except: return 0

    def get_current_expected_round(self):
        start = datetime.datetime(2002, 12, 7, 21, 0, 0)
        diff = datetime.datetime.now() - start
        return diff.days // 7 + 1

    def update_sheet_row(self, data):
        ws = self.gc.open(self.sheet_name).get_worksheet(0)
        row = [data['drwNo'], data['drwNoDate'], data['drwtNo1'], data['drwtNo2'], data['drwtNo3'], data['drwtNo4'], data['drwtNo5'], data['drwtNo6'], data['bnusNo'], data['firstPrzwnerCo'], data['firstAccumamnt'], data.get('firstPrzwnerStore', '')]
        ws.append_row(row)

# --- Ensemble Engine (Unified) ---
class EnsemblePredictor:
    def __init__(self):
        self.models = []

    def train_group_a(self, X, y):
        self.models = []
        # Random Forest (Classification)
        for d in [10, 20, 30, 40, None]:
            rf = RandomForestClassifier(n_estimators=100, max_depth=d, n_jobs=USED_CORES)
            rf.fit(X, y)
            self.models.append((f'RF_d{d}', rf))
            gc.collect()

        # XGBoost (Classification)
        if xgb:
            for d in [3, 5, 7]:
                xgb_est = xgb.XGBClassifier(n_estimators=50, max_depth=d, n_jobs=1, tree_method='hist')
                model = MultiOutputClassifier(xgb_est, n_jobs=USED_CORES)
                model.fit(X, y)
                self.models.append((f'XGB_d{d}', model))
                gc.collect()

        # CatBoost (Classification)
        if cb:
            for d in [4, 6, 8]:
                cbm = cb.CatBoostClassifier(iterations=50, depth=d, verbose=0, thread_count=1)
                model = MultiOutputClassifier(cbm, n_jobs=USED_CORES)
                model.fit(X, y)
                self.models.append((f'CatBoost_d{d}', model))
                gc.collect()

        # KNN (Classification)
        for k in [3, 5, 7, 9, 11]:
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=USED_CORES)
            knn.fit(X, y)
            self.models.append((f'KNN_k{k}', knn))
            gc.collect()

    def predict_group_a(self, X_input, is_single=False):
        preds = {}
        if is_single and X_input.ndim == 1: X_input = X_input.reshape(1, -1)
        for name, model in self.models:
            probs_raw = np.array(model.predict_proba(X_input))
            try: p_vec = np.array([col[:, 1] for col in probs_raw]).T
            except:
                p_vec = np.array([col[0][1] if len(col[0]) > 1 else 0 for col in probs_raw])
                if is_single: p_vec = p_vec.reshape(1, -1)
            if is_single: preds[name] = p_vec[0]
            else: preds[name] = p_vec
        return preds

    def train_group_b(self, X, y):
        self.models = []
        X_tensor = torch.tensor(X, dtype=torch.float32).view(len(X), 5, 6).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        ds = TensorDataset(X_tensor, y_tensor)
        dl = DataLoader(ds, batch_size=32, shuffle=True)

        # Configs for Visibility
        configs = [
            ('LSTM_h64', SimpleLSTM(6, 64)), ('LSTM_h128', SimpleLSTM(6, 128)), ('LSTM_h256', SimpleLSTM(6, 256)),
            ('GRU_h64', SimpleGRU(6, 64)), ('GRU_h128', SimpleGRU(6, 128)), ('GRU_h256', SimpleGRU(6, 256)),
            ('CNN_k2', SimpleCNN(2)), ('CNN_k3', SimpleCNN(3)), ('CNN_k4', SimpleCNN(4))
        ]

        total_models = len(configs)
        for idx, (name, model) in enumerate(configs):
            print(f"   > Training DL Model [{idx+1}/{total_models}] {name}...")
            model = model.to(DEVICE)
            train_torch_model(model, dl)
            self.models.append((name, model))
            gc.collect()
            time.sleep(0.5)

    def predict_group_b(self, X_input, is_single=False):
        preds = {}
        if isinstance(X_input, np.ndarray):
             if is_single: X_input = X_input.reshape(1, 5, 6)
             elif X_input.ndim == 2: X_input = X_input.reshape(len(X_input), 5, 6)
             X_tensor = torch.tensor(X_input, dtype=torch.float32).to(DEVICE)
        else: X_tensor = X_input
        for name, model in self.models:
            model.eval()
            with torch.no_grad(): out = model(X_tensor).cpu().numpy()
            if is_single: preds[name] = out[0]
            else: preds[name] = out
        return preds

# --- Helpers ---
class SimpleLSTM(nn.Module):
    """[Architecture] Encoder-Decoder Structure for Temporal Feature Extraction"""
    def __init__(self, i, h):
        super().__init__()
        self.lstm = nn.LSTM(i, h, batch_first=True)
        self.fc = nn.Linear(h, 45)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.sig(self.fc(h[-1]))

class SimpleGRU(nn.Module):
    """[Architecture] Encoder-Decoder Structure for Temporal Feature Extraction"""
    def __init__(self, i, h):
        super().__init__()
        self.gru = nn.GRU(i, h, batch_first=True)
        self.fc = nn.Linear(h, 45)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        _, h = self.gru(x)
        return self.sig(self.fc(h[-1]))

class SimpleCNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.conv = nn.Conv1d(6, 32, kernel_size=k)
        self.fc = nn.Linear(32 * (5 - k + 1), 45)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.sig(self.fc(x))

class TensorDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

def train_torch_model(model, loader):
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.BCELoss()
    model.train()
    epochs = 50
    for e in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        if (e+1) % 10 == 0:
            pass

class GeneticEvolution:
    def __init__(self, probs, population_size=1000, generations=500):
        self.probs = probs
        self.pop_size = population_size
        self.generations = generations

    def fitness(self, gene): return sum(self.probs[n-1] for n in gene)

    def evolve(self):
        print("🧬 [Evolutionary] Running Genetic Algorithm...")
        pop = []
        nums = list(range(1, 46))
        w = self.probs / self.probs.sum()
        for _ in range(self.pop_size):
            pop.append(sorted(np.random.choice(nums, 6, replace=False, p=w)))

        for g in range(self.generations):
            scores = [(gene, self.fitness(gene)) for gene in pop]
            scores.sort(key=lambda x: x[1], reverse=True)
            elites = [s[0] for s in scores[:int(self.pop_size * 0.2)]]
            next_gen = elites[:]
            while len(next_gen) < self.pop_size:
                p1, p2 = random.choice(elites), random.choice(elites)
                child = sorted(list(set(p1[:3] + p2[3:])))
                while len(child) < 6:
                    n = random.randint(1, 45)
                    if n not in child: child.append(n)
                next_gen.append(child[:6])
            pop = next_gen
            if (g+1) % 50 == 0:
                time.sleep(1.5)
                print(f"   > Gen {g+1} Cooling...")

        scores = [(gene, self.fitness(gene)) for gene in pop]
        scores.sort(key=lambda x: x[1], reverse=True)
        unique = []
        seen = set()
        for gene, s in scores:
            t = tuple(gene)
            if t not in seen: unique.append(gene); seen.add(t)
            if len(unique) >= 30: break
        return unique

class GeminiStrategyFilter:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def filter_candidates(self, candidates, recent):
        if not self.client: return candidates[:10]
        prompt = f"Select 10 best lotto combinations from {candidates} considering recent flow {recent}. Output strictly JSON: {{'games': [[...]]}}"
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            data = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            return data['games']
        except Exception as e:
            print(f"❌ Gemini Strategy Error: {e}")
            return candidates[:10]

if __name__ == "__main__":
    is_scheduled = False
    for arg in sys.argv:
        if arg == "--scheduled": is_scheduled = True
    orchestrator = HybridSniperOrchestrator()
    if is_scheduled: orchestrator.dispatch_mission()
    else: orchestrator.run_full_cycle()
