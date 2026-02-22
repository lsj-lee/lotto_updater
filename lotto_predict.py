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
import xgboost as xgb
import catboost as cb
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import joblib

# Load environment variables
load_dotenv()

# --- Configuration & Constants ---
CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'
LOG_SHEET_NAME = 'Log'
REC_SHEET_NAME = '추천번호'
STATE_A_FILE = 'state_A.pkl'
STATE_B_FILE = 'state_B.pkl'

# Device Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Deep Learning: Running on Mac M-Series GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Deep Learning: Running on CPU")

# Hardware Safety
TOTAL_CORES = multiprocessing.cpu_count()
USED_CORES = max(1, TOTAL_CORES - 2)
torch.set_num_threads(USED_CORES)

# --- Integrated Orchestrator Class ---
class HybridSniperOrchestrator:
    def __init__(self):
        self.creds_file = CREDS_FILE
        self.sheet_name = SHEET_NAME
        self.gc = self._authenticate_google_sheets()
        self.genai_model = self._setup_gemini()
        self.data_manager = LottoDataManager(self.gc, self.sheet_name)
        self.ensemble = EnsemblePredictor()

    def _authenticate_google_sheets(self):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        if not os.path.exists(self.creds_file):
            raise FileNotFoundError(f"Credential file {self.creds_file} not found.")
        creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, scope)
        return gspread.authorize(creds)

    def _setup_gemini(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-pro')

    def dispatch_mission(self, force_day=None):
        """
        Executes specific missions based on the day of the week.
        Sunday: Sync Data
        Monday: Analysis A (ML Models)
        Tuesday: Analysis B (DL Models)
        Wednesday: Final Strike (GA + Gemini)
        """
        day = force_day if force_day else datetime.datetime.now().strftime("%a")
        print(f"🗓️ Mission Control: Today is {day}. Initiating protocols...")

        self.log_to_sheet("SYSTEM", "START", f"Mission started for {day}")

        if day == 'Sun':
            self.mission_sunday_sync()
        elif day == 'Mon':
            self.mission_monday_analysis_a()
        elif day == 'Tue':
            self.mission_tuesday_analysis_b()
        elif day == 'Wed':
            self.mission_wednesday_final_strike()
        else:
            print("💤 No scheduled mission for today. Resting M5.")
            self.log_to_sheet("SYSTEM", "SLEEP", "No mission scheduled.")

    def mission_sunday_sync(self):
        print("☀️ Sunday Mission: Data Synchronization")
        self.update_data()
        print("✅ Sync Complete.")
        self.log_to_sheet("DataSync", "COMPLETE", "Updated latest rounds.")

    def mission_monday_analysis_a(self):
        print("🌙 Monday Mission: Group A Analysis (ML Models)")
        full_data = self.data_manager.fetch_data()
        if len(full_data) < 100: return

        # Split for PPO: Train on N-5, Predict Validation (N-5 to N) & Next (N+1)
        split_idx = len(full_data) - 5
        train_data = full_data[:split_idx]
        val_data = full_data[split_idx:] # Validation Target
        val_history = full_data[split_idx-5:split_idx] # History for Val Input

        # 1. Train on History -> Predict Validation
        print("   > Training Group A on Historical Data...")
        X_train, y_train = self.data_manager.prepare_training_data(train_data)
        self.ensemble.train_group_a(X_train, y_train)

        # Predict Validation (for PPO Weighting)
        X_val, _ = self.data_manager.prepare_training_data(val_history + val_data, lookback=5)
        # Note: prepare_training_data generates multiple samples. We just need the last 5.
        # Actually, X_val generated from (val_history + val_data) will output predictions corresponding to val_data targets.
        val_preds_a = self.ensemble.predict_group_a(X_val[-5:])

        # 2. Retrain on Full Data -> Predict Next Round
        print("   > Retraining Group A on Full Data...")
        X_full, y_full = self.data_manager.prepare_training_data(full_data)
        self.ensemble.train_group_a(X_full, y_full)

        last_seq = full_data[-5:] # Input for next round
        X_next = np.array(last_seq).flatten().reshape(1, -1)
        next_preds_a = self.ensemble.predict_group_a(X_next, is_single=True) # Dictionary of preds

        # Save State
        state = {
            'val_preds': val_preds_a,
            'next_preds': next_preds_a,
            'val_targets': [d['nums'] for d in val_data] # Save targets for PPO comparison
        }
        joblib.dump(state, STATE_A_FILE)
        print(f"✅ Group A Analysis Saved to {STATE_A_FILE}")
        self.log_to_sheet("Analysis_A", "SAVED", "ML Models (1-8) Processed.")
        gc.collect()

    def mission_tuesday_analysis_b(self):
        print("🔥 Tuesday Mission: Group B Analysis (DL Models)")
        full_data = self.data_manager.fetch_data()

        split_idx = len(full_data) - 5
        train_data = full_data[:split_idx]
        val_data = full_data[split_idx:]
        val_history = full_data[split_idx-5:split_idx]

        # 1. Train on History -> Predict Validation
        print("   > Training Group B on Historical Data...")
        X_train, y_train = self.data_manager.prepare_training_data(train_data)
        self.ensemble.train_group_b(X_train, y_train)

        X_val, _ = self.data_manager.prepare_training_data(val_history + val_data, lookback=5)
        val_preds_b = self.ensemble.predict_group_b(X_val[-5:])

        # 2. Retrain on Full Data -> Predict Next Round
        print("   > Retraining Group B on Full Data...")
        X_full, y_full = self.data_manager.prepare_training_data(full_data)
        self.ensemble.train_group_b(X_full, y_full)

        last_seq = full_data[-5:]
        X_next_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        next_preds_b = self.ensemble.predict_group_b(X_next_tensor, is_single=True)

        state = {
            'val_preds': val_preds_b,
            'next_preds': next_preds_b
        }
        joblib.dump(state, STATE_B_FILE)
        print(f"✅ Group B Analysis Saved to {STATE_B_FILE}")
        self.log_to_sheet("Analysis_B", "SAVED", "DL Models (9-17) Processed.")
        gc.collect()
        torch.mps.empty_cache()

    def mission_wednesday_final_strike(self):
        print("🚀 Wednesday Mission: Final Strike (PPO + GA + Gemini)")

        # Load States
        if not os.path.exists(STATE_A_FILE) or not os.path.exists(STATE_B_FILE):
            print("❌ Missing State Files! Cannot proceed.")
            self.log_to_sheet("FinalStrike", "FAIL", "Missing State Files")
            return

        state_a = joblib.load(STATE_A_FILE)
        state_b = joblib.load(STATE_B_FILE)

        # 1. PPO Weight Calculation
        # Compare val_preds (A & B) with val_targets (from A)
        val_targets = state_a['val_targets'] # List of lists
        all_val_preds = {**state_a['val_preds'], **state_b['val_preds']}

        weights = self.calculate_ppo_weights(all_val_preds, val_targets)
        print(f"⚖️ PPO Weights: {weights}")

        # 2. Combine Predictions
        all_next_preds = {**state_a['next_preds'], **state_b['next_preds']}
        final_probs = np.zeros(45)

        for name, pred_probs in all_next_preds.items():
            w = weights.get(name, 1.0)
            final_probs += pred_probs * w

        final_probs /= len(all_next_preds)

        # 3. Genetic Evolution
        ga = GeneticEvolution(final_probs, population_size=1000, generations=500)
        elite_candidates = ga.evolve()

        # 4. Gemini Filter
        # Need recent data for context
        full_data = self.data_manager.fetch_data()
        last_seq = [d['nums'] for d in full_data[-5:]]

        gemini_filter = GeminiStrategyFilter(self.genai_model)
        final_games = gemini_filter.filter_candidates(elite_candidates, last_seq)

        # 5. Report
        self.update_report(final_games)
        print("✅ Final Strike Complete.")

        # Cleanup
        if os.path.exists(STATE_A_FILE): os.remove(STATE_A_FILE)
        if os.path.exists(STATE_B_FILE): os.remove(STATE_B_FILE)

    def calculate_ppo_weights(self, all_preds, targets):
        # all_preds: {model_name: np.array(5, 45)}
        # targets: list of 5 lists
        weights = {}
        total_score = 0

        for name, preds in all_preds.items():
            score = 0
            for i in range(len(targets)):
                target_set = set(targets[i])
                # Top 15 prediction
                top_15 = preds[i].argsort()[::-1][:15]
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

        for r in range(last_round + 1, expected_round + 1):
            print(f"🔍 Fetching Round {r}...")
            data = self.fetch_lotto_data_via_gemini(r)
            if data:
                self.data_manager.update_sheet_row(data)
                print(f"💾 Updated Round {r}")
            time.sleep(2)

    def fetch_lotto_data_via_gemini(self, round_no):
        url = f"https://search.naver.com/search.naver?query=로또+{round_no}회+당첨번호"
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()[:5000]
            prompt = f"Extract Lotto data for round {round_no} from text into JSON. Fields: drwNo(int), drwtNo1..6(int), bnusNo(int), firstAccumamnt(int), firstPrzwnerCo(int), drwNoDate(str YYYY-MM-DD). Text: {text}"
            res = self.genai_model.generate_content(prompt)
            json_str = res.text.strip().replace('```json', '').replace('```', '')
            data = json.loads(json_str)
            return data if int(data['drwNo']) == round_no else None
        except Exception as e:
            print(f"❌ Fetch Error: {e}")
            return None

    def update_report(self, games):
        try:
            ws = self.gc.open(self.sheet_name).worksheet(REC_SHEET_NAME)
            ws.clear()
            ws.update(range_name='A1', values=[['🏆 Hybrid Sniper V5: Distributed Orchestration']])
            ws.update(range_name='A3', values=[['🔥 Gemini 1.5 Pro Selected Top 10']])
            rows = [[f"Rank {i+1}"] + g for i, g in enumerate(games)]
            ws.update(range_name='A5', values=rows)
            ws.update(range_name='A18', values=[['🚀 AI Future Technology Lab (R&D Insight)']])
            ws.update(range_name='A19', values=[
                ["Strategy", "Weekly Distributed (Sun-Wed) + PPO + GA"],
                ["Status", "Final Strike Complete"],
                ["Safety", "M5 Hardware Integrity Protected"]
            ])
        except Exception as e:
            print(f"Report Error: {e}")

    def log_to_sheet(self, agent, status, msg):
        try:
            ws = self.gc.open(self.sheet_name).worksheet(LOG_SHEET_NAME)
            ws.append_row([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), agent, status, msg])
        except: pass

# --- Data Manager ---
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

    def prepare_training_data(self, data_source, lookback=5):
        X, y = [], []
        if len(data_source) <= lookback: return np.array([]), np.array([])
        numbers = data_source if isinstance(data_source[0], list) else [d['nums'] for d in data_source]
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

# --- Ensemble Engine (Distributed) ---
class EnsemblePredictor:
    def __init__(self):
        self.models = [] # List of (name, model)

    def train_group_a(self, X, y):
        """Train ML Models (RF, XGB, CatBoost, KNN)"""
        self.models = [] # Clear local state
        print("   > [Group A] Starting ML Training...")

        # 1. Random Forest (5 Variations)
        for d in [10, 20, 30, 40, None]:
            rf = RandomForestClassifier(n_estimators=100, max_depth=d, n_jobs=USED_CORES)
            rf.fit(X, y)
            self.models.append((f'RF_d{d}', rf))
            gc.collect()

        # 2. XGBoost (3 Variations)
        for d in [3, 5, 7]:
            xgb_est = xgb.XGBClassifier(n_estimators=50, max_depth=d, n_jobs=USED_CORES, tree_method='hist')
            model = MultiOutputClassifier(xgb_est, n_jobs=USED_CORES)
            model.fit(X, y)
            self.models.append((f'XGB_d{d}', model))
            gc.collect()

        # 3. KNN (5 Variations)
        for k in [3, 5, 7, 9, 11]:
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=USED_CORES)
            knn.fit(X, y)
            self.models.append((f'KNN_k{k}', knn))
            gc.collect()

        print(f"   > [Group A] Trained {len(self.models)} Models.")

    def predict_group_a(self, X_input, is_single=False):
        """Returns dict of {model_name: probs}"""
        preds = {}
        # X_input: (Batch, Features) or (Features,) if single
        if is_single and X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)

        for name, model in self.models:
            probs_raw = np.array(model.predict_proba(X_input))
            # MultiOutput predict_proba -> list of arrays per label
            try:
                # Shape: (45, Samples, 2)
                # We need (Samples, 45) prob of class 1
                p_vec = np.array([col[:, 1] for col in probs_raw]).T
            except:
                # Fallback for some wrappers
                p_vec = np.array([col[0][1] if len(col[0]) > 1 else 0 for col in probs_raw])
                if is_single: p_vec = p_vec.reshape(1, -1)

            if is_single:
                preds[name] = p_vec[0]
            else:
                preds[name] = p_vec
        return preds

    def train_group_b(self, X, y):
        """Train DL Models (LSTM, GRU, CNN)"""
        self.models = []
        print("   > [Group B] Starting DL Training...")

        # Reshape for DL
        X_tensor = torch.tensor(X, dtype=torch.float32).view(len(X), 5, 6).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        ds = TensorDataset(X_tensor, y_tensor)
        dl = DataLoader(ds, batch_size=32, shuffle=True)

        # LSTM (3 Variations)
        for h in [64, 128, 256]:
            lstm = SimpleLSTM(6, h).to(DEVICE)
            train_torch_model(lstm, dl)
            self.models.append((f'LSTM_h{h}', lstm))
            gc.collect()
            time.sleep(0.5)

        # GRU (3 Variations)
        for h in [64, 128, 256]:
            gru = SimpleGRU(6, h).to(DEVICE)
            train_torch_model(gru, dl)
            self.models.append((f'GRU_h{h}', gru))
            gc.collect()
            time.sleep(0.5)

        # CNN (3 Variations)
        for k in [2, 3, 4]:
            cnn = SimpleCNN(k).to(DEVICE)
            train_torch_model(cnn, dl)
            self.models.append((f'CNN_k{k}', cnn))
            gc.collect()
            time.sleep(0.5)

        print(f"   > [Group B] Trained {len(self.models)} Models.")

    def predict_group_b(self, X_input, is_single=False):
        preds = {}
        # X_input: (Batch, Seq, Feat) or (Seq, Feat)
        if isinstance(X_input, np.ndarray):
             # Ensure shape (Batch, 5, 6)
             if is_single:
                 X_input = X_input.reshape(1, 5, 6)
             elif X_input.ndim == 2: # (Samples, Flattened) -> Need to reshape if passed flat?
                 # Assuming passed as flat (Samples, 30) from prepare_training_data
                 X_input = X_input.reshape(len(X_input), 5, 6)

             X_tensor = torch.tensor(X_input, dtype=torch.float32).to(DEVICE)
        else:
            X_tensor = X_input # Assuming already tensor

        for name, model in self.models:
            model.eval()
            with torch.no_grad():
                out = model(X_tensor).cpu().numpy()

            if is_single:
                preds[name] = out[0]
            else:
                preds[name] = out
        return preds

# --- Helpers ---
class SimpleLSTM(nn.Module):
    def __init__(self, i, h):
        super().__init__()
        self.lstm = nn.LSTM(i, h, batch_first=True)
        self.fc = nn.Linear(h, 45)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.sig(self.fc(h[-1]))

class SimpleGRU(nn.Module):
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
    for _ in range(50):
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

class GeneticEvolution:
    def __init__(self, probs, population_size=1000, generations=500):
        self.probs = probs
        self.pop_size = population_size
        self.generations = generations

    def fitness(self, gene):
        return sum(self.probs[n-1] for n in gene)

    def evolve(self):
        print("🧬 Running Genetic Evolution...")
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

            # [Safety] Cooling
            if (g+1) % 50 == 0:
                time.sleep(1.5)
                print(f"   > Gen {g+1} Cooling...")

        scores = [(gene, self.fitness(gene)) for gene in pop]
        scores.sort(key=lambda x: x[1], reverse=True)
        unique = []
        seen = set()
        for gene, s in scores:
            t = tuple(gene)
            if t not in seen:
                unique.append(gene)
                seen.add(t)
            if len(unique) >= 30: break
        return unique

class GeminiStrategyFilter:
    def __init__(self, model): self.model = model
    def filter_candidates(self, candidates, recent):
        if not self.model: return candidates[:10]
        prompt = f"Select 10 best lotto combinations from {candidates} considering recent flow {recent}. Output strictly JSON: {{'games': [[...]]}}"
        try:
            res = self.model.generate_content(prompt)
            data = json.loads(res.text.strip().replace('```json', '').replace('```', ''))
            return data['games']
        except: return candidates[:10]

if __name__ == "__main__":
    # Force Run Mode for testing: python lotto_predict.py --force=Wed
    import sys
    force_day = None
    for arg in sys.argv:
        if arg.startswith("--force="):
            force_day = arg.split("=")[1]

    orchestrator = HybridSniperOrchestrator()
    orchestrator.dispatch_mission(force_day)
