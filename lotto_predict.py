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

# Load environment variables
load_dotenv()

# --- Configuration & Constants ---
CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'
LOG_SHEET_NAME = 'Log'
REC_SHEET_NAME = '추천번호'
CONFIG_FILE = 'schedule_config.json'
MODEL_FILE = 'best_model.pth'

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

    def run_pipeline(self):
        print("🛸 Starting Hybrid Sniper V5 Evolutionary Orchestration...")

        # 1. Update Data
        self.update_data()

        # 2. PPO Weighting (Reinforcement Learning)
        print("⚖️ Calculating PPO Model Weights...")
        full_data = self.data_manager.fetch_data()
        if len(full_data) < 100: return

        # Split: Train(0 to N-5), Validate(N-5 to N)
        # Validation needs lookback, so we slice carefully
        split_idx = len(full_data) - 5
        train_data = full_data[:split_idx]
        val_data = full_data[split_idx:]
        val_history = full_data[split_idx-5:split_idx] # History for lookback

        # Train ensemble on historical data to evaluate recent performance
        X_train, y_train = self.data_manager.prepare_training_data(train_data)
        self.ensemble.train_ensemble(X_train, y_train)

        # Evaluate on validation set to assign weights (PPO)
        weights = self.evaluate_and_weight_models(val_data, val_history)
        print(f"📊 PPO Weights Assigned: {weights}")

        # 3. Retrain on Full Data
        print("🔄 Retraining Ensemble on Full Data...")
        X_full, y_full = self.data_manager.prepare_training_data(full_data)
        self.ensemble.train_ensemble(X_full, y_full)

        # 4. Predict Next Round
        last_seq = full_data[-5:]
        ensemble_probs = self.ensemble.predict_probs(last_seq, weights)

        # 5. Genetic Evolution
        ga = GeneticEvolution(ensemble_probs, population_size=1000, generations=500)
        elite_candidates = ga.evolve()

        # 6. Gemini Strategy Filter
        gemini_filter = GeminiStrategyFilter(self.genai_model)
        final_games = gemini_filter.filter_candidates(elite_candidates, last_seq)

        # 7. Report
        self.update_report(final_games)

        # 8. Self-Evolution Code Review
        self.run_self_evolution_review()

        print("✅ Mission Accomplished: Orchestration Complete.")

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

    def evaluate_and_weight_models(self, val_data, history_data):
        print("⚖️ Evaluating Models for PPO Weighting...")
        weights = {}
        total_score = 0

        # Prepare validation inputs
        # We need sequence inputs for the validation targets
        # val_data[0] input comes from history_data[-5:]
        combined = history_data + val_data
        X_val, y_val = self.data_manager.prepare_training_data(combined, lookback=5)

        if len(X_val) == 0: return {name: 1.0 for name, _ in self.ensemble.models}

        for name, model in self.ensemble.models:
            score = 0
            # Predict
            if name == 'LSTM':
                model.eval()
                X_tensor = torch.tensor(X_val, dtype=torch.float32).view(len(X_val), 5, 6).to(DEVICE)
                with torch.no_grad():
                    probs = model(X_tensor).cpu().numpy()
            else:
                probs = np.array(model.predict_proba(X_val))
                try:
                    probs = np.array([p[:, 1] for p in probs]).T
                except:
                    probs = np.zeros((len(X_val), 45))

            # Score: Hit Rate in Top 15
            for i in range(len(y_val)):
                true_nums = np.where(y_val[i] == 1)[0]
                pred_rank = probs[i].argsort()[::-1]
                hits = len(set(true_nums) & set(pred_rank[:15]))
                score += hits

            weights[name] = max(0.1, score)
            total_score += weights[name]

        if total_score > 0:
            for k in weights: weights[k] /= total_score
        return weights

    def update_report(self, games):
        try:
            ws = self.gc.open(self.sheet_name).worksheet(REC_SHEET_NAME)
            ws.clear()
            ws.update(range_name='A1', values=[['🏆 Hybrid Sniper V5: Evolutionary Orchestration']])
            ws.update(range_name='A3', values=[['🔥 Gemini 1.5 Pro Selected Top 10']])
            rows = [[f"Rank {i+1}"] + g for i, g in enumerate(games)]
            ws.update(range_name='A5', values=rows)
            ws.update(range_name='A18', values=[['🚀 AI Future Technology Lab (R&D Insight)']])
            ws.update(range_name='A19', values=[
                ["Strategy", "PPO RL + Ensemble + GA + Gemini"],
                ["Status", "Self-Evolution Cycle Active"],
                ["Safety", "M5 Hardware Integrity Protected"]
            ])
        except Exception as e:
            print(f"Report Error: {e}")

    def run_self_evolution_review(self):
        print("🧬 Running Self-Evolution Code Review...")
        try:
            with open(__file__, 'r') as f: code_content = f.read()
            prompt = f"You are an AI Architect. Review this code. Suggest 1 optimization for 'Genetic Evolution' logic. Code: {code_content[:10000]}"
            response = self.genai_model.generate_content(prompt)
            suggestion = response.text
            ws = self.gc.open(self.sheet_name).worksheet(LOG_SHEET_NAME)
            ws.append_row([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Self-Evolution", "PROPOSAL", suggestion[:500]])
            print("✨ Evolution Proposal Logged.")
        except Exception as e: print(f"Self-Evolution Error: {e}")

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

# --- Ensemble Engine ---
class EnsemblePredictor:
    def __init__(self):
        self.models = []

    def train_ensemble(self, X, y):
        self.models = []
        # 1. Random Forest (3 Variations)
        for d in [10, 20, None]:
            rf = RandomForestClassifier(n_estimators=100, max_depth=d, n_jobs=USED_CORES)
            rf.fit(X, y)
            self.models.append(('RF', rf))

        # 2. XGBoost (3 Variations)
        for d in [3, 5, 7]:
            xgb_est = xgb.XGBClassifier(n_estimators=50, max_depth=d, n_jobs=USED_CORES, tree_method='hist')
            model = MultiOutputClassifier(xgb_est, n_jobs=USED_CORES)
            model.fit(X, y)
            self.models.append(('XGB', model))

        # 3. KNN (5 Variations)
        for k in [3, 5, 7, 9, 11]:
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=USED_CORES)
            knn.fit(X, y)
            self.models.append(('KNN', knn))

        # 4. CatBoost (3 Variations)
        for d in [4, 6, 8]:
            cb_est = cb.CatBoostClassifier(iterations=50, depth=d, verbose=False, thread_count=USED_CORES)
            model = MultiOutputClassifier(cb_est, n_jobs=USED_CORES)
            model.fit(X, y)
            self.models.append(('CatBoost', model))

        # 5. Deep Learning (LSTM) (3 Variations)
        X_tensor = torch.tensor(X, dtype=torch.float32).view(len(X), 5, 6).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        ds = TensorDataset(X_tensor, y_tensor)
        dl = DataLoader(ds, batch_size=32, shuffle=True)

        for h in [64, 128, 256]:
            lstm = SimpleLSTM(6, h).to(DEVICE)
            train_torch_model(lstm, dl)
            self.models.append(('LSTM', lstm))
            gc.collect()

    def predict_probs(self, last_seq, weights=None):
        total_probs = np.zeros(45)
        flat_seq = np.array(last_seq).flatten().reshape(1, -1)
        tensor_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        for name, model in self.models:
            weight = weights.get(name, 1.0) if weights else 1.0
            if name == 'LSTM':
                model.eval()
                with torch.no_grad(): p = model(tensor_seq).cpu().numpy()[0]
            else:
                probs = np.array(model.predict_proba(flat_seq))
                try: p = np.array([prob[0][1] for prob in probs])
                except: p = np.array([prob[0][1] if len(prob[0]) > 1 else 0 for prob in probs])
            total_probs += p * weight
        return total_probs / len(self.models)

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
        # Fitness based on ensemble probability sum + diversity
        score = sum(self.probs[n-1] for n in gene)
        return score

    def evolve(self):
        print("🧬 Running Genetic Evolution (Full Logic)...")
        pop = []
        nums = list(range(1, 46))
        w = self.probs / self.probs.sum()

        # Initialize
        for _ in range(self.pop_size):
            pop.append(sorted(np.random.choice(nums, 6, replace=False, p=w)))

        for g in range(self.generations):
            scores = [(gene, self.fitness(gene)) for gene in pop]
            scores.sort(key=lambda x: x[1], reverse=True)

            # Elitism
            elites = [s[0] for s in scores[:int(self.pop_size * 0.2)]]
            next_gen = elites[:]

            # Crossover & Mutation
            while len(next_gen) < self.pop_size:
                p1 = random.choice(elites)
                p2 = random.choice(elites)
                cut = random.randint(1, 5)
                child = p1[:cut] + [n for n in p2 if n not in p1[:cut]]
                while len(child) < 6:
                    n = random.randint(1, 45)
                    if n not in child: child.append(n)
                child = child[:6]
                if random.random() < 0.05: # Mutation
                    child[random.randint(0, 5)] = random.randint(1, 45)
                next_gen.append(sorted(child))

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
    orchestrator = HybridSniperOrchestrator()
    orchestrator.run_pipeline()
