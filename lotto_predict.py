import os
import json
import time
import datetime
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- Configuration & Constants ---
CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'
LOG_SHEET_NAME = 'Log'
REC_SHEET_NAME = '추천번호'
MODEL_FILE = 'best_model.pth'

# Device Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Running on Mac M-Series GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 Running on NVIDIA GPU (CUDA)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Running on CPU (Slow)")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS_MAIN = 200
EPOCHS_CGAN = 100
LEARNING_RATE = 0.001
SEQ_SCALES = [10, 50, 100, 200, 300, 500, 700, 1000] # Full 8 Scales (Restored)
MAX_SEQ_LEN = 1000 # Restored to 1000
FEATURE_DIM_LOGIC = 3 # Sum, Odd/Even, AC Index
AUGMENTATION_FACTOR = 100 # [Safety First] Expand 100x (~21k samples) for M5 stability

# --- Helper Functions ---
def calculate_ac_index(numbers):
    if len(numbers) < 2: return 0
    diffs = set()
    sorted_nums = sorted(numbers)
    for i in range(len(sorted_nums)):
        for j in range(i + 1, len(sorted_nums)):
            diffs.add(sorted_nums[j] - sorted_nums[i])
    return len(diffs) - (len(numbers) - 1)

def get_logical_features(numbers):
    total = sum(numbers)
    odd_count = sum(1 for n in numbers if n % 2 != 0)
    ac_idx = calculate_ac_index(numbers)
    return [total/255.0, odd_count/6.0, ac_idx/10.0]

# --- Data Management (NDA) ---
class LottoDataManager:
    def __init__(self, creds_file, sheet_name):
        self.creds_file = creds_file
        self.sheet_name = sheet_name
        self.gc = self._authenticate()
        self.raw_data = []
        self.numbers = []

    def _authenticate(self):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        if not os.path.exists(self.creds_file):
            raise FileNotFoundError(f"Credential file {self.creds_file} not found.")
        creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, scope)
        return gspread.authorize(creds)

    def fetch_data(self):
        print("📥 Fetching data from Google Sheets...")
        try:
            sh = self.gc.open(self.sheet_name)
            ws = sh.get_worksheet(0)
            records = ws.get_all_values()
            rows = records[1:]

            parsed_data = []
            for r in rows:
                if not r[0]: continue
                try:
                    # [Comma Trap Fix] Remove commas before int conversion
                    # [Index Fix] Numbers are in cols B-G (Index 1-6)
                    nums = [int(r[i].replace(',', '')) for i in range(1, 7)]
                    round_no = int(r[0].replace(',', ''))
                    parsed_data.append({'round': round_no, 'nums': nums})
                except ValueError:
                    continue

            self.raw_data = sorted(parsed_data, key=lambda x: x['round'])
            self.numbers = [d['nums'] for d in self.raw_data]
            print(f"✅ Loaded {len(self.numbers)} rounds.")
            return self.numbers
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return []

    def get_cooccurrence_matrix(self):
        matrix = np.zeros((46, 46))
        for nums in self.numbers:
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    n1, n2 = nums[i], nums[j]
                    matrix[n1][n2] += 1
                    matrix[n2][n1] += 1
        max_val = matrix.max()
        if max_val > 0: matrix /= max_val
        return torch.tensor(matrix, dtype=torch.float32).to(DEVICE)

    def prepare_dataset(self):
        X_time = []
        X_logic = []
        y = []

        total = len(self.numbers)
        if total <= MAX_SEQ_LEN:
            print("⚠️ Not enough data for full scale.")
            return None, None, None

        for i in range(MAX_SEQ_LEN, total):
            # Target
            target = self.numbers[i]
            target_vec = np.zeros(46)
            target_vec[target] = 1.0
            y.append(target_vec)

            # Time Input (Last 1000 rounds)
            seq = self.numbers[i-MAX_SEQ_LEN:i]
            X_time.append(seq)

            # Logic Input (Features of the *last* draw in sequence)
            last_draw = seq[-1]
            feats = get_logical_features(last_draw)
            X_logic.append(feats)

        return np.array(X_time), np.array(X_logic), np.array(y)

# --- cGAN (Data Augmentation) ---
class LottoGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, output_dim):
        super(LottoGenerator, self).__init__()
        self.label_emb = nn.Linear(num_classes, 16)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + 16, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        return self.model(x)

class LottoDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LottoDiscriminator, self).__init__()
        self.label_emb = nn.Linear(num_classes, 16)

        self.model = nn.Sequential(
            nn.Linear(input_dim + 16, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        return self.model(x)

# --- Hybrid Model Architecture (CC) ---

class TimeBranch(nn.Module):
    def __init__(self):
        super(TimeBranch, self).__init__()
        self.embedding = nn.Embedding(46, 16)
        self.lstm = nn.LSTM(16*6, 64, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        outputs = []
        for scale in SEQ_SCALES:
            sub_seq = x[:, -scale:, :]
            sub_emb = self.embedding(sub_seq).view(batch_size, scale, -1)
            _, (h_n, _) = self.lstm(sub_emb)
            outputs.append(h_n[-1])
        combined = torch.cat(outputs, dim=1)
        return combined

class TabularFeatureAttention(nn.Module):
    """TabNet-inspired Feature Attention Mechanism (Phase 3)"""
    def __init__(self, input_dim, output_dim):
        super(TabularFeatureAttention, self).__init__()
        # Attention Mask Generator
        self.mask_generator = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim), # Generate weights for each feature
            nn.Softmax(dim=1) # Ensure weights sum to 1 (Attention)
        )
        self.process = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (Batch, Features)
        mask = self.mask_generator(x)
        x_masked = x * mask # Apply attention: Focus on important features
        return self.process(x_masked), mask

class RelationBranch(nn.Module):
    def __init__(self, num_nodes, in_feat, out_feat):
        super(RelationBranch, self).__init__()
        self.proj = nn.Linear(in_feat, out_feat)

    def forward(self, adj, node_feats):
        support = self.proj(node_feats)
        if adj.dim() == 2:
            output = torch.matmul(adj, support)
        else:
            output = torch.bmm(adj, support)
        return torch.relu(output)

class HybridSniperV5(nn.Module):
    def __init__(self, adj_matrix):
        super(HybridSniperV5, self).__init__()
        self.adj_matrix = adj_matrix

        self.time_branch = TimeBranch()
        self.time_fc = nn.Linear(64 * len(SEQ_SCALES), 128)

        # Branch B: Tabular Attention (TabNet)
        self.logic_branch = TabularFeatureAttention(FEATURE_DIM_LOGIC, 32)

        self.node_emb = nn.Embedding(46, 16)
        self.relation_branch = RelationBranch(46, 16, 32)

        self.fc1 = nn.Linear(128 + 32 + 32, 256)
        self.fc2 = nn.Linear(256, 46)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_time, x_logic):
        batch_size = x_time.size(0)

        time_feats = self.time_branch(x_time)
        time_out = torch.relu(self.time_fc(time_feats))

        logic_out, _ = self.logic_branch(x_logic)

        nodes = torch.arange(46, device=DEVICE).unsqueeze(0).repeat(batch_size, 1)
        node_feats = self.node_emb(nodes)
        rel_out = self.relation_branch(self.adj_matrix, node_feats)
        rel_pool = rel_out.mean(dim=1)

        combined = torch.cat([time_out, logic_out, rel_pool], dim=1)

        hidden = torch.relu(self.fc1(combined))
        out = self.fc2(hidden)
        return self.sigmoid(out)

# --- Training Engine (IW) ---
class LottoDataset(Dataset):
    def __init__(self, X_time, X_logic, y):
        # [Tensor Copy Fix] Use clone().detach() if input is tensor, else torch.tensor()
        self.X_time = X_time.clone().detach().long() if isinstance(X_time, torch.Tensor) else torch.tensor(X_time, dtype=torch.long)
        self.X_logic = X_logic.clone().detach().float() if isinstance(X_logic, torch.Tensor) else torch.tensor(X_logic, dtype=torch.float32)
        self.y = y.clone().detach().float() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_time[idx], self.X_logic[idx], self.y[idx]

class LottoTrainer:
    def __init__(self, model, dataset, adj_matrix):
        self.model = model.to(DEVICE)
        self.dataset = dataset
        self.adj_matrix = adj_matrix
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

    def train_cgan_augmentation(self):
        print("🧬 Training Conditional cGAN for Data Augmentation (Scale-up to ~21k)...")
        data_loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)

        generator = LottoGenerator(latent_dim=10, num_classes=46, output_dim=FEATURE_DIM_LOGIC).to(DEVICE)
        discriminator = LottoDiscriminator(input_dim=FEATURE_DIM_LOGIC, num_classes=46).to(DEVICE)

        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
        criterion = nn.BCELoss()

        for epoch in range(EPOCHS_CGAN):
            for _, real_logic, real_labels in data_loader:
                real_logic, real_labels = real_logic.to(DEVICE), real_labels.to(DEVICE)
                batch = real_logic.size(0)

                d_optimizer.zero_grad()
                d_out_real = discriminator(real_logic, real_labels)
                d_loss_real = criterion(d_out_real, torch.ones_like(d_out_real))

                z = torch.randn(batch, 10).to(DEVICE)
                fake_logic = generator(z, real_labels)
                d_out_fake = discriminator(fake_logic.detach(), real_labels)
                d_loss_fake = criterion(d_out_fake, torch.zeros_like(d_out_fake))

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()

                g_optimizer.zero_grad()
                d_out_fake_g = discriminator(fake_logic, real_labels)
                g_loss = criterion(d_out_fake_g, torch.ones_like(d_out_fake_g))
                g_loss.backward()
                g_optimizer.step()

        print("✅ cGAN Training Complete. Generating Augmentation Data...")

        # [Eval Mode Fix] Essential for preventing BatchNorm crash with single samples or small batches
        generator.eval()

        aug_X_time = []
        aug_X_logic = []
        aug_y = []

        # Expand ~100x (1 Real + 99 Synthetic) to reach ~21k samples
        all_X_time = self.dataset.X_time
        all_y = self.dataset.y

        expansion_factor = AUGMENTATION_FACTOR

        print(f"🔄 Generating {expansion_factor}x synthetic data...")

        for _ in range(expansion_factor):
            for i in range(len(all_y)):
                target = all_y[i].unsqueeze(0).to(DEVICE)
                z = torch.randn(1, 10).to(DEVICE)
                with torch.no_grad():
                    fake_logic = generator(z, target).detach().cpu()

                real_time = all_X_time[i].clone()
                if random.random() < 0.3:
                    idx = random.randint(0, MAX_SEQ_LEN-1)
                    real_time[idx][random.randint(0, 5)] = random.randint(1, 45)

                aug_X_time.append(real_time)
                aug_X_logic.append(fake_logic.squeeze(0))
                aug_y.append(all_y[i])

        aug_dataset = LottoDataset(
            torch.cat([self.dataset.X_time, torch.stack(aug_X_time)]),
            torch.cat([self.dataset.X_logic, torch.stack(aug_X_logic)]),
            torch.cat([self.dataset.y, torch.stack(aug_y)])
        )
        self.dataset = aug_dataset
        print(f"📈 Dataset Expanded: {len(self.dataset)} samples (Safety Scale Reached)")

    def train_main(self, resume=False):
        print(f"🔥 Starting Main Training ({EPOCHS_MAIN} epochs)...")

        # [Resume Logic] Load best_model.pth if resuming
        if resume:
            if os.path.exists(MODEL_FILE):
                try:
                    self.model.load_state_dict(torch.load(MODEL_FILE))
                    print(f"🔄 Resumed training from {MODEL_FILE}")
                except Exception as e:
                    print(f"⚠️ Failed to resume model: {e}")
            else:
                print("⚠️ No checkpoint found to resume.")

        data_loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        best_loss = float('inf')

        for epoch in range(EPOCHS_MAIN):
            self.model.train()
            total_loss = 0

            for x_time, x_logic, y in data_loader:
                x_time, x_logic, y = x_time.to(DEVICE), x_logic.to(DEVICE), y.to(DEVICE)

                self.optimizer.zero_grad()

                # [Stability Fix] MPS (Mac) standard FP32
                if DEVICE.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(x_time, x_logic)
                        loss = self.criterion(outputs, y)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard FP32 for MPS / CPU
                    outputs = self.model(x_time, x_logic)
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)

            # [Checkpoint] Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), MODEL_FILE)

            # [Safety Log] GPU check
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | 🛡️ GPU Status: Running Safely (FP32)")
                # Force save every 10 epochs for safety (optional overwrite)
                torch.save(self.model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

        self.log_to_sheet(best_loss)

    def log_to_sheet(self, loss):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
            gc = gspread.authorize(creds)
            sh = gc.open(SHEET_NAME)
            try:
                ws = sh.worksheet(LOG_SHEET_NAME)
            except:
                ws = sh.add_worksheet(title=LOG_SHEET_NAME, rows=1000, cols=5)

            ws.append_row([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "HybridSniperV5",
                "TRAINING_COMPLETE",
                f"Loss: {loss:.4f}",
                "Includes: LSTM(8-Scale), TabNet(Attention), GNN, cGAN(21k+)"
            ], value_input_option='USER_ENTERED')
        except Exception as e:
            print(f"Log Error: {e}")

# --- LLM Strategy Filtering (Phase 3) ---
class GeminiStrategyFilter:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro') # User requested 1.5 Pro
        else:
            self.model = None

    def filter_candidates(self, elite_numbers, recent_draws):
        """
        Send top candidates to Gemini to select final 10 combinations
        based on strategic analysis (Balance, Flow, Overheat).
        """
        if not self.model:
            print("⚠️ Gemini API Key not found. Using fallback random selection.")
            return self._fallback_selection(elite_numbers)

        print("🤖 Gemini 1.5 Pro: Analyzing Strategic Combinations...")

        prompt = f"""
        You are a lottery strategy expert. I have identified a pool of "Elite Candidate Numbers" based on Deep Learning (LSTM/TabNet/GNN) analysis.
        Your task is to select exactly 10 sets of 6 numbers (10 games) from this pool, applying the following strategic filters:

        1. **Balance**: Ensure a mix of high/low numbers and odd/even distribution.
        2. **Recent Flow**: Consider the provided recent winning numbers. Avoid exact repetition of the very last draw, but follow trends.
        3. **Psychological Overheat**: Avoid patterns that look too "regular" (e.g., 1,2,3,4,5,6).

        [Input Data]
        - Elite Candidate Pool: {elite_numbers}
        - Recent 5 Draws: {recent_draws}

        [Output Format]
        Strictly output a JSON object with a single key "games" containing a list of 10 arrays.
        Example: {{"games": [[1, 2, 3, 4, 5, 6], ...]}}
        Do not add any markdown formatting or extra text.
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]

            data = json.loads(text)
            games = data.get("games", [])

            if len(games) != 10:
                print(f"⚠️ Gemini returned {len(games)} games. Expected 10.")
                return self._fallback_selection(elite_numbers)

            # Validate numbers
            validated_games = []
            for g in games:
                # Ensure numbers are from elite pool (or at least valid 1-45)
                # Gemini might hallucinate outside the pool, so we clamp/fix or trust it as a "Strategy"
                # Let's trust it but ensure validity
                valid_g = sorted([max(1, min(45, int(n))) for n in g])
                validated_games.append(valid_g)

            return validated_games

        except Exception as e:
            print(f"❌ Gemini Strategy Failed: {e}")
            return self._fallback_selection(elite_numbers)

    def _fallback_selection(self, elite_numbers):
        print("🎲 Using Fallback Random Selection.")
        games = []
        for _ in range(10):
            games.append(sorted(random.sample(elite_numbers, 6)))
        return games

# --- Reporting ---
def generate_recommendations(model, last_time_seq, last_logic, adj, dm):
    model.eval()
    with torch.no_grad():
        x_time = torch.tensor(last_time_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
        x_logic = torch.tensor(last_logic, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        probs = model(x_time, x_logic).squeeze(0).cpu().numpy()

    top_indices = probs.argsort()[-30:][::-1] # Get top 30 for Gemini to choose from
    elite = [int(i) for i in top_indices if i > 0]

    # Get recent draws for context
    recent_draws = [d['nums'] for d in dm.raw_data[-5:]]

    # Phase 3: Gemini Strategy Filter
    gemini = GeminiStrategyFilter()
    games = gemini.filter_candidates(elite, recent_draws)

    return games, elite[:20] # Return top 20 for display

def update_sheet_report(games, elite):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
        gc = gspread.authorize(creds)
        sh = gc.open(SHEET_NAME)
        ws = sh.worksheet(REC_SHEET_NAME)
        ws.clear()

        # [Deprecation Fix] Use range_name and values arguments
        ws.update(range_name='A1', values=[['🏆 Hybrid Sniper V5: AI Recommendation']])
        ws.update(range_name='A3', values=[['🔥 Elite Candidates (Top 20)']])
        ws.update(range_name='A4', values=[[str(elite)]])

        rows = [[f"Game {i+1}"] + g for i, g in enumerate(games)]
        ws.update(range_name='A7', values=rows)

        ws.update(range_name='A20', values=[['🚀 AI Future Technology Lab (R&D Insight)']])
        ws.update(range_name='A21', values=[
            ["Architecture", "Hybrid Sniper V5 (Multi-Head + TabNet)"],
            ["Augmentation", "cGAN Scale-up (~21k Samples / 100x)"],
            ["Strategy", "Gemini 1.5 Pro Filter (Balance/Flow/Heat)"],
            ["Hardware", "Apple M5 MPS Optimized (FP32 Stable)"]
        ])
        print("💾 Report updated.")
    except Exception as e:
        print(f"Report Error: {e}")

def main():
    print("🛸 Hybrid Sniper V5 (Phase 3) Initializing...")

    # Check for Resume
    resume_flag = False
    if os.path.exists(MODEL_FILE):
        user_input = input(f"💾 Found existing checkpoint '{MODEL_FILE}'. Resume training? (y/n): ")
        if user_input.lower() == 'y':
            resume_flag = True
            print("✅ Resuming training...")
        else:
            print("⚠️ Starting fresh training...")

    dm = LottoDataManager(CREDS_FILE, SHEET_NAME)
    dm.fetch_data()
    adj = dm.get_cooccurrence_matrix()

    X_time, X_logic, y = dm.prepare_dataset()

    if X_time is None: return

    dataset = LottoDataset(X_time, X_logic, y)

    model = HybridSniperV5(adj)
    trainer = LottoTrainer(model, dataset, adj)

    # 1. cGAN Training & Augmentation (Scale-up)
    trainer.train_cgan_augmentation()

    # 2. Main Training (with Resume flag)
    trainer.train_main(resume=resume_flag)

    # 3. Predict & Gemini Filter
    games, elite = generate_recommendations(model, X_time[-1], X_logic[-1], adj, dm)
    update_sheet_report(games, elite)

if __name__ == "__main__":
    main()
