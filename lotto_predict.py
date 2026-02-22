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

# Load environment variables
load_dotenv()

# --- Configuration & Constants ---
CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'
LOG_SHEET_NAME = 'Log'
REC_SHEET_NAME = '추천번호'

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
BATCH_SIZE = 32
EPOCHS_MAIN = 200
EPOCHS_CGAN = 100
LEARNING_RATE = 0.001
SEQ_SCALES = [10, 50, 100, 200, 300, 500, 700, 1000] # Full 8 Scales
MAX_SEQ_LEN = 1000
FEATURE_DIM_LOGIC = 3 # Sum, Odd/Even, AC Index

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
                    nums = [int(r[i]) for i in range(2, 8)]
                    round_no = int(r[0])
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
            target_vec[target] = 1.0 # Multi-hot encoding? Or use single label for cGAN?
            # For multi-label prediction, y is (46,).
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
        # Condition on target (class/features). Here condition on y (46-dim multi-hot)
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
        # z: (Batch, Latent), labels: (Batch, 46)
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
        # x: (Batch, 1000, 6)
        batch_size = x.size(0)

        # 8-Scale Processing
        outputs = []
        for scale in SEQ_SCALES:
            # Slice the sequence for this scale
            sub_seq = x[:, -scale:, :] # (Batch, Scale, 6)

            # Embed
            sub_emb = self.embedding(sub_seq).view(batch_size, scale, -1) # (Batch, Scale, 16*6)

            # Process with LSTM (Shared weights across scales)
            # Efficient implementation: We could pack sequences or run sequentially.
            # Running sequentially is fine for 8 scales.
            _, (h_n, _) = self.lstm(sub_emb) # h_n: (1, Batch, 64)
            outputs.append(h_n[-1]) # (Batch, 64)

        # Concatenate outputs from all scales
        combined = torch.cat(outputs, dim=1) # (Batch, 64 * 8)
        return combined

class LogicBranch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogicBranch, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.ReLU(),
            nn.Linear(input_dim*4, input_dim),
            nn.Sigmoid()
        )
        self.process = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        att = self.attention(x)
        return self.process(x * att)

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

        # Branch A: Time (Multi-Scale LSTM)
        self.time_branch = TimeBranch()
        self.time_fc = nn.Linear(64 * len(SEQ_SCALES), 128) # Compress 8 scales

        # Branch B: Logic (TabNet)
        self.logic_branch = LogicBranch(FEATURE_DIM_LOGIC, 32)

        # Branch C: Relation (GNN)
        self.node_emb = nn.Embedding(46, 16)
        self.relation_branch = RelationBranch(46, 16, 32)

        # Decision Head
        self.fc1 = nn.Linear(128 + 32 + 32, 256)
        self.fc2 = nn.Linear(256, 46)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_time, x_logic):
        batch_size = x_time.size(0)

        # A. Time
        time_feats = self.time_branch(x_time) # (Batch, 512)
        time_out = torch.relu(self.time_fc(time_feats)) # (Batch, 128)

        # B. Logic
        logic_out = self.logic_branch(x_logic) # (Batch, 32)

        # C. Relation
        nodes = torch.arange(46, device=DEVICE).unsqueeze(0).repeat(batch_size, 1)
        node_feats = self.node_emb(nodes)
        rel_out = self.relation_branch(self.adj_matrix, node_feats)
        rel_pool = rel_out.mean(dim=1)

        # Concatenate
        combined = torch.cat([time_out, logic_out, rel_pool], dim=1)

        # Decision
        hidden = torch.relu(self.fc1(combined))
        out = self.fc2(hidden)
        return self.sigmoid(out)

# --- Training Engine (IW) ---
class LottoDataset(Dataset):
    def __init__(self, X_time, X_logic, y):
        self.X_time = torch.tensor(X_time, dtype=torch.long)
        self.X_logic = torch.tensor(X_logic, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_time[idx], self.X_logic[idx], self.y[idx]

class LottoTrainer:
    def __init__(self, model, dataset, adj_matrix):
        self.model = model.to(DEVICE)
        self.dataset = dataset
        self.adj_matrix = adj_matrix
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        # MPS Scaler support is tricky, relying on auto-casting or default behavior
        self.scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

    def train_cgan_augmentation(self):
        print("🧬 Training Conditional cGAN for Data Augmentation...")
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

                # Discriminator
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

                # Generator
                g_optimizer.zero_grad()
                d_out_fake_g = discriminator(fake_logic, real_labels)
                g_loss = criterion(d_out_fake_g, torch.ones_like(d_out_fake_g))
                g_loss.backward()
                g_optimizer.step()

        print("✅ cGAN Training Complete. Generating Augmentation Data...")

        # Augment Dataset
        aug_X_time = []
        aug_X_logic = []
        aug_y = []

        # Expand 10x (1 Real + 9 Synthetic)
        all_X_time = self.dataset.X_time
        all_y = self.dataset.y

        for _ in range(9):
            for i in range(len(all_y)):
                # Generate fake logic for existing target
                target = all_y[i].unsqueeze(0).to(DEVICE)
                z = torch.randn(1, 10).to(DEVICE)
                fake_logic = generator(z, target).detach().cpu()

                # Reuse real time sequence (maybe add noise?)
                real_time = all_X_time[i].clone()
                # Simple noise injection to time sequence (randomly change 1 number)
                if random.random() < 0.3:
                    idx = random.randint(0, MAX_SEQ_LEN-1)
                    real_time[idx][random.randint(0, 5)] = random.randint(1, 45)

                aug_X_time.append(real_time)
                aug_X_logic.append(fake_logic.squeeze(0))
                aug_y.append(all_y[i])

        # Merge with original dataset
        aug_dataset = LottoDataset(
            torch.cat([self.dataset.X_time, torch.stack(aug_X_time)]),
            torch.cat([self.dataset.X_logic, torch.stack(aug_X_logic)]),
            torch.cat([self.dataset.y, torch.stack(aug_y)])
        )
        self.dataset = aug_dataset
        print(f"📈 Dataset Expanded: {len(self.dataset)} samples (Original + Augmented)")

    def train_main(self):
        print(f"🔥 Starting Main Training ({EPOCHS_MAIN} epochs) with FP16...")
        data_loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        best_loss = float('inf')

        for epoch in range(EPOCHS_MAIN):
            self.model.train()
            total_loss = 0

            for x_time, x_logic, y in data_loader:
                x_time, x_logic, y = x_time.to(DEVICE), x_logic.to(DEVICE), y.to(DEVICE)

                self.optimizer.zero_grad()

                # FP16 / AMP Context
                # For MPS (Apple Silicon), 'cuda' amp works in some versions, or 'cpu'.
                # PyTorch 2.x supports device_type='mps' for autocast.
                if DEVICE.type == 'cuda' or DEVICE.type == 'mps':
                    with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16):
                        outputs = self.model(x_time, x_logic)
                        loss = self.criterion(outputs, y)

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                else:
                    outputs = self.model(x_time, x_logic)
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), "best_model.pth")

            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")

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
                "Includes: LSTM(8-Scale), TabNet, GNN, cGAN"
            ])
        except Exception as e:
            print(f"Log Error: {e}")

# --- Reporting ---
def generate_recommendations(model, last_time_seq, last_logic, adj):
    model.eval()
    with torch.no_grad():
        x_time = torch.tensor(last_time_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
        x_logic = torch.tensor(last_logic, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        probs = model(x_time, x_logic).squeeze(0).cpu().numpy()

    top_indices = probs.argsort()[-20:][::-1]
    elite = [int(i) for i in top_indices if i > 0][:20]

    games = []
    for _ in range(10):
        games.append(sorted(random.sample(elite, 6)))

    return games, elite

def update_sheet_report(games, elite):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
        gc = gspread.authorize(creds)
        sh = gc.open(SHEET_NAME)
        ws = sh.worksheet(REC_SHEET_NAME)
        ws.clear()

        ws.update('A1', [['🏆 Hybrid Sniper V5: AI Recommendation']])
        ws.update('A3', [['🔥 Elite Candidates (Top 20)']])
        ws.update('A4', [[str(elite)]])

        rows = [[f"Game {i+1}"] + g for i, g in enumerate(games)]
        ws.update('A7', rows)

        ws.update('A20', [['🚀 AI Future Technology Lab (R&D Insight)']])
        ws.update('A21', [
            ["Architecture", "Hybrid Sniper V5 (Multi-Head)"],
            ["Components", "LSTM(8-Scale) + TabNet(Logic) + GNN(Relation)"],
            ["Augmentation", "cGAN (Conditional) + Time Noise"],
            ["Hardware", "Apple M5 MPS Optimized (FP16)"]
        ])
        print("💾 Report updated.")
    except Exception as e:
        print(f"Report Error: {e}")

def main():
    print("🛸 Hybrid Sniper V5 Initializing...")
    dm = LottoDataManager(CREDS_FILE, SHEET_NAME)
    dm.fetch_data()
    adj = dm.get_cooccurrence_matrix()

    X_time, X_logic, y = dm.prepare_dataset()

    if X_time is None: return

    dataset = LottoDataset(X_time, X_logic, y)

    model = HybridSniperV5(adj)
    trainer = LottoTrainer(model, dataset, adj)

    # 1. cGAN Training & Augmentation
    trainer.train_cgan_augmentation()

    # 2. Main Training
    trainer.train_main()

    # 3. Predict
    games, elite = generate_recommendations(model, X_time[-1], X_logic[-1], adj)
    update_sheet_report(games, elite)

if __name__ == "__main__":
    main()
