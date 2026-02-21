import gspread
from google.oauth2.service_account import Credentials
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import datetime
import random
import os
from google import genai
import json
from dotenv import load_dotenv

# ==========================================
# [1] 환경 설정 및 장치 확인
# ==========================================
# .env 파일 로드
load_dotenv()
print(f"DEBUG: 로드된 키1: {os.getenv('GEMINI_API_KEY_1')[:10]}...")

# M5 칩(Apple Silicon) 가속 모드 확인
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🚀 학습 장치 설정: {device} (MacBook Pro M5 가속 모드)")

# 구글 서비스 계정 키 경로
KEY_PATH = "/Users/lsj/Desktop/구글 연결 키/creds lotto.json"

# 제미나이 API 키 로드 (멀티 키 로테이션)
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")

API_KEYS = [key for key in [GEMINI_API_KEY_1, GEMINI_API_KEY_2] if key]

if API_KEYS:
    print(f"✅ 제미나이 API 키가 {len(API_KEYS)}개 로드되었습니다.")
else:
    print("⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")

# 학습 시야(Window Size) 설정 - 8가지 관점
SCALES = [10, 50, 100, 200, 300, 500, 700, 1000]

# ==========================================
# [2] 하이브리드 모델 정의 (Tabular-Insight V5)
# ==========================================

# 2-1. [TabNet 응용] Feature-Attention Layer
# 입력 특징(Feature) 간의 중요도를 학습하여 비선형 상호작용을 포착합니다.
class TabularFeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super(TabularFeatureAttention, self).__init__()
        # 각 특징에 대한 가중치 마스크 학습 (0~1 사이 값)
        self.mask = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # 마스크 생성: (batch, seq_len, input_dim)
        mask_val = self.mask(x)
        # 입력 값에 중요도(mask)를 곱하여 중요한 특징을 강조
        return x * mask_val

# 2-2. 기존 LSTM + Self-Attention 구조에 Feature-Attention 추가
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch, seq_len, hidden_size)
        energy = self.projection(encoder_outputs) # (batch, seq_len, 1)
        weights = torch.softmax(energy.squeeze(-1), dim=1) # (batch, seq_len)
        # (batch, 1, seq_len) * (batch, seq_len, hidden_size) -> (batch, 1, hidden_size)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class LottoBrain(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=3, output_size=12):
        super(LottoBrain, self).__init__()
        # [V5 Upgrade] Tabular Feature Attention 도입
        self.feature_attention = TabularFeatureAttention(input_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = SelfAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 1. Feature Attention 적용 (TabNet 개념)
        x = self.feature_attention(x)

        # 2. LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0)) # (batch, seq_len, hidden_size)

        # 3. Temporal Self-Attention
        attn_out, _ = self.attention(lstm_out) # (batch, hidden_size)

        # 4. Final Prediction
        out = self.fc(attn_out)
        return out

# 2-3. [cGAN 응용] 데이터 증강용 생성적 적대 신경망
# 과거 당첨 패턴을 학습하여 "당첨 가능성이 높은 가상의 10만 개 조합"을 생성합니다.
class LottoGenerator(nn.Module):
    def __init__(self, z_dim=16, output_dim=45):
        super(LottoGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid() # 1~45번 번호별 확률 출력 (Multi-label)
        )

    def forward(self, z):
        return self.net(z)

class LottoDiscriminator(nn.Module):
    def __init__(self, input_dim=45):
        super(LottoDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid() # Real(1) or Fake(0)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# [3] 데이터 로드 및 전처리
# ==========================================
def connect_jules():
    """구글 시트 연결 객체 반환"""
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    try:
        if not os.path.exists(KEY_PATH):
            print(f"❌ 인증 파일이 존재하지 않습니다: {KEY_PATH}")
            return None

        creds = Credentials.from_service_account_file(KEY_PATH, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet = client.open("로또 max") 
        return spreadsheet
    except Exception as e:
        print(f"❌ 줄스 연결 실패: {e}")
        return None

def calculate_advanced_features(df):
    """
    기존 9개 컬럼에 추가 3개 컬럼(Gap Analysis, Odd/Even, Sum)을 생성하여 반환
    """
    number_cols = ['1번', '2번', '3번', '4번', '5번', '6번']

    gaps_list = []
    odd_even_list = []
    sum_list = []

    last_seen = {i: -1 for i in range(1, 46)}

    for idx, row in df.iterrows():
        current_nums = [int(row[col]) for col in number_cols]

        # 1. Sum
        current_sum = sum(current_nums)
        sum_list.append(current_sum)

        # 2. Odd/Even Ratio
        odd_count = sum(1 for n in current_nums if n % 2 != 0)
        odd_even_ratio = odd_count / 6.0
        odd_even_list.append(odd_even_ratio)

        # 3. Gap Analysis
        current_gaps = []
        for n in current_nums:
            if last_seen[n] == -1:
                gap = idx
            else:
                gap = idx - last_seen[n]
            current_gaps.append(gap)
            last_seen[n] = idx

        avg_gap = sum(current_gaps) / 6.0
        gaps_list.append(avg_gap)

    df['Average_Gap'] = gaps_list
    df['Odd_Even_Ratio'] = odd_even_list
    df['Sum'] = sum_list

    return df

def load_data():
    """'시트1'에서 로또 데이터 로드 및 전처리"""
    sheet = connect_jules()
    if not sheet: return None

    try:
        ws = sheet.worksheet("시트1")
        data = ws.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])

        # 전처리
        df['당첨자 수'] = df['당첨자 수'].astype(str).str.replace('명', '').str.replace(',', '').astype(float)
        df['1게임당 총 당첨금액'] = df['1게임당 총 당첨금액'].astype(str).str.replace('원', '').str.replace(',', '').astype(float)

        cols = ['1번', '2번', '3번', '4번', '5번', '6번', '보너스', '당첨자 수', '1게임당 총 당첨금액']
        df = df[cols].apply(pd.to_numeric)

        # LSTM 학습용 (과거 -> 최신)
        df_reversed = df.iloc[::-1].reset_index(drop=True)

        # 특성 공학 추가
        df_enhanced = calculate_advanced_features(df_reversed)

        return df_enhanced
    except Exception as e:
        print(f"⚠️ 데이터 로드 중 오류: {e}")
        return None

# ==========================================
# [4] AI 자율 학습 및 예측 파이프라인
# ==========================================

# 4-1. cGAN 학습 및 가상 데이터 생성 함수
def train_cgan_and_generate(df, epochs=500, samples=100000):
    """
    cGAN을 학습하고 10만 개의 가상 당첨 번호를 생성하여
    각 번호의 출현 확률(가중치)을 반환합니다.
    """
    print("\n⚡ [cGAN Data Augmentation] 가상 시뮬레이션 시작...")

    # 실제 당첨 번호 데이터 준비 (One-hot encoding과 유사하게 45차원 벡터화)
    real_data = []
    number_cols = ['1번', '2번', '3번', '4번', '5번', '6번']

    for _, row in df.iterrows():
        vec = np.zeros(45)
        for col in number_cols:
            idx = int(row[col]) - 1 # 0-indexed
            if 0 <= idx < 45:
                vec[idx] = 1.0
        real_data.append(vec)

    real_tensor = torch.tensor(np.array(real_data), dtype=torch.float32).to(device)

    # 모델 초기화
    z_dim = 16
    generator = LottoGenerator(z_dim=z_dim).to(device)
    discriminator = LottoDiscriminator().to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    # 학습 루프 (간소화됨)
    start_time = time.time()
    batch_size = 64

    for epoch in range(epochs):
        # 1. Discriminator 학습
        idx = np.random.randint(0, real_tensor.size(0), batch_size)
        real_batch = real_tensor[idx]

        # Real Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Real
        d_optimizer.zero_grad()
        outputs = discriminator(real_batch)
        d_loss_real = criterion(outputs, real_labels)

        # Train Fake
        z = torch.randn(batch_size, z_dim).to(device)
        fake_batch = generator(z)
        outputs = discriminator(fake_batch.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # 2. Generator 학습
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, z_dim).to(device)
        fake_batch = generator(z)
        outputs = discriminator(fake_batch)

        # Generator는 Discriminator를 속여야 함 (Label=1)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()

    print(f"  └─ cGAN 학습 완료 ({epochs} epochs, {time.time()-start_time:.2f}s)")

    # 10만 개 가상 샘플 생성
    generator.eval()
    with torch.no_grad():
        z_large = torch.randn(samples, z_dim).to(device)
        generated_data = generator(z_large).cpu().numpy() # (100000, 45) 확률값

    # 각 번호별 평균 확률 계산 (가중치로 사용)
    # generated_data는 각 번호가 나올 확률(0~1)을 나타냄
    # 전체 샘플에 대해 평균을 내면, cGAN이 예측하는 해당 번호의 "당첨 가능성"이 됨
    cgan_weights = np.mean(generated_data, axis=0) # (45,)

    # 정규화 (최대값 1.0)
    cgan_weights = cgan_weights / np.max(cgan_weights)

    # 1-indexed 딕셔너리로 변환
    cgan_weight_dict = {i+1: float(cgan_weights[i]) for i in range(45)}
    print(f"  └─ 10만 개 가상 조합 생성 및 패턴 분석 완료.")

    return cgan_weight_dict

# 4-2. LSTM 학습 함수
def train_model(X, y, epochs=1000):
    model = LottoBrain(12, 128, 3, 12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model

# 4-3. 통합 파이프라인
def run_pipeline(df):
    """
    1. cGAN 데이터 증강 및 패턴 학습
    2. LSTM-Attention 8단계 시야 학습
    3. PPO 개념의 동적 가중치(Dynamic Weighting) 적용
    """
    print("\n" + "="*50)
    print("🧠 [Hybrid Sniper V5: Tabular-Insight] 엔진 가동")
    print("="*50)

    # (1) cGAN 기반 가중치 생성
    cgan_weights = train_cgan_and_generate(df)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    results = [] # (prediction_nums, weight)

    for seq_len in SCALES:
        if len(scaled_data) <= seq_len + 5: continue

        print(f"\n🔭 [{seq_len}주 시야] Tabular-Attention 분석 및 PPO 최적화...")

        # (2) Dynamic Weighting (PPO 개념: Reward 기반 Policy 업데이트)
        val_size = 5
        train_data_len = len(scaled_data) - val_size

        X_val_train = []
        y_val_train = []
        for i in range(seq_len, train_data_len):
            X_val_train.append(scaled_data[i-seq_len:i])
            y_val_train.append(scaled_data[i])

        X_val_tensor = torch.tensor(np.array(X_val_train), dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(np.array(y_val_train), dtype=torch.float32).to(device)

        # 검증용 모델 학습
        val_model = train_model(X_val_tensor, y_val_tensor, epochs=300)

        # 최근 5회차 예측 및 보상(Reward) 계산
        val_score = 0
        val_model.eval()
        with torch.no_grad():
            for k in range(val_size):
                idx = train_data_len + k
                input_seq = scaled_data[idx-seq_len:idx]
                input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)

                pred_scaled = val_model(input_tensor).cpu().numpy()
                pred_original = scaler.inverse_transform(pred_scaled)
                actual_original = scaler.inverse_transform(scaled_data[idx].reshape(1, -1))

                pred_nums = set(np.round(pred_original[0][:6]).astype(int))
                actual_nums = set(np.round(actual_original[0][:6]).astype(int))

                match_cnt = len(pred_nums.intersection(actual_nums))
                val_score += match_cnt

        # PPO Policy: 보상이 높을수록 해당 모델(Policy)의 가중치를 높임
        model_weight = 1.0 + (val_score * 0.5)
        print(f"  └─ ⚖️ Policy Weight: {model_weight:.2f} (Reward: {val_score})")

        # (3) 본 학습
        print(f"  └─ 🚀 전체 데이터 실전 학습 중...")
        X_full, y_full = [], []
        for i in range(seq_len, len(scaled_data)):
            X_full.append(scaled_data[i-seq_len:i])
            y_full.append(scaled_data[i])

        X_full_tensor = torch.tensor(np.array(X_full), dtype=torch.float32).to(device)
        y_full_tensor = torch.tensor(np.array(y_full), dtype=torch.float32).to(device)

        final_model = train_model(X_full_tensor, y_full_tensor, epochs=500 if seq_len < 500 else 300)

        # (4) 미래 예측
        final_model.eval()
        with torch.no_grad():
            last_seq = scaled_data[-seq_len:]
            last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
            predicted_scaled = final_model(last_seq_tensor).cpu().numpy()
            predicted_original = scaler.inverse_transform(predicted_scaled)

            lotto_nums = np.round(predicted_original[0][:6]).astype(int)
            lotto_nums = np.clip(lotto_nums, 1, 45)
            unique_nums = np.unique(lotto_nums)

            if len(unique_nums) < 6:
                missing = 6 - len(unique_nums)
                avail = list(set(range(1, 46)) - set(unique_nums))
                filled = random.sample(avail, missing)
                final_nums = sorted(list(unique_nums) + filled)
            else:
                final_nums = sorted(list(unique_nums))

            final_nums = [int(n) for n in final_nums]
            results.append({'nums': final_nums, 'weight': model_weight})

    return results, cgan_weights

# ==========================================
# [5] 제미나이 AI 전략가 (Hyper-Sniper V5 Mode)
# ==========================================
def get_gemini_strategy(scores):
    if not API_KEYS:
        print("⚠️ API 키가 없습니다. 기본 알고리즘으로 전환합니다.")
        return None

    models = [
        'gemini-3-flash-preview',
        'gemini-2.5-flash',
        'gemini-flash-latest'
    ]

    prompt = f"""
    너는 최고의 로또 전략가이자 최첨단 AI 연구원이야.
    이번 주는 **'Hybrid Sniper V5: Tabular-Insight Edition'** 모드로 작동한다.

    [시스템 업그레이드 내역]
    1. **Tabular Feature Attention (TabNet 기반):** 번호 간의 비선형 상호작용을 포착하여 LSTM 입력 전처리 강화.
    2. **cGAN Data Augmentation:** 과거 패턴을 학습한 생성적 적대 신경망이 10만 개의 가상 당첨 조합을 생성하여 필터링 가중치로 사용.
    3. **PPO (Proximal Policy Optimization) Inspired:** 최근 5주 성과(Reward)에 따라 모델별 가중치(Policy)를 동적으로 최적화.

    아래 데이터는 위 기술들이 적용된 최종 번호별 확률 점수야.

    [확률 데이터 (Top 45)]
    {json.dumps(scores)}

    [너의 임무]
    1. 전체 45개 번호 중 이번 주 당첨 확률이 가장 강력한 **'정예 번호 15~20개'**를 엄선하라.
    2. 엄선된 정예 번호 *만을* 사용하여 수학적으로 가장 당첨 확률이 높은 **'최종 10게임'**을 구성하라.
    3. **R&D Insight 섹션 작성:**
       - 이번에 적용된 **TabNet, cGAN, PPO** 기술이 실제 로또 예측에 어떻게 기여했는지, 혹은 향후 어떻게 발전시킬 수 있을지 연구원 관점에서 3줄 요약해줘.

    [출력 형식]
    반드시 JSON 형식으로만 출력해. 설명이나 마크다운 없이 순수 JSON만.
    {{
        "strategy_summary": "전략 요약 텍스트 (한글 3문장 이내)",
        "elite_numbers": [1, 5, 10, ...],
        "final_10_games": [[1, 2, 3, 4, 5, 6], ... (총 10개)],
        "rd_insight": "R&D 제안 내용 (TabNet, cGAN, PPO 언급 필수)"
    }}
    """

    print("\n🤖 [Gemini AI] 'Hyper-Sniper V5' 전략 수립 및 R&D 분석 중...")

    for model_idx, model_name in enumerate(models):
        print(f"🔍 [{model_idx + 1}단계] {model_name} 시도 중...")

        for i, key in enumerate(API_KEYS):
            try:
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                text_content = response.text
                if "```json" in text_content:
                    text_content = text_content.split("```json")[1].split("```")[0].strip()
                elif "```" in text_content:
                    text_content = text_content.split("```")[1].split("```")[0].strip()

                result = json.loads(text_content)

                if "final_10_games" in result and len(result["final_10_games"]) > 0:
                    print(f"✨ [최종 승인] '{model_name}' 엔진이 전략을 확정했습니다.")
                    return result

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Quota exceeded" in error_msg:
                    break

                print(f"❌ Key {i+1} 호출 실패 ({model_name}): {error_msg}")
                if i < len(API_KEYS) - 1:
                    time.sleep(2) # 짧은 대기

    print("⚠️ 모든 모델 및 API 키 시도 실패.")
    return None

# ==========================================
# [6] AI 자율 필터링 및 게임 생성
# ==========================================
def analyze_and_generate(results, cgan_weights, df):
    """
    LSTM Ensemble 결과 + cGAN 가중치 -> 최종 점수 산출
    """
    print("\n" + "="*50)
    print("🤖 [AI 자율 필터링] 확률 데이터 분석 및 게임 생성")
    print("="*50)

    # 1. 통합 점수 계산
    scores = {i: 0.0 for i in range(1, 46)}
    
    # (A) Recency Score
    recent_10 = df.iloc[-10:]
    for i, row in enumerate(recent_10.itertuples()):
        weight = i + 1
        # row는 Index, 1번, ... 순서
        nums = [row[1], row[2], row[3], row[4], row[5], row[6]]
        for n in nums:
            scores[int(n)] += weight * 0.5

    # (B) Ensemble Score (LSTM)
    for res in results:
        pred_nums = res['nums']
        weight = res['weight']
        for num in pred_nums:
            scores[int(num)] += 30.0 * weight

    # (C) cGAN Weight 적용 (V5 신규 기능)
    # cGAN이 예측한 패턴에 가중치 부여 (최대 20점 추가)
    for num, weight in cgan_weights.items():
        if num in scores:
            scores[num] += weight * 20.0

    # 2. Gemini AI에게 최종 판단 요청
    gemini_result = get_gemini_strategy(scores)

    if gemini_result:
        print("✨ Gemini가 최종 전략을 확정했습니다.")
        final_games = gemini_result.get('final_10_games', [])
        strategy_summary = gemini_result.get('strategy_summary', "전략 요약 없음")
        elite_nums = gemini_result.get('elite_numbers', [])
        rd_insight = gemini_result.get('rd_insight', "R&D 제안 없음")

        validated_games = []
        for game in final_games:
            game = sorted([int(n) for n in game])
            if len(game) == 6:
                validated_games.append(game)

        while len(validated_games) < 10:
             if len(elite_nums) >= 6:
                 fill_game = sorted([int(n) for n in random.sample(elite_nums, 6)])
                 validated_games.append(fill_game)
             else:
                 validated_games.append([1,2,3,4,5,6])

        return validated_games[:10], len(elite_nums), strategy_summary, rd_insight

    # 3. Fallback
    print("⚠️ Gemini 사용 불가. 자체 Elite-20 알고리즘으로 전환합니다.")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    elite_20 = [num for num, score in sorted_scores[:20]]
    elite_20_int = [int(n) for n in elite_20]

    final_games = []
    attempts = 0
    while len(final_games) < 10 and attempts < 1000:
        attempts += 1
        weights = [scores[n] for n in elite_20_int]
        selected = []
        temp_pool = elite_20_int[:]
        temp_weights = weights[:]

        while len(selected) < 6:
             pick = random.choices(temp_pool, weights=temp_weights, k=1)[0]
             if pick not in selected:
                 selected.append(pick)

        new_game = sorted(selected)
        if new_game not in final_games:
            final_games.append(new_game)

    while len(final_games) < 10:
        final_games.append(final_games[-1] if final_games else [1,2,3,4,5,6])

    return final_games, 20, "📉 Gemini 응답 실패 | 자체 알고리즘 가동", "R&D 데이터 없음"

# ==========================================
# [7] 리포트 작성
# ==========================================
def update_report(games, elite_count, strategy_summary, rd_insight):
    """구글 시트에 10게임 및 R&D 정보 작성"""
    sheet = connect_jules()
    if not sheet: return

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    try:
        ws = sheet.worksheet("추천번호")
    except:
        ws = sheet.add_worksheet(title="추천번호", rows=100, cols=20)

    ws.clear()

    try:
        ws.unmerge_cells('A1:G50')
    except Exception as e:
        print(f"⚠️ 병합 해제 중 경고 (무시 가능): {e}")

    # 데이터 준비
    data = [['' for _ in range(7)] for _ in range(50)]

    # [섹션 1] 타이틀 및 전략
    data[0][0] = f"💰 [Hyper-Sniper V5] Tabular-Insight Edition ({now})"
    data[1][0] = "🧠 이번 주 AI 전략 요약"
    data[2][0] = strategy_summary

    # [섹션 2] 게임 데이터
    headers = ["No.", "A", "B", "C", "D", "E", "F"]
    for j, h in enumerate(headers):
        data[5][j] = h

    for i, game in enumerate(games):
        row_idx = 6 + i
        data[row_idx][0] = f"Game {i+1}"
        for j, num in enumerate(game):
            data[row_idx][j+1] = int(num)

    # [섹션 3] R&D Insight (20행부터)
    rd_start_row = 20
    data[rd_start_row][0] = "🚀 AI Future Technology Lab (R&D Insight)"
    data[rd_start_row + 1][0] = rd_insight

    try:
        ws.update(range_name='A1', values=data)
    except Exception as e:
        print(f"⚠️ 데이터 업데이트 오류: {e}")

    try:
        ws.merge_cells('A1:G1')
        ws.merge_cells('A2:G2')
        ws.merge_cells('A3:G5')
        ws.merge_cells('A21:G21')
        ws.merge_cells('A22:G30')
    except Exception as e:
        print(f"⚠️ 셀 병합 중 경고: {e}")

    print(f"✅ [리포트] 10게임 및 R&D 제안 작성 완료.")

# ==========================================
# [8] AI 진화 제안 생성 (신규 추가)
# ==========================================
def generate_evolution_proposal(api_keys):
    """
    현재 코드를 분석하고 TabNet, cGAN, PPO 등을 적용한 차세대 버전을 제안합니다.
    """
    print("\n" + "="*50)
    print("🧬 [Evolution System] 차세대 코드 진화 프로세스 시작...")

    if not api_keys:
        print("⚠️ API 키가 없어 진화를 수행할 수 없습니다.")
        return

    # 현재 코드 읽기
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            current_code = f.read()
    except Exception as e:
        print(f"⚠️ 현재 코드를 읽을 수 없습니다: {e}")
        return

    # 프롬프트 구성
    prompt = f"""
    당신은 세계 최고의 AI 아키텍트이자 파이썬 전문가입니다.
    현재 실행 중인 로또 예측 시스템('lotto_predict.py')의 전체 코드를 분석하고,
    다음 단계로 진화시킨 '완전한 실행 가능한 파이썬 스크립트'를 작성하십시오.

    [진화 목표]
    아래 기술 중 하나를 선택하여 심도 있게 구현하십시오 (TabNet, cGAN 개선, PPO 강화 중 택 1).
    1. **TabNet (Tabular-Insight 강화):** 기존 FeatureAttention을 더 정교한 TabNet 구조(Attentive Transformer, Feature Transformer)로 업그레이드.
    2. **cGAN (Data Augmentation 고도화):** Generator/Discriminator 구조를 개선하거나 WGAN-GP 등을 도입하여 학습 안정성 확보.
    3. **PPO (Reinforcement Learning):** 단순 가중치 조정을 넘어, 강화학습 에이전트가 하이퍼파라미터나 모델 선택을 수행하도록 개선.

    [필수 요구사항]
    1. **기존 기능 완벽 유지:**
       - Apple Silicon (M5) mps 가속 지원 필수 (`torch.device("mps")`).
       - 구글 시트 연동 (gspread), 8단계 시야, Gap 분석 등 기존 로직 유지.
       - .env 환경 변수 로드 및 API 키 처리 로직 유지.
    2. **전체 코드 생성:** 부분 수정이 아닌, 'import'부터 'if __name__'까지 전체 코드를 출력해야 합니다.
    3. **제안서 헤더 (Docstring) 필수:** 코드 최상단에 아래 형식을 반드시 포함하십시오.
       \"\"\"
       [Evolution Proposal]
       - Key Change: <핵심 변경 사항 1줄 요약>
       - Expected Benefit: <기대 효과 1줄 요약>
       - Technical Details: <적용된 기술에 대한 상세 설명>
       \"\"\"

    [출력 형식]
    - 마크다운(```python ... ```) 없이 순수 파이썬 코드만 출력하거나, 마크다운이 있다면 파싱 가능한 형태로 제공하십시오.

    [현재 코드 컨텍스트]
    {current_code}
    """

    models = ["gemini-3-flash-preview", "gemini-2.0-flash-exp", "gemini-1.5-pro"]
    generated_code = None
    selected_model = ""

    for model_name in models:
        print(f"🔍 [{model_name}] 진화 모델 시도 중...")
        for key in api_keys:
            try:
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                text_content = response.text
                if "```python" in text_content:
                    generated_code = text_content.split("```python")[1].split("```")[0].strip()
                elif "```" in text_content:
                    generated_code = text_content.split("```")[1].split("```")[0].strip()
                else:
                    generated_code = text_content.strip()

                if generated_code and "import" in generated_code and "if __name__" in generated_code:
                    selected_model = model_name
                    break
                else:
                    print(f"⚠️ 생성된 코드 검증 실패 ({model_name}): import 또는 if __name__ 구문 누락")
            except Exception as e:
                print(f"⚠️ 에러 발생 ({model_name}): {e}")
                continue
        if generated_code:
            break

    if not generated_code:
        print("⚠️ 진화된 코드를 생성하지 못했습니다.")
        return

    # 저장
    os.makedirs("proposals", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"proposals/{timestamp}_proposal.py"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(generated_code)

    print(f"✨ [진화 완료] 새로운 제안서가 도착했습니다: {filename} (Model: {selected_model})")
    print("="*50)


# ==========================================
# [9] 메인 실행부
# ==========================================
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        # 학습 및 예측 (LSTM Ensemble + cGAN)
        results, cgan_weights = run_pipeline(df)

        # AI 분석 및 게임 생성
        final_games, elite_cnt, strategy_summary, rd_insight = analyze_and_generate(results, cgan_weights, df)

        # 결과 출력
        print(f"\n🎲 최종 생성된 10게임 (Hyper-Sniper V5):")
        print(f"📝 전략 요약: {strategy_summary}")
        print(f"💡 R&D Insight: {rd_insight[:50]}...\n")
        for idx, game in enumerate(final_games):
            print(f"  Game {idx+1}: {game}")

        # 리포트 전송
        update_report(final_games, elite_cnt, strategy_summary, rd_insight)

        # [NEW] 진화 프로세스 실행
        generate_evolution_proposal(API_KEYS)

    print("\n" + "="*50)
    print("🎉 모든 작업이 완료되었습니다.")
    print("="*50)
