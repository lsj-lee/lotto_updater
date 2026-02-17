import gspread
from google.oauth2.service_account import Credentials
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import datetime
import random
import os

# ==========================================
# [1] 환경 설정 및 장치 확인
# ==========================================
# M5 칩(Apple Silicon) 가속 모드 확인
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🚀 학습 장치 설정: {device} (MacBook Pro M5 가속 모드)")

# 구글 서비스 계정 키 경로
KEY_PATH = "/Users/lsj/Desktop/구글 연결 키/creds lotto.json"

# 학습 시야(Window Size) 설정 - 8가지 관점
SCALES = [10, 50, 100, 200, 300, 400, 500, 1000]

# ==========================================
# [2] LottoBrain 모델 정의 (LSTM)
# ==========================================
class LottoBrain(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=3, output_size=9):
        super(LottoBrain, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

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
        creds = Credentials.from_service_account_file(KEY_PATH, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet = client.open("로또 max") 
        return spreadsheet
    except Exception as e:
        print(f"❌ 줄스 연결 실패: {e}")
        return None

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
        return df_reversed
    except Exception as e:
        print(f"⚠️ 데이터 로드 중 오류: {e}")
        return None

# ==========================================
# [4] AI 자율 학습 및 예측 파이프라인
# ==========================================
def run_pipeline(df):
    """8가지 시야(Scale)에 대해 학습 후, 앙상블 예측"""
    print("\n" + "="*50)
    print("🧠 [통합 자율 주행 엔진] 9차원 데이터 학습 및 예측 시작")
    print("="*50)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    predictions = []

    for seq_len in SCALES:
        if len(scaled_data) <= seq_len: continue

        print(f"\n🔭 [{seq_len}주 시야] 9차원 데이터 학습 시작...")
        epochs = 1000 if seq_len < 100 else (500 if seq_len < 500 else 300)

        x_train, y_train = [], []
        for i in range(seq_len, len(scaled_data)):
            x_train.append(scaled_data[i-seq_len:i])
            y_train.append(scaled_data[i])

        x_train = torch.tensor(np.array(x_train), dtype=torch.float32).to(device)
        y_train = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)

        model = LottoBrain(9, 128, 3, 9).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(x_train), y_train)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

        # 예측
        model.eval()
        with torch.no_grad():
            last_seq = scaled_data[-seq_len:]
            last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
            predicted_scaled = model(last_seq_tensor).cpu().numpy()
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

            predictions.append(final_nums)
            print(f"🔮 예측 결과 ({seq_len}주 모델): {final_nums}")

    return predictions

# ==========================================
# [5] AI 자율 필터링 및 게임 생성 (핵심 로직)
# ==========================================
def analyze_and_generate(predictions, df):
    """
    통합 점수 분석 -> 확률의 절벽 발견 -> 하위 번호 제외 -> 15게임 생성
    """
    print("\n" + "="*50)
    print("🤖 [AI 자율 필터링] 확률의 절벽 분석 및 게임 생성")
    print("="*50)

    # 1. 통합 점수 계산
    scores = {i: 0.0 for i in range(1, 46)}
    
    # (A) Recency Score (최근 10회차 가중치)
    recent_10 = df.iloc[-10:]
    for i, row in enumerate(recent_10.itertuples()):
        # 최신일수록 높은 점수 (1점 ~ 10점)
        weight = i + 1
        # itertuples Index=0, columns start from 1.
        # But DataFrame columns are '1번', '2번' etc.
        # Check column index mapping carefully.
        # df structure: '1번' is col 0 in df (after loading).
        # row is a named tuple.
        nums = [row._1, row._2, row._3, row._4, row._5, row._6]
        for n in nums:
            scores[int(n)] += weight * 0.5

    # (B) Ensemble Score (AI 모델 예측 빈도)
    for pred_set in predictions:
        for num in pred_set:
            scores[int(num)] += 30.0  # 모델 예측 번호에 강력한 가중치

    # 2. 확률의 절벽(Probability Cliff) 탐지
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    cliff_idx = -1
    max_drop = -1.0

    # 하위 10개(idx 35) ~ 30개(idx 15) 사이 탐색
    search_start = 15
    search_end = 35

    for i in range(search_start, search_end):
        current_score = sorted_scores[i][1]
        next_score = sorted_scores[i+1][1]
        drop = current_score - next_score

        if drop > max_drop:
            max_drop = drop
            cliff_idx = i

    elite_group_tuples = sorted_scores[:cliff_idx+1]
    elite_group = [num for num, score in elite_group_tuples]
    excluded_group = [num for num, score in sorted_scores[cliff_idx+1:]]

    print(f"📉 확률의 절벽 발견: Rank {cliff_idx+1} (점수 낙폭: {max_drop:.2f})")
    print(f"🚫 제외된 번호 ({len(excluded_group)}개): {excluded_group}")
    print(f"💎 정예 번호 ({len(elite_group)}개): {elite_group[:10]}...")

    # 3. 게임 생성 (15게임)
    final_games = []

    # [Phase 1] 보험용: 1~45번 모든 번호가 최소 1회 포함 (약 8게임)
    all_nums = list(range(1, 46))
    random.shuffle(all_nums)

    chunks = [all_nums[i:i + 6] for i in range(0, len(all_nums), 6)]

    for chunk in chunks:
        if len(chunk) == 6:
            final_games.append(sorted(chunk))
        else:
            # 나머지 처리 (중복 방지 로직 적용)
            remainder = set(chunk)
            needed = 6 - len(remainder)
            fillers = []
            for num in elite_group:
                if num not in remainder:
                    fillers.append(num)
                if len(fillers) == needed:
                    break
            final_games.append(sorted(list(remainder) + fillers))

    # [Phase 2] 정예용: 남은 게임 수만큼 Elite 번호로 채움 (상위 번호 중복 허용)
    attempts = 0
    max_attempts = 1000

    while len(final_games) < 15 and attempts < max_attempts:
        attempts += 1
        weights = [scores[n] for n in elite_group]
        selected = []

        # 번호 6개 뽑기 (한 게임 내 중복 불가)
        temp_weights = weights[:]
        temp_pool = elite_group[:]

        while len(selected) < 6:
            # 가중치 기반 선택
            if sum(temp_weights) == 0: # 예외 처리
                 pick = random.choice(temp_pool)
            else:
                 pick = random.choices(temp_pool, weights=temp_weights, k=1)[0]

            if pick not in selected:
                selected.append(pick)

        new_game = sorted(selected)

        # 게임 간 중복 체크 (Phase 2 내에서는 유니크하게, Phase 1과는 겹쳐도 허용하나 가급적 회피)
        if new_game not in final_games:
            final_games.append(new_game)

    # 만약 루프를 다 돌아도 15개가 안되면 (그럴리 없지만) 중복 허용해서 채움
    while len(final_games) < 15:
        final_games.append(final_games[-1])

    return final_games, len(excluded_group), cliff_idx + 1

# ==========================================
# [6] 리포트 작성 (셀 병합 시각화)
# ==========================================
def update_report(games, excluded_count, cliff_rank):
    """구글 시트에 15게임 및 분석 정보 작성 (병합 적용)"""
    sheet = connect_jules()
    if not sheet: return

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    try:
        ws = sheet.worksheet("추천번호")
    except:
        ws = sheet.add_worksheet(title="추천번호", rows=100, cols=20)

    ws.clear()

    # 데이터 준비 (30행 x 7열)
    data = [['' for _ in range(7)] for _ in range(30)]

    # 타이틀 & 요약
    data[0][0] = f"💰 [AI 자율 필터링] 15게임 최종 리포트 ({now})"
    data[1][0] = f"📉 확률 절벽: Rank {cliff_rank} | 🚫 제외: {excluded_count}수 | 💎 정예 집중 모드"

    # 헤더
    headers = ["No.", "A", "B", "C", "D", "E", "F"]
    for j, h in enumerate(headers):
        data[2][j] = h

    # 게임 데이터 입력 (4행부터)
    for i, game in enumerate(games):
        row_idx = 3 + i
        data[row_idx][0] = f"Game {i+1}"
        for j, num in enumerate(game):
            data[row_idx][j+1] = int(num) # Python int 변환 필수

    # 업데이트
    ws.update(range_name='A1', values=data)

    # 셀 병합 (가독성 극대화)
    try:
        ws.merge_cells('A1:G1') # 메인 타이틀
        ws.merge_cells('A2:G2') # 요약 정보
    except Exception as e:
        print(f"⚠️ 셀 병합 중 경고: {e}")

    print(f"✅ [리포트] 15게임 작성 및 셀 병합 완료.")

# ==========================================
# [7] 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 1. 데이터 로드
    df = load_data()
    if df is not None:
        # 2. 학습 및 예측 (앙상블)
        raw_predictions = run_pipeline(df)

        # 3. AI 분석 및 게임 생성
        final_games, excluded_cnt, cliff_rank = analyze_and_generate(raw_predictions, df)

        # 4. 결과 출력
        print(f"\n🎲 최종 생성된 15게임:")
        for idx, game in enumerate(final_games):
            tag = "[보험]" if idx < 8 else "[정예]"
            print(f"  Game {idx+1} {tag}: {game}")

        # 5. 리포트 전송
        update_report(final_games, excluded_cnt, cliff_rank)

    print("\n" + "="*50)
    print("🎉 모든 작업이 완료되었습니다.")
    print("="*50)
