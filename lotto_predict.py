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
from google import genai
import json
from dotenv import load_dotenv

# ==========================================
# [1] 환경 설정 및 장치 확인
# ==========================================
# .env 파일 로드
load_dotenv()

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

# 학습 시야(Window Size) 설정 - 8가지 관점 (수정됨)
SCALES = [10, 50, 100, 200, 300, 500, 700, 1000]

# ==========================================
# [2] LottoBrain 모델 정의 (LSTM + Attention)
# ==========================================
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
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = SelfAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0)) # (batch, seq_len, hidden_size)
        attn_out, _ = self.attention(lstm_out) # (batch, hidden_size)
        out = self.fc(attn_out)
        return out

# ==========================================
# [3] 데이터 로드 및 전처리 (확장됨)
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
    입력 df는 최신순(행 0이 최신)이라고 가정하지 않고,
    load_data에서 호출 시점의 순서에 맞게 처리.
    여기서는 df가 '과거 -> 최신' 순서로 정렬되어 있다고 가정하고 처리.
    """
    # 1~6번 번호 추출
    number_cols = ['1번', '2번', '3번', '4번', '5번', '6번']

    # 결과 담을 리스트
    gaps_list = []
    odd_even_list = []
    sum_list = []

    # 마지막 출현 시점 기록 (번호 1~45)
    last_seen = {i: -1 for i in range(1, 46)}

    for idx, row in df.iterrows():
        # 현재 회차 번호들
        current_nums = [int(row[col]) for col in number_cols]

        # 1. Sum
        current_sum = sum(current_nums)
        sum_list.append(current_sum)

        # 2. Odd/Even Ratio (홀수 비율)
        odd_count = sum(1 for n in current_nums if n % 2 != 0)
        odd_even_ratio = odd_count / 6.0
        odd_even_list.append(odd_even_ratio)

        # 3. Gap Analysis (이번 회차에 나온 번호들의 평균 미출현 기간)
        # 이번에 나온 번호들이 직전에 언제 나왔었는지 확인
        current_gaps = []
        for n in current_nums:
            if last_seen[n] == -1:
                # 처음 나온 경우, 적당히 큰 값 혹은 인덱스 자체를 gap으로
                gap = idx
            else:
                gap = idx - last_seen[n]
            current_gaps.append(gap)
            # 출현 시점 업데이트 (이번 회차가 idx)
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
        # 원본 데이터가 최신->과거 순이라면 뒤집어야 함.
        # 보통 엑셀/시트는 1회가 맨 아래, 최신이 맨 위인 경우가 많음 (확인 필요).
        # 기존 코드: df_reversed = df.iloc[::-1] -> 즉 원본이 최신->과거 였다는 뜻.
        # 따라서 뒤집으면 과거 -> 최신이 됨.
        df_reversed = df.iloc[::-1].reset_index(drop=True)

        # 특성 공학 추가 (순서가 과거->최신인 상태에서 수행)
        df_enhanced = calculate_advanced_features(df_reversed)

        return df_enhanced
    except Exception as e:
        print(f"⚠️ 데이터 로드 중 오류: {e}")
        return None

# ==========================================
# [4] AI 자율 학습 및 예측 파이프라인 (동적 가중치 적용)
# ==========================================
def train_model(X, y, epochs=1000):
    model = LottoBrain(12, 128, 3, 12).to(device) # input/output 12
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 500 == 0:
             pass # 로그 너무 많아서 생략 가능
    return model

def run_pipeline(df):
    """8가지 시야(Scale)에 대해 학습 후, 앙상블 예측 (Dynamic Weighting 적용)"""
    print("\n" + "="*50)
    print("🧠 [통합 자율 주행 엔진] 12차원 데이터(Gap/Odd/Sum 추가) 학습 및 예측 시작")
    print("="*50)

    scaler = MinMaxScaler()
    # 12개 컬럼 모두 스케일링
    scaled_data = scaler.fit_transform(df.values)

    results = [] # (prediction_nums, weight)

    for seq_len in SCALES:
        if len(scaled_data) <= seq_len + 5: continue

        print(f"\n🔭 [{seq_len}주 시야] 동적 가중치 분석 및 학습 시작...")

        # 1. Dynamic Weighting: 최근 5회차 검증
        # 최근 5개를 검증하기 위해, 데이터의 끝에서 5개를 떼어놓고 학습해본다.
        # Train: 0 ~ (End-5)
        # Val: (End-5) ~ End

        val_size = 5
        train_data_len = len(scaled_data) - val_size

        # 검증용 데이터셋 구성
        X_val_train = []
        y_val_train = []
        for i in range(seq_len, train_data_len):
            X_val_train.append(scaled_data[i-seq_len:i])
            y_val_train.append(scaled_data[i])

        X_val_tensor = torch.tensor(np.array(X_val_train), dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(np.array(y_val_train), dtype=torch.float32).to(device)

        # 검증용 모델 학습 (Epoch 절반만 사용 - 속도 최적화)
        print(f"  └─ 🧪 최근 5회차 검증을 위한 선행 학습 중...")
        val_model = train_model(X_val_tensor, y_val_tensor, epochs=300)

        # 최근 5회차 예측 및 정확도 측정
        val_score = 0
        val_model.eval()
        with torch.no_grad():
            for k in range(val_size):
                # 예측할 시점: train_data_len + k
                # 입력: 그 전 seq_len 개
                idx = train_data_len + k
                input_seq = scaled_data[idx-seq_len:idx]
                input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)

                pred_scaled = val_model(input_tensor).cpu().numpy()
                pred_original = scaler.inverse_transform(pred_scaled)

                # 실제 값
                actual_original = scaler.inverse_transform(scaled_data[idx].reshape(1, -1))

                # 번호 비교 (앞 6개)
                pred_nums = set(np.round(pred_original[0][:6]).astype(int))
                actual_nums = set(np.round(actual_original[0][:6]).astype(int))

                # 맞춘 개수만큼 점수 부여
                match_cnt = len(pred_nums.intersection(actual_nums))
                val_score += match_cnt

        # 가중치 계산 (기본 1.0 + 검증 점수)
        weight = 1.0 + (val_score * 0.5)
        print(f"  └─ ⚖️ 모델 가중치 산출: {weight:.2f} (최근 5회 적중수 합계: {val_score})")

        # 2. 본 학습 (전체 데이터)
        print(f"  └─ 🚀 전체 데이터 실전 학습 중...")
        X_full, y_full = [], []
        for i in range(seq_len, len(scaled_data)):
            X_full.append(scaled_data[i-seq_len:i])
            y_full.append(scaled_data[i])

        X_full_tensor = torch.tensor(np.array(X_full), dtype=torch.float32).to(device)
        y_full_tensor = torch.tensor(np.array(y_full), dtype=torch.float32).to(device)

        final_model = train_model(X_full_tensor, y_full_tensor, epochs=500 if seq_len < 500 else 300)

        # 3. 미래 예측
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
            results.append({'nums': final_nums, 'weight': weight})
            print(f"🔮 예측 결과 ({seq_len}주): {final_nums}")

    return results

# ==========================================
# [5] 제미나이 AI 전략가 (Hyper-Sniper V4 Mode)
# ==========================================
def get_gemini_strategy(scores):
    """
    제미나이 AI에게 '20수 정예 스나이퍼' 전략 및 R&D 인사이트 요청.
    """
    if not API_KEYS:
        print("⚠️ API 키가 없습니다. 기본 알고리즘으로 전환합니다.")
        return None

    # [차세대 모델 우선순위 수정]
    models = [
        'gemini-3-flash-preview', # 1순위
        'gemini-2.5-flash',
        'gemini-flash-latest'
    ]

    prompt = f"""
    너는 최고의 로또 전략가이자 최첨단 AI 연구원이야. 이번 주는 'Hyper-Sniper V4: R&D Edition' 모드로 작동한다.
    아래 데이터는 LSTM-Attention 모델들이 동적 가중치 앙상블로 분석한 이번 주 로또 번호별 확률 점수야.

    [확률 데이터]
    {json.dumps(scores)}

    [너의 임무]
    1. 전체 45개 번호 중 이번 주 당첨 확률이 가장 강력한 '정예 번호 15~20개'를 엄선하라.
    2. 엄선된 15~20개 번호 *만을* 사용하여 수학적으로 가장 당첨 확률이 높은 '최종 10게임'을 구성하라.
    3. R&D 자문:
       - 현재의 LSTM-Attention 구조를 넘어, 당첨률을 높이기 위해 추가할 만한 최신 딥러닝 기법(예: GAN, RL, Transformer 변형 등) 3가지를 추천하고 그 이유를 구체적으로 설명하라.

    [출력 형식]
    반드시 JSON 형식으로만 출력해. 설명이나 마크다운 없이 순수 JSON만.
    {{
        "strategy_summary": "전략 요약 텍스트 (한글 3문장 이내)",
        "elite_numbers": [1, 5, 10, ...],
        "final_10_games": [[1, 2, 3, 4, 5, 6], ... (총 10개)],
        "rd_insight": "R&D 제안 내용 (각 제안은 줄바꿈으로 구분)"
    }}
    """

    print("\n🤖 [Gemini AI] 'Hyper-Sniper V4' 전략 수립 및 R&D 분석 중...")

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
                else:
                    print(f"⚠️ {model_name}: 응답 형식이 올바르지 않습니다.")

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Quota exceeded" in error_msg:
                    if model_idx + 1 < len(models):
                        next_model = models[model_idx + 1]
                        print(f"🔄 [전환] {model_name} 할당량 초과. 더 안정적인 {next_model}로 교체합니다.")
                    else:
                        print(f"⚠️ {model_name} 할당량 초과. 더 이상 사용할 모델이 없습니다.")
                    break # 다음 모델로

                print(f"❌ Key {i+1} 호출 실패 ({model_name}): {error_msg}")
                if i < len(API_KEYS) - 1:
                    print("⏳ 10초 대기 후 다음 키 시도...")
                    time.sleep(10)
                else:
                    print(f"⚠️ {model_name}: 모든 키 실패.")

    print("⚠️ 모든 모델 및 API 키 시도 실패.")
    return None

# ==========================================
# [6] AI 자율 필터링 및 게임 생성 (통합 로직)
# ==========================================
def analyze_and_generate(results, df):
    """
    통합 점수 분석 -> (Gemini Elite-20) -> 최종 10게임 생성
    """
    print("\n" + "="*50)
    print("🤖 [AI 자율 필터링] 확률 데이터 분석 및 게임 생성")
    print("="*50)

    # 1. 통합 점수 계산
    scores = {i: 0.0 for i in range(1, 46)}
    
    # (A) Recency Score (최근 10회차 가중치)
    recent_10 = df.iloc[-10:]
    for i, row in enumerate(recent_10.itertuples()): # row는 namedtuple
        weight = i + 1
        # row의 컬럼명에 따라 접근. load_data에서 컬럼명 변경 없음.
        # df_reversed는 '1번'...'6번' 등을 가짐.
        # itertuples()에서는 한글 컬럼명이 _1, _2 등으로 변환될 수 있음.
        # 안전하게 인덱스로 접근하거나, df 컬럼을 확인해야 함.
        # 여기서는 pandas itertuples 동작 특성상 순서대로 접근
        # (Index, 1번, 2번, 3번, 4번, 5번, 6번, ...)
        # 1번~6번은 index 1~6에 해당.
        nums = [row[1], row[2], row[3], row[4], row[5], row[6]]
        for n in nums:
            scores[int(n)] += weight * 0.5

    # (B) Ensemble Score (AI 모델 예측 빈도 + 가중치)
    for res in results:
        pred_nums = res['nums']
        weight = res['weight']
        for num in pred_nums:
            scores[int(num)] += 30.0 * weight

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

        validated_games = validated_games[:10]

        return validated_games, len(elite_nums), strategy_summary, rd_insight

    # 3. Fallback
    print("⚠️ Gemini 사용 불가. 자체 Elite-20 알고리즘으로 전환합니다.")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    elite_20 = [num for num, score in sorted_scores[:20]]
    elite_20_int = [int(n) for n in elite_20]

    print(f"💎 추출된 정예 20수: {elite_20_int}")

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

    fallback_summary = "📉 Gemini 응답 실패 | 💎 자체 Elite-20 알고리즘 가동"
    return final_games, 20, fallback_summary, "Gemini 연결 실패로 R&D 제안 없음"

# ==========================================
# [7] 리포트 작성 (확장됨)
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

    # 데이터 준비 (50행 x 7열)
    data = [['' for _ in range(7)] for _ in range(50)]

    # [섹션 1] 타이틀 및 전략
    data[0][0] = f"💰 [Hyper-Sniper V4] 10게임 최종 리포트 ({now})"
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
    rd_start_row = 20 # 0-indexed -> 21행
    data[rd_start_row][0] = "🚀 AI 미래 기술 연구소 (R&D Insight)"

    # rd_insight가 길 수 있으므로 여러 줄에 나눠서 넣거나 한 셀에 넣고 병합
    data[rd_start_row + 1][0] = rd_insight

    # 업데이트
    try:
        ws.update(range_name='A1', values=data)
    except Exception as e:
        print(f"⚠️ 데이터 업데이트 오류: {e}")

    # 셀 병합
    try:
        # 타이틀
        ws.merge_cells('A1:G1')
        # 전략 요약 헤더
        ws.merge_cells('A2:G2')
        # 전략 요약 내용
        ws.merge_cells('A3:G5')

        # R&D 타이틀 (21행)
        ws.merge_cells('A21:G21')
        # R&D 내용 (22행~30행)
        ws.merge_cells('A22:G30')

    except Exception as e:
        print(f"⚠️ 셀 병합 중 경고: {e}")

    print(f"✅ [리포트] 10게임 및 R&D 제안 작성 완료.")

# ==========================================
# [8] 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 1. 데이터 로드
    df = load_data()
    if df is not None:
        # 2. 학습 및 예측 (앙상블 + 동적 가중치)
        results = run_pipeline(df)

        # 3. AI 분석 및 게임 생성 (Elite-20 Sniper + R&D)
        final_games, elite_cnt, strategy_summary, rd_insight = analyze_and_generate(results, df)

        # 4. 결과 출력
        print(f"\n🎲 최종 생성된 10게임 (Hyper-Sniper V4):")
        print(f"📝 전략 요약: {strategy_summary}")
        print(f"💡 R&D Insight: {rd_insight[:50]}...\n")
        for idx, game in enumerate(final_games):
            print(f"  Game {idx+1}: {game}")

        # 5. 리포트 전송
        update_report(final_games, elite_cnt, strategy_summary, rd_insight)

    print("\n" + "="*50)
    print("🎉 모든 작업이 완료되었습니다.")
    print("="*50)
