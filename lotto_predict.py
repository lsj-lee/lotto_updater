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

            # np.int64 -> int 변환
            final_nums = [int(n) for n in final_nums]
            predictions.append(final_nums)
            print(f"🔮 예측 결과 ({seq_len}주 모델): {final_nums}")

    return predictions

# ==========================================
# [5] 제미나이 AI 전략가 (Gemini Strategist)
# ==========================================
def get_gemini_strategy(scores):
    """
    제미나이 AI에게 확률 데이터를 제공하고 최종 15세트와 전략 요약을 요청
    Tiered Model Fallback: gemini-2.0-flash -> gemini-1.5-flash
    """
    if not API_KEYS:
        print("⚠️ API 키가 없습니다. 기본 알고리즘으로 전환합니다.")
        return None

    # [모델 우선순위 설정]
    models = ['gemini-2.0-flash', 'gemini-1.5-flash']

    prompt = f"""
    너는 최고의 로또 전략가야. 아래 데이터는 LSTM 모델들이 분석한 이번 주 로또 번호별 확률 점수야.
    점수가 높을수록 당첨 확률이 높다고 판단된 번호야.

    [확률 데이터]
    {json.dumps(scores)}

    [너의 임무]
    1. 이 데이터를 분석해서, 이번 주에 가장 확률이 낮거나 제외해야 한다고 판단되는 번호들을 10~30개 사이에서 네 직관과 데이터에 기반해 필터링해.
    2. 남은 '정예 번호'들을 조합하여 당첨 확률이 가장 높은 최종 15세트(각 세트 6개 번호)를 구성해줘.
    3. 왜 이 번호들을 필터링했는지, 왜 이 조합이 강력한지 짧은 통찰을 담은 '이번 주 전략 요약'을 한글로 작성해줘 (3문장 이내).

    [출력 형식]
    반드시 JSON 형식으로만 출력해. 설명이나 마크다운 없이 순수 JSON만.
    {{
        "strategy_summary": "전략 요약 텍스트 (한글)",
        "recommended_sets": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], ... (총 15개)]
    }}
    """

    print("\n🤖 [Gemini AI] 전략 수립 중... (최종 판단자)")

    for model_idx, model_name in enumerate(models):
        print(f"🔍 [{model_idx + 1}단계] {model_name} 시도 중...")

        for i, key in enumerate(API_KEYS):
            try:
                # print(f"  🔑 Key {i+1} 시도...") # 디버깅용
                client = genai.Client(api_key=key)

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                # 응답 처리
                text_content = response.text
                if "```json" in text_content:
                    text_content = text_content.split("```json")[1].split("```")[0].strip()
                elif "```" in text_content:
                    text_content = text_content.split("```")[1].split("```")[0].strip()

                result = json.loads(text_content)
                return result

            except Exception as e:
                error_msg = str(e)
                # 429 Error check (Quota exceeded)
                if "429" in error_msg or "Quota exceeded" in error_msg:
                    print(f"🔄 할당량 초과 ({model_name}). 다음 모델로 전환합니다.")
                    break # Break inner key loop to switch model immediately

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
def analyze_and_generate(predictions, df):
    """
    통합 점수 분석 -> (Gemini 또는 확률의 절벽) -> 최종 15게임 생성
    """
    print("\n" + "="*50)
    print("🤖 [AI 자율 필터링] 확률 데이터 분석 및 게임 생성")
    print("="*50)

    # 1. 통합 점수 계산
    scores = {i: 0.0 for i in range(1, 46)}
    
    # (A) Recency Score (최근 10회차 가중치)
    recent_10 = df.iloc[-10:]
    for i, row in enumerate(recent_10.itertuples()):
        weight = i + 1
        nums = [row._1, row._2, row._3, row._4, row._5, row._6]
        for n in nums:
            scores[int(n)] += weight * 0.5

    # (B) Ensemble Score (AI 모델 예측 빈도)
    for pred_set in predictions:
        for num in pred_set:
            scores[int(num)] += 30.0

    # 2. Gemini AI에게 최종 판단 요청
    gemini_result = get_gemini_strategy(scores)

    if gemini_result:
        print("✨ Gemini가 최종 전략을 확정했습니다.")
        final_games = gemini_result['recommended_sets']
        strategy_summary = gemini_result['strategy_summary']

        # 데이터 정합성 체크
        validated_games = []
        for game in final_games:
            game = sorted([int(n) for n in game])
            if len(game) == 6:
                validated_games.append(game)

        # 부족하면 채우기
        while len(validated_games) < 15:
            validated_games.append(validated_games[-1] if validated_games else [1,2,3,4,5,6])

        return validated_games[:15], 0, 0, strategy_summary

    # 3. Fallback: 기존 확률의 절벽(Probability Cliff) 로직
    print("⚠️ Gemini 사용 불가. 기존 알고리즘으로 전환합니다.")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    cliff_idx = -1
    max_drop = -1.0
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

    final_games = []
    # [Phase 1] 보험용
    all_nums = list(range(1, 46))
    random.shuffle(all_nums)
    chunks = [all_nums[i:i + 6] for i in range(0, len(all_nums), 6)]

    for chunk in chunks:
        chunk = [int(n) for n in chunk] # Ensure int
        if len(chunk) == 6:
            final_games.append(sorted(chunk))
        else:
            remainder = set(chunk)
            needed = 6 - len(remainder)
            fillers = []
            for num in elite_group:
                if num not in remainder:
                    fillers.append(int(num))
                if len(fillers) == needed:
                    break
            final_games.append(sorted(list(remainder) + fillers))

    # [Phase 2] 정예용
    attempts = 0
    max_attempts = 1000
    while len(final_games) < 15 and attempts < max_attempts:
        attempts += 1
        weights = [scores[n] for n in elite_group]
        selected = []
        temp_weights = weights[:]
        temp_pool = elite_group[:]

        while len(selected) < 6:
            if sum(temp_weights) == 0:
                 pick = random.choice(temp_pool)
            else:
                 pick = random.choices(temp_pool, weights=temp_weights, k=1)[0]
            if pick not in selected:
                selected.append(int(pick))
        new_game = sorted(selected)
        if new_game not in final_games:
            final_games.append(new_game)

    while len(final_games) < 15:
        final_games.append(final_games[-1])

    default_summary = f"📉 확률 절벽: Rank {cliff_idx+1} | 🚫 제외: {len(excluded_group)}수 | 💎 정예 집중 모드 (Fallback Algorithm)"
    return final_games, len(excluded_group), cliff_idx + 1, default_summary

# ==========================================
# [7] 리포트 작성 (셀 병합 시각화 업데이트)
# ==========================================
def update_report(games, excluded_count, cliff_rank, strategy_summary):
    """구글 시트에 15게임 및 분석 정보 작성 (병합 적용)"""
    sheet = connect_jules()
    if not sheet: return

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    try:
        ws = sheet.worksheet("추천번호")
    except:
        ws = sheet.add_worksheet(title="추천번호", rows=100, cols=20)

    ws.clear()

    # 데이터 준비 (35행 x 7열) - 요약 공간 확보
    data = [['' for _ in range(7)] for _ in range(35)]

    # 타이틀
    data[0][0] = f"💰 [AI 자율 필터링] 15게임 최종 리포트 ({now})"

    # 전략 요약 (헤더 및 내용)
    data[1][0] = "🧠 이번 주 제미나이(Gemini) 전략 요약"
    data[2][0] = strategy_summary

    # 헤더 (6행으로 이동 - index 5)
    headers = ["No.", "A", "B", "C", "D", "E", "F"]
    for j, h in enumerate(headers):
        data[5][j] = h

    # 게임 데이터 입력 (7행부터 - index 6)
    for i, game in enumerate(games):
        row_idx = 6 + i
        data[row_idx][0] = f"Game {i+1}"
        for j, num in enumerate(game):
            data[row_idx][j+1] = int(num) # Python int 변환 필수

    # 업데이트 (Named Arguments 사용)
    try:
        ws.update(range_name='A1', values=data)
    except Exception as e:
        print(f"⚠️ 데이터 업데이트 오류: {e}")

    # 셀 병합 (가독성 극대화)
    try:
        # 1. 메인 타이틀 병합 (A1:G1)
        ws.merge_cells('A1:G1')
        # 2. 전략 요약 헤더 병합 (A2:G2)
        ws.merge_cells('A2:G2')
        # 3. 전략 요약 내용 병합 (A3:G5)
        ws.merge_cells('A3:G5')

    except Exception as e:
        print(f"⚠️ 셀 병합 중 경고: {e}")

    print(f"✅ [리포트] 15게임 작성 및 셀 병합 완료.")

# ==========================================
# [8] 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 1. 데이터 로드
    df = load_data()
    if df is not None:
        # 2. 학습 및 예측 (앙상블)
        raw_predictions = run_pipeline(df)

        # 3. AI 분석 및 게임 생성 (Gemini 통합)
        final_games, excluded_cnt, cliff_rank, strategy_summary = analyze_and_generate(raw_predictions, df)

        # 4. 결과 출력
        print(f"\n🎲 최종 생성된 15게임:")
        print(f"📝 전략 요약: {strategy_summary}\n")
        for idx, game in enumerate(final_games):
            tag = "[보험]" if idx < 8 else "[정예]"
            if excluded_cnt == 0: tag = "[AI추천]"
            print(f"  Game {idx+1} {tag}: {game}")

        # 5. 리포트 전송
        update_report(final_games, excluded_cnt, cliff_rank, strategy_summary)

    print("\n" + "="*50)
    print("🎉 모든 작업이 완료되었습니다.")
    print("="*50)
