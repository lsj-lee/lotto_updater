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
# 사용자의 요청에 따라 mps 장치를 우선 사용하며, 없을 경우 cpu로 폴백합니다.
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🚀 학습 장치 설정: {device} (MacBook Pro M5 가속 모드)")

# 구글 서비스 계정 키 경로 (사용자 환경 절대 경로 유지)
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
        # Hidden state, Cell state 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # LSTM 순전파
        out, _ = self.lstm(x, (h0, c0))

        # 마지막 시퀀스의 출력만 사용 (Many-to-One)
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# [3] 줄스(Google Sheets) 접속 및 데이터 로드
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
    if not sheet:
        return None

    try:
        ws = sheet.worksheet("시트1")
        data = ws.get_all_values()

        # 데이터프레임 생성 (헤더 포함)
        df = pd.DataFrame(data[1:], columns=data[0])

        # 전처리: '명', '원', ',' 제거 후 숫자 변환
        df['당첨자 수'] = df['당첨자 수'].astype(str).str.replace('명', '').str.replace(',', '').astype(float)
        df['1게임당 총 당첨금액'] = df['1게임당 총 당첨금액'].astype(str).str.replace('원', '').str.replace(',', '').astype(float)

        # 필요한 9개 컬럼 추출 및 숫자형 변환
        cols = ['1번', '2번', '3번', '4번', '5번', '6번', '보너스', '당첨자 수', '1게임당 총 당첨금액']
        df = df[cols].apply(pd.to_numeric)

        # LSTM 학습을 위해 과거 데이터가 먼저 오도록 역순 정렬 (최신이 마지막에 오도록)
        # 원본 데이터(시트1)는 최신 회차가 상단에 있으므로, 역순으로 뒤집어야 시간 순서가 됨.
        df_reversed = df.iloc[::-1].reset_index(drop=True)

        return df_reversed
    except Exception as e:
        print(f"⚠️ 데이터 로드 중 오류: {e}")
        return None

# ==========================================
# [4] 통합 학습 및 예측 파이프라인
# ==========================================
def run_pipeline():
    """8가지 시야(Scale)에 대해 학습 후, 앙상블 예측"""
    df = load_data()
    if df is None:
        return [], 0.0

    print("\n" + "="*50)
    print("🧠 [통합 자율 주행 엔진] 9차원 데이터 학습 및 예측 시작")
    print("="*50)

    # 데이터 스케일링 (0~1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    predictions = []

    for seq_len in SCALES:
        if len(scaled_data) <= seq_len:
            print(f"⚠️ 데이터 부족으로 스킵: {seq_len}주 시야")
            continue

        print(f"\n🔭 [{seq_len}주 시야] 9차원 데이터 학습 시작...")

        # 에포크 설정 (기존 로직 유지: 짧은 시야는 많이, 긴 시야는 적게)
        epochs = 1000 if seq_len < 100 else (500 if seq_len < 500 else 300)

        # 학습 데이터셋 구성
        x_train = []
        y_train = []
        for i in range(seq_len, len(scaled_data)):
            x_train.append(scaled_data[i-seq_len:i])
            y_train.append(scaled_data[i])

        x_train = torch.tensor(np.array(x_train), dtype=torch.float32).to(device)
        y_train = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)

        # 모델 초기화
        model = LottoBrain(9, 128, 3, 9).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 학습
        model.train()
        start_time = time.time()

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            # 로그 출력 (100 에포크 단위)
            if (epoch+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

        # 모델 저장
        model_name = f"lotto_model_{seq_len}.pth"
        torch.save(model.state_dict(), model_name)
        duration = time.time() - start_time
        print(f"✅ {model_name} 학습 완료 (소요시간: {duration:.1f}초)")

        # [예측] 다음 회차 예측
        model.eval()
        with torch.no_grad():
            last_seq = scaled_data[-seq_len:] # (seq_len, 9)
            last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device) # (1, seq_len, 9)

            predicted_scaled = model(last_seq_tensor).cpu().numpy() # (1, 9)

            # 스케일 역변환
            predicted_original = scaler.inverse_transform(predicted_scaled) # (1, 9)

            # 로또 번호 (앞 6개) 추출 및 정수 반올림
            lotto_nums = predicted_original[0][:6]
            lotto_nums = np.round(lotto_nums).astype(int)

            # 1~45 범위 제한 및 중복 처리
            lotto_nums = np.clip(lotto_nums, 1, 45)
            unique_nums = np.unique(lotto_nums)

            # 중복 제거 후 6개가 안 되면 부족한 개수만큼 랜덤 추가 (기존 번호 제외)
            if len(unique_nums) < 6:
                missing_count = 6 - len(unique_nums)
                available = list(set(range(1, 46)) - set(unique_nums))
                filled = random.sample(available, missing_count)
                final_nums = sorted(list(unique_nums) + filled)
            else:
                final_nums = sorted(list(unique_nums))

            predictions.append(final_nums)
            print(f"🔮 예측 결과 ({seq_len}주 모델): {final_nums}")

    # 조작 의심 지수 계산 (예측된 번호들의 분산 활용)
    if predictions:
        all_nums = [num for sublist in predictions for num in sublist]
        std_dev = np.std(all_nums)
        anomaly_score = round(std_dev, 2)
    else:
        anomaly_score = 0.0

    return predictions, anomaly_score

# ==========================================
# [5] 리포트 작성 (구글 시트)
# ==========================================
def update_jules_report(prediction_list, anomaly_score):
    """추천번호 시트에 결과 작성"""
    sheet = connect_jules()
    if not sheet: return

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    
    try:
        ws_report = sheet.worksheet("추천번호")
    except:
        ws_report = sheet.add_worksheet(title="추천번호", rows=100, cols=20)

    # 시트 초기화 (Clear)
    ws_report.clear()
    print("🧹 [초기화] '추천번호' 시트 내용을 삭제하고 새로 작성을 시작합니다.")

    try:
        # 리포트 데이터 준비 (20행 x 7열)
        report_data = [['' for _ in range(7)] for _ in range(20)]

        # (A) 제목
        report_data[0][0] = "[AI 9차원 앙상블] 주간 분석 리포트"

        # (B) 분석 개요
        report_data[2][0] = "1. 분석 개요"
        report_data[3][0] = f"작성 일시: {now}"
        report_data[3][3] = "분석 모델: 9차원 LSTM 앙상블 (통합 학습)"

        # (C) AI 추천 번호
        report_data[5][0] = "2. AI 추천 번호 (5 Game)"

        # 5세트 번호 입력
        row_offset = 6
        for i, numbers in enumerate(prediction_list):
            if i >= 5: break # 최대 5게임

            report_data[row_offset + i][0] = f"Game {i+1}"
            for j, num in enumerate(numbers):
                if j < 6:
                    report_data[row_offset + i][j+1] = int(num) # numpy int -> int 변환

        # (D) 조작 의심 지수
        sec3_row_idx = 13
        report_data[sec3_row_idx][0] = "3. 조작 의심 지수 (모델 간 변동성)"
        report_data[sec3_row_idx+1][0] = f"Anomaly Score: {anomaly_score}"

        # (E) 시스템 로그
        sec4_row_idx = 16
        report_data[sec4_row_idx][0] = "4. 시스템 로그"
        report_data[sec4_row_idx+1][0] = "M5 9차원 앙상블 완료"
        report_data[sec4_row_idx+1][3] = "자율 주행 성공"

        # 일괄 업데이트 (최신 gspread 문법 적용)
        # DeprecationWarning 방지를 위해 range_name, values 명시
        ws_report.update(range_name='A1', values=report_data)

        # 셀 병합 (A열~G열)
        ws_report.merge_cells('A1:G1')
        ws_report.merge_cells('A3:G3')
        ws_report.merge_cells('A6:G6')
        ws_report.merge_cells('A14:G14')
        ws_report.merge_cells('A17:G17')

        print(f"✅ [리포트] '추천번호' 탭에 5게임 분석 결과 작성 완료 ({now})")

    except Exception as e:
        print(f"⚠️ 리포트 작성 중 오류: {e}")

    # 실행로그 탭 기록
    try:
        try:
            ws_log = sheet.worksheet("실행로그")
        except:
            ws_log = sheet.add_worksheet(title="실행로그", rows=1000, cols=10)

        ws_log.append_row([now, "자율 주행 성공", f"M5 9차원 앙상블 완료 (Score: {anomaly_score})"])
    except:
        pass

# ==========================================
# [6] 메인 실행부
# ==========================================
if __name__ == "__main__":
    print("🚀 AI 분석 및 전송 시스템 가동...")
    
    # 1. 학습 및 예측 수행 (파이프라인 실행)
    raw_predictions, anomaly_val = run_pipeline()

    # 2. 결과 처리 (5게임 선정)
    final_games = []

    # 중복 제거 (리스트는 unhashable하므로 튜플로 변환하여 set 사용)
    unique_preds = set(tuple(p) for p in raw_predictions)
    unique_preds_list = [list(p) for p in unique_preds]

    # 8개 모델의 예측 중 유니크한 것들을 우선 채택
    if len(unique_preds_list) >= 5:
        final_games = unique_preds_list[:5]
    else:
        final_games = unique_preds_list[:]
        # 부족한 게임 수는 랜덤 생성으로 채움 (단, 기존 예측값과 안 겹치게 노력)
        while len(final_games) < 5:
            new_game = sorted(random.sample(range(1, 46), 6))
            if new_game not in final_games:
                final_games.append(new_game)

    # 정렬 (보기 좋게)
    final_games.sort(key=lambda x: x[0]) # 첫 번째 번호 기준 정렬 등

    print(f"\n🎲 최종 선정된 5게임:")
    for idx, game in enumerate(final_games):
        print(f"  Game {idx+1}: {game}")

    # 3. 리포트 전송
    update_jules_report(final_games, anomaly_val)
    
    print("\n" + "="*50)
    print("🎉 모든 작업이 완료되었습니다.")
    print("="*50)
