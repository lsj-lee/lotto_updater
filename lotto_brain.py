import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gspread
import time
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.preprocessing import MinMaxScaler

# [1] 환경 설정 및 장치 확인
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🚀 학습 장치: {device} (MacBook Pro M5 가속 모드)")

key_path = "/Users/lsj/Desktop/구글 연결 키/creds lotto.json"
scales = [10, 50, 100, 200, 300, 400, 500, 1000]

# [2] 9차원 확장 모델 구조 (LottoBrain)
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

# [3] 데이터 로드 및 전처리 (당첨자 수, 금액 포함)
def load_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
    client = gspread.authorize(creds)
    sheet1 = client.open("로또 max").worksheet("시트1")
    
    data = sheet1.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    
    # 문자열 데이터 전처리 ('명', '원', ',' 제거)
    df['당첨자 수'] = df['당첨자 수'].astype(str).str.replace('명', '').str.replace(',', '').astype(float)
    df['1게임당 총 당첨금액'] = df['1게임당 총 당첨금액'].astype(str).str.replace('원', '').str.replace(',', '').astype(float)
    
    # 9개 컬럼 추출 및 숫자형 변환
    df = df[['1번', '2번', '3번', '4번', '5번', '6번', '보너스', '당첨자 수', '1게임당 총 당첨금액']].apply(pd.to_numeric)
    return df.iloc[::-1].reset_index(drop=True)

df = load_data()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

# [4] 멀티 스케일 라이트급 학습 루프
print("\n" + "="*50)
print("🧠 9차원 데이터 라이트급(Lightweight) 학습 루틴을 시작합니다.")
print("="*50)

for seq_len in scales:
    if len(scaled_data) <= seq_len: 
        continue
        
    print(f"\n🔭 [{seq_len}주 시야] 9차원 데이터 학습 시작...")
    
    # [수정 핵심] 에포크 축소로 과적합 방지 및 유연성(Generalization) 확보
    epochs = 1000 if seq_len < 100 else (500 if seq_len < 500 else 300)
    
    x_train, y_train = [], []
    for i in range(seq_len, len(scaled_data)):
        x_train.append(scaled_data[i-seq_len:i])
        y_train.append(scaled_data[i])
    
    x_train = torch.tensor(np.array(x_train), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)

    # 모델 초기화
    model = LottoBrain(9, 128, 3, 9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 실제 학습 진행
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        
        # 학습량이 줄었으므로 100번 단위로 진행 상황 출력 (화면 멈춤 방지)
        if (epoch+1) % 100 == 0: 
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # 뇌(모델) 개별 저장
    model_name = f"lotto_model_{seq_len}.pth"
    torch.save(model.state_dict(), model_name)
    
    duration = time.time() - start_time
    print(f"✅ {model_name} 저장 완료 (소요시간: {duration:.1f}초)")
    
    # M5 칩 휴식 시간 (가벼운 학습이므로 발열이 적어 안전하게 60초 휴식)
    if seq_len != scales[-1]: 
        print("🌡️ 다음 시야로 넘어가기 전 60초간 숨을 고릅니다...")
        time.sleep(60)

print("\n" + "="*50)
print("🎉 모든 라이트급 시야(Scale)에 대한 순차 학습이 완벽하게 끝났습니다!")
print("="*50)