# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SniperModel(nn.Module):
    """
    🧠 [설계국] Sniper V7 하이브리드 신경망 (어텐션 메커니즘 탑재)
    """
    def __init__(self, input_size=9, hidden_size=256, dropout=0.2): # 입력 특성 9개로 확장
        super(SniperModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        
        # 🎯 [신규 탑재] 어텐션(Attention) 레이어: 중요한 과거 시점에 집중
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.fc = nn.Linear(hidden_size, 45)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x) # lstm_out: (batch, seq_len, hidden_size)
        
        # 각 과거 시점(Window)의 중요도 가중치 계산
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1) 
        
        # 가중치를 적용하여 최종 문맥(Context) 파악
        context = torch.sum(attn_weights * lstm_out, dim=1) 
        
        out = self.fc(context)
        return self.sigmoid(out)

def get_device():
    """🚀 M5 하드웨어 가속(MPS) 설정"""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")