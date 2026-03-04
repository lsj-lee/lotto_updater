# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SniperModel(nn.Module):
    """
    🧠 [설계국] Sniper V7 LSTM 신경망 구조 (시트 연동형)
    """
    def __init__(self, input_size=6, hidden_size=128, dropout=0.2):
        super(SniperModel, self).__init__()
        # 시트에서 받아온 기억력(hidden_size)과 망각률(dropout) 적용
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 45)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def get_device():
    """🚀 M5 하드웨어 가속(MPS) 설정"""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")