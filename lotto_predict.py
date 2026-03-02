# -*- coding: utf-8 -*-
import os
import time
import gc
import random
import json
import datetime
import re
import sys
import traceback
import psutil

# [필수 라이브러리]
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import gspread
from oauth2client.service_account import ServiceAccountCredentials

try:
    from google import genai
except ImportError:
    print("❌ 'google-genai' 라이브러리가 필요합니다. pip install google-genai")
    sys.exit(1)

load_dotenv()

# ==========================================
# ⚙️ [Configuration] 기지 설정 (간소화)
# ==========================================
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID', '1lOifE_xRUocAY_Av-P67uBMKOV1BAb4mMwg_wde_tyA')
CREDS_FILE = 'creds_lotto.json'
SHEET_NAME = '로또 max'
REC_SHEET_NAME = '추천번호'
LOG_SHEET_NAME = '작전로그'
STATE_FILE = 'hybrid_sniper_v5_state.pth'
SNIPER_STATE_JSON = 'sniper_state.json'

# 🚀 M5 하드웨어 가속 설정
USED_CORES = 6
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ==========================================
# 🧠 [Model] Sniper Neural Network
# ==========================================
class SniperModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=128):
        super(SniperModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 45)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# ==========================================
# 🛰️ [Orchestrator] 핵심 엔진
# ==========================================
class LottoOrchestrator:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        try:
            self.client = genai.Client(api_key=self.api_key)
        except:
            self.client = None
        self.model = SniperModel().to(DEVICE)
        self._load_model_state()

    def _load_model_state(self):
        if os.path.exists(STATE_FILE):
            try:
                self.model.load_state_dict(torch.load(STATE_FILE, map_location=DEVICE))
            except: pass

    def get_sheet(self):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
        return gspread.authorize(creds).open_by_key(SPREADSHEET_ID)

    # --- Phase 1: Sync ---
    def sync_data(self):
        print("\n🔄 [Phase 1] 데이터 동기화...")
        # (생략: 기존의 Naver/Gemini 데이터 수집 로직 수행)
        time.sleep(1)
        print("   ✅ 동기화 완료.")

    # --- Phase 2: Train ---
    def train_brain(self):
        print("\n🧠 [Phase 2] M5 가속 학습...")
        self.model.train()
        # (생략: 간단한 학습 루프 수행)
        torch.save(self.model.state_dict(), STATE_FILE)
        print("   ✅ 학습 및 모델 저장 완료.")

    # --- Phase 3: Predict (사령관 지침 반영) ---
    def load_and_predict(self):
        print("\n🔮 [Phase 3] 제미나이 정예 번호 예측...")
        
        # 1. 딥러닝 기반 후보 숫자 추출 (Mock-up)
        top_numbers = random.sample(range(1, 46), 20)
        
        # 2. 제미나이에게 전술 하달 (Prompt)
        prompt = f"""
        당신은 로또 분석 전문가 'Sniper V5'입니다.
        아래 후보 숫자들을 바탕으로 이번 주 최적의 조합을 생성하세요.
        후보 숫자: {top_numbers}

        [지휘관 절대 준수 지침]
        1. 반드시 총 10세트(게임)의 조합을 생성할 것.
        2. 10세트 전체에 사용된 '고유 숫자'의 총 개수가 반드시 25개 이상이 되도록 넓게 분산할 것.
        3. 출력 포맷:
           - 각 세트 번호 나열
           - 마지막에 사용된 주요 숫자별 사유를 '(숫자: 사유)' 형식으로 작성
           - 맨 마지막에 '총 사용 고유 숫자 개수: XX개' 출력
        """
        
        response = self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        result_text = response.text
        
        # 3. 구글 시트 저장 (타임스탬프 포함)
        sh = self.get_sheet()
        try:
            rec_ws = sh.worksheet(REC_SHEET_NAME)
        except:
            rec_ws = sh.add_worksheet(title=REC_SHEET_NAME, rows="100", cols="20")
        
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rec_ws.append_row([now_str, "제미나이 전략 예측 결과", result_text])
        print(f"   ✅ [생성일시: {now_str}] 구글 시트 저장 완료.")

    # --- Phase 4: Evaluate (12명 에러 수리 완료) ---
    def evaluate_performance(self):
        print("\n🏅 [Phase 4] 성과 평가...")
        try:
            sh = self.get_sheet()
            main_ws = sh.get_worksheet(0)
            row = main_ws.row_values(2)
            
            # 🛠️ 수리 포인트: 정규표현식으로 숫자만 추출 ('12명' -> '12')
            def clean_int(val):
                nums = re.sub(r'[^0-9]', '', str(val))
                return int(nums) if nums else 0

            real = set([clean_int(x) for x in row[2:8]])
            print(f"   📊 이번 주 당첨 번호: {real}")
            
            # 성공 로그 기록
            self.log_operation("Phase 4", "SUCCESS", "데이터 클리닝 및 평가 완료")
        except Exception as e:
            print(f"   ❌ 평가 중 오류: {e}")

    def log_operation(self, phase, status, msg):
        try:
            sh = self.get_sheet()
            log_ws = sh.worksheet(LOG_SHEET_NAME)
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_ws.append_row([now, phase, status, msg])
        except: pass

if __name__ == "__main__":
    orc = LottoOrchestrator()
    orc.evaluate_performance()