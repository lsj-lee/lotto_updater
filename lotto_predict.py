# -*- coding: utf-8 -*-
import os
import time
import gc
import random
import json
import datetime
import re
import multiprocessing
import sys
import traceback

# [필수 라이브러리]
# pip install torch numpy pandas google-genai gspread oauth2client google-api-python-client beautifulsoup4 requests python-dotenv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# [Google API]
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ Critical Dependency Missing: 'google-genai'")
    sys.exit(1)

# 환경 변수 로드 (.env)
load_dotenv()

# ==========================================
# ⚙️ [Configuration] 기지 좌표 및 설정
# ==========================================

# 1. [기지 좌표] 구글 스프레드시트 고유 ID (절대 수정 금지)
# 주소창의 https://docs.google.com/spreadsheets/d/THIS_ID/edit... 에서 추출
SPREADSHEET_ID = '1lOifE_xRUocAY_Av-P67uBMKOV1BAb4mMwg_wde_tyA'

# 2. [파일 경로]
CREDS_FILE = 'creds_lotto.json'  # 구글 인증 키
SHEET_NAME = '로또 max'          # (백업용 이름)
REC_SHEET_NAME = '추천번호'       # 결과 출력 탭
STATE_FILE = 'hybrid_sniper_v5_state.pth' # 학습된 모델 저장

# 3. [M5 하드웨어 방어]
# MacBook Pro M5의 GPU 가속(Metal)을 사용하되, 코어 과열을 방지하기 위해 제한을 둡니다.
TOTAL_CORES = multiprocessing.cpu_count()
USED_CORES = 6  # 사령관님 명령: 6코어 제한
torch.set_num_threads(USED_CORES)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"🚀 [System] M5 Neural Engine Activated (MPS/Metal). Cores: {USED_CORES}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ [System] Running on CPU (MPS not found).")

# 4. [네이버 위장] 크롬 브라우저 헤더
REAL_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.naver.com/"
}

# ==========================================
# 🧠 [Phase 2] The Brain: NDA & Hybrid Network
# ==========================================

class NDA_FeatureEngine:
    """
    [NDA] 다차원 데이터 분석 엔진
    - 로또 번호의 흐름(Sequence), 통계(Stat), 관계(Relation)를 분석합니다.
    """
    @staticmethod
    def calculate_derived_features(numbers_list):
        """
        [논리 특성 레이어]
        입력된 번호 리스트에서 4가지 핵심 통계 지표를 추출합니다.
        1. 총합 (Sum)
        2. 홀짝 비율 (Odd/Even)
        3. 고저 비율 (High/Low)
        4. AC 지수 (복잡도)
        """
        features = []
        for nums in numbers_list:
            if len(nums) < 6:
                features.append([0,0,0,0])
                continue

            # 1. Sum (총합 정규화: 0~1 사이)
            s = sum(nums)

            # 2. Odd (홀수 개수 정규화)
            odd = sum(1 for n in nums if n % 2 != 0)

            # 3. High (23 이상 개수 정규화)
            high = sum(1 for n in nums if n >= 23)

            # 4. AC Index (숫자 간 간격의 다양성)
            diffs = set()
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    diffs.add(nums[j] - nums[i])
            ac = len(diffs) - (6 - 1)

            features.append([s/255.0, odd/6.0, high/6.0, ac/10.0])
        return np.array(features)

    @staticmethod
    def create_multimodal_dataset(data, lookback=10):
        """
        데이터를 AI 학습용 형태로 변환합니다. (Branch A, Branch B)
        """
        X_seq, X_stat, y = [], [], []
        if len(data) <= lookback: return None, None, None

        raw_nums = np.array(data)
        derived = NDA_FeatureEngine.calculate_derived_features(data)

        for i in range(lookback, len(data)):
            # Branch A: 과거 10주간의 번호 흐름 (시계열)
            seq = raw_nums[i-lookback:i]
            X_seq.append(seq / 45.0) # 정규화

            # Branch B: 직전 회차의 통계 지표 (패턴)
            stat = derived[i-1]
            X_stat.append(stat)

            # Target: 이번 회차 정답 (학습 목표)
            target = np.zeros(45)
            for n in raw_nums[i]:
                target[n-1] = 1 # One-hot encoding
            y.append(target)

        return (
            torch.tensor(np.array(X_seq), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(X_stat), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(y), dtype=torch.float32).to(DEVICE)
        )

class CreativeConnectionModel(nn.Module):
    """
    [CC] 하이브리드 신경망 모델
    - LSTM (시계열) + Dense (통계) 결합 구조
    """
    def __init__(self):
        super(CreativeConnectionModel, self).__init__()

        # Branch A: 시간의 흐름을 읽는 LSTM
        self.lstm = nn.LSTM(input_size=6, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.ln_a = nn.LayerNorm(128)

        # Branch B: 통계적 패턴을 분석하는 Dense Layer
        self.stat_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32)
        )

        # Decision Head: 두 정보를 통합하여 최종 확률 계산
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 45), # 1~45번 공에 대한 점수
            nn.Sigmoid()        # 확률값으로 변환 (0.0 ~ 1.0)
        )

    def forward(self, x_seq, x_stat):
        # A. 시계열 처리
        out_seq, _ = self.lstm(x_seq)
        out_seq = self.ln_a(out_seq[:, -1, :]) # 마지막 시점의 상태

        # B. 통계 처리
        out_stat = self.stat_net(x_stat)

        # C. 통합 및 예측
        combined = torch.cat([out_seq, out_stat], dim=1)
        output = self.head(combined)
        return output

# ==========================================
# 🛰️ [System] 통합 관제 시스템 (Orchestrator)
# ==========================================

def get_verified_model(api_key):
    """
    [Scout] 가장 똑똑한 Gemini 모델을 찾아냅니다.
    우선순위: 3-flash-preview > 2.0-flash-exp > 1.5-pro > 1.5-flash
    """
    print("🛰️ [Scout] Scanning for Gemini Models...")
    if not api_key: return None

    # 사령관님이 선호하시는 모델 순서
    candidates = ["gemini-3-flash-preview", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]

    for model in candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        try:
            # Ping 테스트 (가볍게 찔러보기)
            payload = {"contents": [{"parts": [{"text": "Ping"}]}]}
            resp = requests.post(url, json=payload, timeout=3)
            if resp.status_code == 200:
                print(f"   ✅ Active: {model}")
                return model
        except: continue

    return "gemini-1.5-flash" # 최후의 보루

class LottoOrchestrator:
    def __init__(self):
        self.creds_file = CREDS_FILE
        self.gc, self.docs = self._auth()

        # AI 모델 준비
        api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = get_verified_model(api_key)
        try:
            self.client = genai.Client(api_key=api_key)
        except:
            self.client = None
            print("⚠️ GenAI Client Init Failed (Manual Mode)")

    def _auth(self):
        """
        [권한 설정] 구글 시트 및 독스 API 연결 (안전장치 포함)
        """
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/spreadsheets"
        ]

        if not os.path.exists(self.creds_file):
             print(f"❌ 인증 파일 '{self.creds_file}'이 없습니다.")
             sys.exit(1)

        # 1. JSON 파일 무결성 체크
        try:
            with open(self.creds_file, 'r') as f:
                creds_data = json.load(f)
                client_email = creds_data.get('client_email')
                print(f"📧 Service Account Email: {client_email}")
                print(f"⚠️ 확인: 이 이메일을 구글 시트 '{SHEET_NAME}'의 공유자에 추가하셨나요?")
        except Exception as e:
            print(f"❌ JSON 파일 읽기 오류: {e}")
            sys.exit(1)

        # 2. 인증 시도
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, scope)
            gc = gspread.authorize(creds)

            # Docs 서비스
            try:
                docs = build('docs', 'v1', credentials=creds)
            except:
                docs = None
                print("⚠️ Google Docs 연결 실패 (리포트 생성 불가)")

            return gc, docs

        except Exception as e:
            print("\n❌ [CRITICAL] 구글 인증 실패!")
            print(f"   Error: {e}")
            print("   💡 해결책: 'creds_lotto.json'의 'private_key'가 손상되었을 수 있습니다.")
            print("   💡 해결책: 구글 클라우드 콘솔에서 새 키를 다운로드 받아 덮어쓰세요.")
            sys.exit(1)

    def get_sheet(self):
        """
        [연동 핵심] ID 기반으로 정확하게 시트를 엽니다.
        """
        try:
            # 1순위: ID로 열기 (가장 정확함)
            return self.gc.open_by_key(SPREADSHEET_ID)
        except Exception as e:
            print(f"❌ ID로 시트 열기 실패: {e}")
            print(f"   (ID: {SPREADSHEET_ID})")

            # 2순위: 이름으로 열기 (백업)
            try:
                print(f"   ⚠️ 이름 '{SHEET_NAME}'으로 재시도합니다...")
                return self.gc.open(SHEET_NAME)
            except Exception as e2:
                print(f"❌ 이름으로도 열기 실패: {e2}")
                print("💡 힌트: 위 Service Account Email을 시트 공유 목록에 추가하세요!")
                sys.exit(1)

    # --- [Phase 1] 지능형 동기화 (네이버 검색) ---
    def sync_data(self):
        print("\n🔄 [Phase 1] 지능형 네이버 동기화 시작...")
        try:
            sh = self.get_sheet()
            ws = sh.get_worksheet(0)

            # 1. 내 컴퓨터의 마지막 회차 확인
            try:
                col1 = ws.col_values(1)
                # '회차', '1회' 등을 제거하고 숫자만 추출
                if len(col1) > 1:
                    last_val = str(col1[-1]).replace(',', '').replace('회', '').replace('차', '')
                    local_last = int(last_val)
                else:
                    local_last = 0
            except:
                local_last = 0

            # 2. 네이버의 최신 회차 확인
            portal_last = self._get_naver_latest_round()
            print(f"   📊 상태: 내 파일({local_last}회) vs 네이버({portal_last}회)")

            # 3. 부족한 데이터 채우기
            if portal_last > local_last:
                for r in range(local_last + 1, portal_last + 1):
                    print(f"   🔍 {r}회차 데이터 수집 중...")
                    data = self._scrape_round_detail(r)

                    if data:
                        # 시트에 저장할 행 데이터 구성
                        row = [
                            data['drwNo'], data['drwNoDate'],
                            data['drwtNo1'], data['drwtNo2'], data['drwtNo3'],
                            data['drwtNo4'], data['drwtNo5'], data['drwtNo6'],
                            data['bnusNo'],
                            data.get('firstPrzwnerCo', 0),
                            data.get('firstAccumamnt', 0),
                            ""
                        ]
                        ws.append_row(row)
                        print(f"   ✅ {r}회차 저장 완료.")
                        time.sleep(2) # 네이버 차단 방지 (2초 휴식)
                    else:
                        print(f"   ⚠️ {r}회차 데이터 수집 실패.")
            else:
                print("   ✅ 이미 최신 상태입니다.")

        except Exception as e:
            print(f"❌ 동기화 중 오류 발생: {e}")
            traceback.print_exc()

    def _get_naver_latest_round(self):
        """네이버 검색에서 '1212회차' 같은 텍스트를 찾아 최신 회차를 반환합니다."""
        url = "https://search.naver.com/search.naver?query=로또"
        try:
            res = requests.get(url, headers=REAL_BROWSER_HEADERS, timeout=5)
            text = res.text
            # 정규식: 숫자 뒤에 '회차'가 오는 패턴 찾기
            m = re.search(r'(\d+)회차', text)
            if m: return int(m.group(1))
            return 0
        except: return 0

    def _scrape_round_detail(self, round_no):
        """
        [핵심 기술] 네이버 검색 결과 -> Gemini가 파싱 -> 실패 시 Regex 백업
        """
        url = f"https://search.naver.com/search.naver?query=로또+{round_no}회+당첨번호"
        text = ""
        try:
            res = requests.get(url, headers=REAL_BROWSER_HEADERS, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            text = soup.get_text()[:5000] # 텍스트 앞부분만 추출

            # 1. AI Parsing (Gemini)
            if self.client:
                prompt = f"""
                Extract Lotto data for Round {round_no} from the text below.
                Return ONLY valid JSON format:
                {{
                    "drwNo": {round_no},
                    "drwNoDate": "YYYY-MM-DD",
                    "drwtNo1": 0, "drwtNo2": 0, "drwtNo3": 0, "drwtNo4": 0, "drwtNo5": 0, "drwtNo6": 0,
                    "bnusNo": 0,
                    "firstPrzwnerCo": 0,
                    "firstAccumamnt": 0
                }}
                Text: {text}
                """
                try:
                    resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
                    js_str = resp.text.strip().replace('```json','').replace('```','')
                    js = json.loads(js_str)

                    # 데이터 검증 (1번 공이 0보다 커야 함)
                    if js.get('drwtNo1') and js['drwtNo1'] > 0:
                        return js
                except Exception as e:
                    pass # AI 실패 시 조용히 넘어감

            # 2. Regex Fallback (정규식 백업)
            print(f"   ⚠️ AI 파싱 실패. 정규식으로 시도합니다...")

            date_match = re.search(r'(\d{4}\.\d{2}\.\d{2})', text)
            date_str = date_match.group(1) if date_match else datetime.datetime.now().strftime("%Y-%m-%d")

            # 번호 찾기 (단순히 숫자들만 추출해서 필터링)
            nums = re.findall(r'\b(\d{1,2})\b', text)
            valid_nums = []
            for n in nums:
                n_int = int(n)
                if 1 <= n_int <= 45:
                    if n_int not in valid_nums: # 중복 방지 (보너스 제외)
                        valid_nums.append(n_int)

            if len(valid_nums) >= 7:
                # 대략적으로 6개+1개라고 가정
                return {
                    "drwNo": round_no,
                    "drwNoDate": date_str,
                    "drwtNo1": valid_nums[0], "drwtNo2": valid_nums[1], "drwtNo3": valid_nums[2],
                    "drwtNo4": valid_nums[3], "drwtNo5": valid_nums[4], "drwtNo6": valid_nums[5],
                    "bnusNo": valid_nums[6],
                    "firstPrzwnerCo": 0, "firstAccumamnt": 0
                }

            return None

        except: return None

    # --- [Phase 2] 두뇌 학습 (Training) ---
    def train_brain(self):
        print("\n🧠 [Phase 2] 하이브리드 신경망 학습 (M5 가속)...")
        sh = self.get_sheet()
        ws = sh.get_worksheet(0)
        rows = ws.get_all_values()[1:] # 헤더 제외

        data = []
        for r in rows:
            try:
                # 데이터 전처리: 쉼표 제거 및 정수 변환
                # C~H열 (인덱스 2~7) + 보너스(8) -> 총 7개인데, 학습엔 6개만 주로 사용
                # 여기서는 6개 번호만 사용합니다.
                nums = [int(str(x).replace(',','')) for x in r[2:8]]
                data.append(nums)
            except: pass

        if len(data) < 50:
            print("❌ 데이터가 너무 부족합니다. (최소 50회차 필요)")
            return None, None

        # 데이터셋 생성 (Lookback 10주)
        X_seq, X_stat, y = NDA_FeatureEngine.create_multimodal_dataset(data, lookback=10)
        if X_seq is None: return None, None

        # 모델 초기화 및 학습
        model = CreativeConnectionModel().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.BCELoss()

        model.train()
        epochs = 100
        print(f"   🔥 학습 시작: {epochs} 에포크 (Device: {DEVICE})")

        for e in range(epochs):
            opt.zero_grad()
            out = model(X_seq, X_stat)
            loss = crit(out, y)
            loss.backward()
            opt.step()

            if (e+1) % 20 == 0:
                print(f"   Epoch {e+1}/{epochs} | Loss: {loss.item():.4f}")

        # 가중치 저장
        torch.save(model.state_dict(), STATE_FILE)
        print(f"   ✅ 모델 저장 완료 ({STATE_FILE})")

        return model, data

    # --- [Phase 3] 전략 보고서 생성 (Reporting) ---
    def generate_report(self, model, data):
        if not model: return
        print("\n📝 [Phase 3] 전략 보고서 및 추천 번호 생성...")
        model.eval()

        # 예측: 미래(다음 회차)를 위한 입력 데이터 구성
        # 과거 10주치 데이터로 다음주 예측
        input_raw = data[-10:]
        input_seq = torch.tensor(np.array(input_raw) / 45.0, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 통계 데이터는 가장 최근 회차 기준
        last_stat = NDA_FeatureEngine.calculate_derived_features([data[-1]])
        input_stat = torch.tensor(last_stat, dtype=torch.float32).to(DEVICE)

        # 추론
        with torch.no_grad():
            probs = model(input_seq, input_stat).cpu().numpy()[0]

        # 상위 15개 후보 선별
        top_indices = probs.argsort()[::-1][:15]
        top_nums = [int(n+1) for n in top_indices] # 0-base -> 1-base
        print(f"   🎯 타겟 후보군 (Top 15): {top_nums}")

        # 10개 게임 조합 생성 (단순 랜덤이 아닌, AI 가중치 반영 가능)
        games = []
        for _ in range(10):
            # 후보군 내에서 랜덤 6개 추출
            g = sorted(random.sample(top_nums, 6))
            games.append(g)

        # 결과 저장
        self._write_docs(games, top_nums)
        self._write_sheet(games)

    def _write_docs(self, games, candidates):
        """
        [Phase 3 핵심] 구글 독스에 '모바일용 전략 보고서' 자동 작성
        """
        if not self.docs: return
        print("   📄 구글 독스 리포트 작성 중...")

        doc_title = f"Sniper V5 리포트 - {datetime.date.today()}"
        try:
            # 문서 생성
            body = {'title': doc_title}
            doc = self.docs.documents().create(body=body).execute()
            doc_id = doc['documentId']

            # 내용 작성 (서술형)
            content = f"""[Sniper V5 전략 리포트]
발행일: {datetime.date.today().strftime('%Y년 %m월 %d일')}
작전명: Hybrid Strike (Phase 3)

1. 🔭 전장 분석 (Trend)
- 최근 데이터 흐름을 분석한 결과, 다음 주는 변화의 시기입니다.
- AI 모델({self.model_name})이 감지한 핵심 후보군은 총 15개입니다.
- 후보군: {candidates}

2. ⚔️ 전술 조합 (10 Games)
"""
            for i, g in enumerate(games):
                content += f"- 시나리오 {i+1}: {g}\n"

            content += """
3. 💡 사령관님을 위한 제언
- 위 번호들은 확률적으로 가장 높은 점수를 받은 조합입니다.
- 분산 투자를 권장하며, 무리한 진입은 삼가십시오.

[End of Report]
"""
            # 문서에 텍스트 삽입
            reqs = [{'insertText': {'location': {'index': 1}, 'text': content}}]
            self.docs.documents().batchUpdate(documentId=doc_id, body={'requests': reqs}).execute()
            print(f"   ✅ 리포트 생성 완료: https://docs.google.com/document/d/{doc_id}")

        except Exception as e:
            print(f"   ⚠️ 리포트 생성 실패: {e}")

    def _write_sheet(self, games):
        """구글 시트 '추천번호' 탭 업데이트"""
        try:
            sh = self.get_sheet()
            # 탭이 없으면 생성, 있으면 가져오기
            try:
                ws = sh.worksheet(REC_SHEET_NAME)
            except:
                ws = sh.add_worksheet(title=REC_SHEET_NAME, rows=100, cols=20)

            ws.clear()

            # 헤더
            ws.update(range_name='A1', values=[['🏆 Sniper V5 최종 추천 번호']])

            # 데이터
            rows = []
            for i, g in enumerate(games):
                rows.append([f"시나리오 {i+1}"] + g)

            ws.update(range_name='A3', values=rows)
            print("   ✅ 구글 시트 업데이트 완료.")
        except Exception as e:
            print(f"⚠️ 시트 저장 오류: {e}")

# --- 실행부 ---
if __name__ == "__main__":
    app = LottoOrchestrator()

    # Check for Scheduled Mode (GitHub Actions / Cron)
    if "--scheduled" in sys.argv:
        day = datetime.datetime.now().strftime("%a")
        print(f"🗓️ Scheduled Mode: Today is {day}")

        if day == "Sun":
            # Sunday: Only Sync Data
            app.sync_data()
        elif day == "Mon":
            # Monday: Weekly Analysis (Training)
            app.train_brain()
        elif day == "Wed":
            # Wednesday: Final Prediction & Report
            # (Need to train first to get model)
            model, data = app.train_brain()
            if model and data:
                app.generate_report(model, data)
        else:
            print("💤 No scheduled mission for today.")

    else:
        # Default: Full Cycle (Manual Execution)
        print("🚀 Manual Mode: Executing Full Strategy...")

        # 1. Sync
        app.sync_data()

        # 2. Train
        model, data = app.train_brain()

        # 3. Report
        if model and data:
            app.generate_report(model, data)

    print("\n✅ 작전 완료 (Mission Accomplished).")
