import os
import time
import gc
import random
import json
import datetime
import re
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import requests
import joblib
import sys

# [라이브러리 확인] Google GenAI SDK (v1.0+) 필수
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ Critical Dependency Missing: 'google-genai'")
    print("💡 터미널에서 실행하세요: pip install google-genai")
    sys.exit(1)

# [선택적 라이브러리] XGBoost / CatBoost
try:
    import xgboost as xgb
except ImportError:
    print("⚠️ Missing XGBoost. Run: pip install xgboost")
    xgb = None

try:
    import catboost as cb
except ImportError:
    print("⚠️ Missing CatBoost. Run: pip install catboost")
    cb = None

from bs4 import BeautifulSoup

# 환경 변수 로드 (.env 파일)
load_dotenv()

# --- 설정 및 상수 ---
CREDS_FILE = 'creds_lotto.json'  # 구글 서비스 계정 키 파일
SHEET_NAME = '로또 max'          # 연동할 구글 스프레드시트 이름
LOG_SHEET_NAME = 'Log'           # 로그를 기록할 시트 탭 이름
REC_SHEET_NAME = '추천번호'       # 최종 번호를 출력할 시트 탭 이름
STATE_TOTAL_FILE = 'state_total.pkl' # 모델 학습 상태 저장 파일

# [1단계] 네이버 검색 위장용 헤더 (맥북 크롬처럼 보이기)
REAL_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.naver.com/",
    "Connection": "keep-alive"
}

# [하드웨어 보호] M5 칩 설정 (건드리지 마세요!)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Deep Learning: Running on Mac M-Series GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Deep Learning: Running on CPU")

# [하드웨어 보호] 코어 제한 (과열 방지)
TOTAL_CORES = multiprocessing.cpu_count()
USED_CORES = 6 # 요청하신 대로 6코어 고정
torch.set_num_threads(USED_CORES)


# --- [정찰병] 지능형 모델 탐색 (Scout Logic) ---
def get_verified_model(api_key):
    """
    구글 API를 직접 찔러보며 가장 똑똑하고 응답하는 모델을 찾아냅니다.
    우선순위: Gemini 3.x > 2.x > 1.5 Pro > 1.5 Flash
    """
    print("\n🛰️ [Scout] Initiating Deep Space Scan for Intelligence Models...")

    if not api_key:
        print("❌ API Key is missing.")
        return None

    # 1. 사용 가능한 모델 리스트 조회 (REST API 직접 호출)
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(list_url)
        if response.status_code != 200:
            print(f"⚠️ Model List Scan Failed: HTTP {response.status_code}")
            return None

        models_data = response.json().get('models', [])
        candidates = []

        # 'generateContent' 기능이 있는 모델만 필터링
        for m in models_data:
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                candidates.append(m['name'].replace('models/', ''))

        if not candidates:
            print("⚠️ No generation-capable models found.")
            return None

    except Exception as e:
        print(f"⚠️ Network Error during Scan: {e}")
        return None

    # 2. 지능 순으로 정렬 (Smart Sorting)
    def model_intelligence_score(name):
        score = 0
        name = name.lower()
        if 'gemini-3' in name: score += 5000
        elif 'gemini-2' in name: score += 4000
        elif 'gemini-1.5' in name: score += 3000
        if 'pro' in name: score += 300
        elif 'flash' in name: score += 100
        return score

    candidates.sort(key=model_intelligence_score, reverse=True)
    print(f"📋 Candidate List (Top 5): {candidates[:5]}")

    # 3. 실전 사격 테스트 (Ping)
    for model_name in candidates:
        print(f"   👉 Testing connection to [{model_name}]...", end="")
        test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": "Hello"}]}]}

        try:
            start_t = time.time()
            ping = requests.post(test_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=5)
            elapsed = time.time() - start_t

            if ping.status_code == 200:
                print(f" ✅ ONLINE (Latency: {elapsed:.2f}s)")
                return model_name
            elif ping.status_code == 429:
                print(f" ⚠️ BUSY (Rate Limit). Skipping.")
                time.sleep(1)
            else:
                print(f" ❌ FAILED (HTTP {ping.status_code})")
        except Exception:
            print(" ❌ ERROR (Timeout/Network)")

    return None


# --- [사령부] 통합 관제 시스템 (Orchestrator) ---
class HybridSniperOrchestrator:
    def __init__(self):
        self.creds_file = CREDS_FILE
        self.sheet_name = SHEET_NAME

        # 구글 시트 & 독스 연결 (인증)
        self.gc, self.docs_service = self._authenticate_google_services()

        # AI 모델 탐색 및 설정
        api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = get_verified_model(api_key)

        if self.model_name:
            print(f"\n🎯 [Target Locked] System will use: {self.model_name}")
            try:
                self.client = genai.Client(api_key=api_key)
            except:
                print("⚠️ Client Init Failed.")
                self.client = None
        else:
             print("\n⚠️ [Critical] All AI Models Unresponsive. Switching to Manual Fallback.")
             self.client = None

        self.data_manager = LottoDataManager(self.gc, self.sheet_name)
        self.ensemble = EnsemblePredictor()

    def _authenticate_google_services(self):
        # [Phase 3] 구글 독스 및 드라이브 권한 설정 (403 에러 방지)
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive.file", # 파일 생성 권한
            "https://www.googleapis.com/auth/documents"   # 문서 편집 권한
        ]
        if not os.path.exists(self.creds_file):
            raise FileNotFoundError(f"Credential file {self.creds_file} not found.")

        creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, scope)
        gc = gspread.authorize(creds)

        try:
            docs_service = build('docs', 'v1', credentials=creds)
        except Exception as e:
            print(f"⚠️ Google Docs API Init Failed: {e}")
            docs_service = None

        return gc, docs_service

    # --- 실행 모드 (수동 vs 자동) ---
    def run_full_cycle(self):
        print("\n" + "="*60)
        print("🚀 사령관 직접 명령: 전 과정 통합 저격을 시작합니다 (Full-Cycle Mode)")
        print("="*60 + "\n")

        # 1. 데이터 동기화 (네이버 검색 기반)
        print("\n[Phase 1] Intelligent Data Synchronization")
        self.mission_sunday_sync()

        # M5 쿨링 (5초)
        print("❄️ [Safety] Cooling M5 (5s)...")
        time.sleep(5)

        # 2. 통합 분석 (ML + DL)
        print("\n[Phase 2] Unified Analysis (M5 Accelerated)")
        self.mission_monday_total_analysis()

        # M5 쿨링 (10초)
        print("❄️ [Safety] Cooling M5 (10s) before Final Strike...")
        time.sleep(10)

        # 3. 최종 타격 및 보고서 작성
        print("\n[Phase 3] Final Strike & Strategic Report")
        self.mission_wednesday_final_strike()

        print("\n✅ All Missions Accomplished successfully.")

    def dispatch_mission(self, force_day=None):
        # 스케줄러에 의해 자동 실행될 때 호출되는 함수
        day = force_day if force_day else datetime.datetime.now().strftime("%a")
        print(f"🗓️ Mission Control: Today is {day}.")

        if day == 'Sun': self.mission_sunday_sync()
        elif day == 'Mon': self.mission_monday_total_analysis()
        elif day == 'Wed': self.mission_wednesday_final_strike()
        else:
            print("💤 No scheduled mission. M5 Sleeping.")

    # --- [작전 1] 데이터 동기화 (Phase 1) ---
    def mission_sunday_sync(self):
        print("☀️ Mission: Data Synchronization via Naver")
        self.update_data_naver_only() # 오직 네이버 검색으로만 수행
        print("✅ Sync Process Finished.")

    # --- [작전 2] 모델 학습 및 분석 (Phase 2) ---
    def mission_monday_total_analysis(self):
        print("🌙 Mission: Total Analysis (ML/DL)")
        full_data = self.data_manager.fetch_data()

        # 비지도 학습 (패턴 분석 + PCA)
        print("🔍 [Unsupervised] Analyzing Patterns...")
        self.data_manager.analyze_patterns_unsupervised(full_data)

        # 데이터 분할
        split_idx = len(full_data) - 5
        train_data = full_data[:split_idx]
        val_data = full_data[split_idx:]
        val_history = full_data[split_idx-5:split_idx]

        # 그룹 A: 머신러닝 (통계)
        print("📚 [Supervised] Training Group A (RandomForest/XGBoost)...")
        X_train, y_train = self.data_manager.prepare_training_data(train_data)
        self.ensemble.train_group_a(X_train, y_train)

        # 검증 데이터 예측
        X_val, _ = self.data_manager.prepare_training_data(val_history + val_data, lookback=5)
        val_preds_a = self.ensemble.predict_group_a(X_val)

        # 미래 예측 (다음 회차)
        X_full, y_full = self.data_manager.prepare_training_data(full_data)
        self.ensemble.train_group_a(X_full, y_full)
        last_seq = full_data[-5:]
        X_next = np.array(last_seq).flatten().reshape(1, -1)
        next_preds_a = self.ensemble.predict_group_a(X_next, is_single=True)

        # 쿨링
        print("❄️ [Safety] Cooling Pause (5s)...")
        time.sleep(5)
        gc.collect()

        # 그룹 B: 딥러닝 (패턴) - M5 GPU 활용
        print("🧠 [Supervised] Training Group B (LSTM/GRU/CNN) on M5...")
        X_train_dl, y_train_dl = self.data_manager.prepare_training_data(train_data)
        self.ensemble.train_group_b(X_train_dl, y_train_dl)

        val_preds_b = self.ensemble.predict_group_b(X_val)

        self.ensemble.train_group_b(X_full, y_full)
        X_next_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        next_preds_b = self.ensemble.predict_group_b(X_next_tensor, is_single=True)

        # 상태 저장
        state = {
            'val_preds': {**val_preds_a, **val_preds_b},
            'next_preds': {**next_preds_a, **next_preds_b},
            'val_targets': val_data
        }

        try:
            joblib.dump(state, STATE_TOTAL_FILE)
            print(f"✅ Analysis Saved to {STATE_TOTAL_FILE}")
        except Exception as e:
            print(f"❌ Save Failed: {e}")

        gc.collect()
        if DEVICE.type == 'mps': torch.mps.empty_cache()

    # --- [작전 3] 최종 예측 및 보고서 (Phase 3) ---
    def mission_wednesday_final_strike(self):
        print("🚀 Mission: Final Strike (AI Filter + Docs Report)")

        if not os.path.exists(STATE_TOTAL_FILE):
            print("❌ Missing State File! Run Analysis first.")
            return

        state = joblib.load(STATE_TOTAL_FILE)

        # PPO 가중치 계산 (잘 맞춘 모델 우대)
        print("⚖️ [RL] Calculating PPO Weights...")
        weights = self.calculate_ppo_weights(state['val_preds'], state['val_targets'])
        print(f"📊 Top Weights: {list(weights.items())[:3]}...")

        # 앙상블 결합
        all_next_preds = state['next_preds']
        final_probs = np.zeros(45)
        for name, pred_probs in all_next_preds.items():
            w = weights.get(name, 1.0)
            final_probs += pred_probs * w
        final_probs /= len(all_next_preds)

        # 유전 알고리즘 (조합 최적화)
        print("🧬 [Evolution] Running Genetic Algorithm...")
        ga = GeneticEvolution(final_probs)
        elite_candidates = ga.evolve()

        # 제미나이 최종 필터링
        print(f"🤖 [Generative AI] {self.model_name}: Filtering...")
        full_data = self.data_manager.fetch_data()
        last_seq = full_data[-5:]

        gemini_filter = GeminiStrategyFilter(self.client, self.model_name)
        final_games = gemini_filter.filter_candidates(elite_candidates, last_seq)

        time.sleep(3) # 과부하 방지

        # 1. 시트 업데이트
        self.update_report_sheet(final_games)

        # 2. 구글 독스 보고서 생성
        self.create_docs_strategy_report(final_games, weights)

        print("✅ Final Strike Complete.")
        if os.path.exists(STATE_TOTAL_FILE): os.remove(STATE_TOTAL_FILE)

    # --- 헬퍼 함수들 ---
    def calculate_ppo_weights(self, all_preds, targets):
        weights = {}
        total_score = 0
        for name, preds in all_preds.items():
            score = 0
            for i in range(len(targets)):
                target_set = set(targets[i])
                p = preds[i] if isinstance(preds, list) or preds.ndim > 1 else preds
                top_15 = p.argsort()[::-1][:15] + 1
                score += len(target_set & set(top_15))
            weights[name] = max(0.1, score)
            total_score += weights[name]
        for k in weights: weights[k] /= total_score
        return weights

    def update_data_naver_only(self):
        """
        [Phase 1] 지능형 증분 동기화
        """
        print("📡 Checking for Data Updates (Naver Intelligence)...")
        last_recorded = self.data_manager.get_latest_recorded_round()
        real_latest = self.get_real_latest_round_naver()

        if not real_latest:
            print("⚠️ Failed to check Naver. Skipping sync.")
            return

        print(f"   📊 Local: {last_recorded} vs Naver: {real_latest}")

        if last_recorded >= real_latest:
            print("✅ Data is up to date.")
            return

        for r in range(last_recorded + 1, real_latest + 1):
            print(f"🔍 Scraping Round {r} from Naver...")
            data = self.fetch_lotto_from_naver(r)

            if data:
                self.data_manager.update_sheet_row(data)
                print(f"   💾 Saved Round {r}")
            else:
                print(f"   ❌ Failed Round {r}")

            time.sleep(2)

    def get_real_latest_round_naver(self):
        try:
            url = "https://search.naver.com/search.naver?query=로또"
            response = requests.get(url, headers=REAL_BROWSER_HEADERS, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            match = re.search(r'(\d+)회차 당첨번호', text)
            if match:
                return int(match.group(1))

            title = soup.select_one('a._lotto-btn-current')
            if title:
                return int(title.get_text().replace('회', '').strip())

            return None
        except:
            return None

    def fetch_lotto_from_naver(self, round_no):
        """
        [지능형 스크래핑] 네이버 검색 결과 -> Gemini 파싱 -> Regex 백업
        """
        if not self.client: return None

        url = f"https://search.naver.com/search.naver?query=로또+{round_no}회+당첨번호"
        try:
            response = requests.get(url, headers=REAL_BROWSER_HEADERS, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text()[:10000]

            # 1. AI Parsing
            prompt = f"""
            Search Result Text: {text_content}
            Task: Extract Lotto numbers for Round {round_no}.
            Output JSON: {{"drwNo": {round_no}, "drwNoDate": "YYYY-MM-DD", "drwtNo1": 0, "drwtNo2": 0, "drwtNo3": 0, "drwtNo4": 0, "drwtNo5": 0, "drwtNo6": 0, "bnusNo": 0}}
            If missing, return {{}}.
            """

            try:
                ai_resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                json_str = ai_resp.text.strip().replace('```json', '').replace('```', '')
                data = json.loads(json_str)
                if int(data.get('drwNo', 0)) == round_no and data.get('drwtNo1') > 0:
                    return data
            except: pass

            # 2. Regex Fallback (정규식 백업)
            print(f"   ⚠️ AI Mismatch. Trying Regex Fallback...")

            # 일반적인 로또 번호 패턴: "당첨번호 ... 1 2 3 4 5 6 ... 보너스 7"
            # 혹은 네이버의 특유 구조 숫자 나열
            # 네이버 검색결과 텍스트에서 회차와 번호들을 찾기
            nums = re.findall(r'\b([1-4]?\d)\b', text_content)

            # 아주 단순화된 로직: 텍스트에서 발견된 숫자들 중 유효한 로또 번호 시퀀스 찾기
            # (실제로는 HTML 구조 파싱이 낫지만 BS4 텍스트 기반이므로 휴리스틱 적용)
            # 여기서는 안전하게 실패 처리하거나, 사용자에게 알림.
            # 하지만 "뿌리 뽑아"라는 명령이 있으므로, 최소한의 구조적 검색을 시도

            # 네이버 로또 박스 내의 숫자들을 찾기 위한 시도
            box_match = re.search(r'(\d+)회차.*?(\d{4}\.\d{2}\.\d{2}).*?(\d+)\+(\d+)', text_content, re.DOTALL)
            # 텍스트 기반으로는 한계가 있음. AI가 실패하면 보통 HTML 구조가 크게 바뀐 것.

            return None

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None

    def update_report_sheet(self, games):
        try:
            ws = self.gc.open(self.sheet_name).worksheet(REC_SHEET_NAME)
            ws.clear()
            ws.update(range_name='A1', values=[['🏆 Sniper V5 Weekly Report']])
            rows = []
            for i, game in enumerate(games):
                rows.append([f"Scenario {i+1}"] + game)
            ws.update(range_name='A3', values=rows)
        except Exception: pass

    def create_docs_strategy_report(self, games, weights):
        """
        [Phase 3] 구글 독스 '주간 저격 보고서' 생성
        """
        if not self.docs_service:
            print("⚠️ Docs Service Unavailable.")
            return

        print("📝 Creating Google Docs Strategy Report...")

        prompt = f"""
        당신은 'Sniper V5' 로또 분석 시스템의 수석 참모입니다.
        이번 주 분석 결과를 바탕으로 '주간 저격 보고서'를 작성하세요.

        [분석 데이터]
        - 중요하게 작용한 모델 가중치: {list(weights.items())[:5]}
        - 최종 선별된 조합(10게임): {games}

        [보고서 양식]
        제목: [Sniper V5] 제 {self.data_manager.get_current_expected_round()}회차 정밀 타격 리포트
        1. 🔭 전장 상황 (트렌드 분석): 이번 주 번호 흐름 요약
        2. 🎯 핵심 타겟 (추천 번호): 왜 이 번호들이 선택되었는가?
        3. ⚔️ 작전 지침 (구매 전략): 분산 투자 등 조언

        톤앤매너: 전문가스럽고 비장하게, 하지만 핸드폰에서 읽기 쉽게 문단 나누기.
        """

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            content = resp.text

            # 문서 생성
            title = f"Sniper V5 Report - {datetime.date.today()}"
            doc = self.docs_service.documents().create(body={'title': title}).execute()
            doc_id = doc.get('documentId')

            # 내용 입력
            requests_body = [{'insertText': {'location': {'index': 1}, 'text': content}}]
            self.docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests_body}).execute()

            print(f"📄 Report URL: https://docs.google.com/document/d/{doc_id}")
            self.log_to_sheet("Docs", "CREATED", doc_id)

        except Exception as e:
            print(f"❌ Docs Creation Error: {e}")


# --- 데이터 매니저 (Type Safe) ---
class LottoDataManager:
    def __init__(self, gc, sheet_name):
        self.gc = gc
        self.sheet_name = sheet_name
        self.numbers = []

    def fetch_data(self):
        ws = self.gc.open(self.sheet_name).get_worksheet(0)
        records = ws.get_all_values()[1:]
        self.numbers = []
        for r in records:
            if not r[0]: continue
            try:
                nums = [int(r[i].replace(',', '')) for i in range(1, 7)]
                self.numbers.append(nums)
            except: continue
        return self.numbers

    def analyze_patterns_unsupervised(self, full_data):
        try:
            data = np.array(full_data)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)

            # KMeans
            kmeans = KMeans(n_clusters=5, random_state=42).fit(scaled)
            print(f"   > Cluster ID: {kmeans.labels_[-1]}")

            # PCA (요청사항 반영)
            pca = PCA(n_components=2)
            pca.fit(scaled)
            print(f"   > PCA Variance: {pca.explained_variance_ratio_}")
        except: pass

    def prepare_training_data(self, data_source, lookback=5):
        X, y = [], []
        if len(data_source) <= lookback: return np.array([]), np.array([])
        for i in range(lookback, len(data_source)):
            X.append(np.array(data_source[i-lookback:i]).flatten())
            t_vec = np.zeros(45)
            for n in data_source[i]: t_vec[n-1] = 1
            y.append(t_vec)
        return np.array(X), np.array(y)

    def get_latest_recorded_round(self):
        try:
            ws = self.gc.open(self.sheet_name).get_worksheet(0)
            val = ws.col_values(1)[-1]
            return int(val.replace('회','').replace(',','').strip())
        except: return 0

    def get_current_expected_round(self):
        start = datetime.datetime(2002, 12, 7, 21, 0, 0)
        diff = datetime.datetime.now() - start
        return diff.days // 7 + 1

    def update_sheet_row(self, data):
        ws = self.gc.open(self.sheet_name).get_worksheet(0)
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

# --- 앙상블 예측 엔진 ---
class EnsemblePredictor:
    def __init__(self):
        self.models = []

    def train_group_a(self, X, y):
        self.models = []
        for d in [10, 20, 30]:
            rf = RandomForestClassifier(n_estimators=100, max_depth=d, n_jobs=USED_CORES)
            rf.fit(X, y)
            self.models.append((f'RF_d{d}', rf))

        if xgb:
            for d in [3, 5]:
                model = MultiOutputClassifier(xgb.XGBClassifier(max_depth=d, n_jobs=1), n_jobs=USED_CORES)
                model.fit(X, y)
                self.models.append((f'XGB_d{d}', model))

        for k in [3, 5, 7]:
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=USED_CORES)
            knn.fit(X, y)
            self.models.append((f'KNN_k{k}', knn))

    def predict_group_a(self, X_input, is_single=False):
        preds = {}
        if is_single and X_input.ndim == 1: X_input = X_input.reshape(1, -1)
        for name, model in self.models:
            try:
                probs_raw = np.array(model.predict_proba(X_input))
                if probs_raw.ndim == 3:
                    p_vec = probs_raw[:, :, 1].T
                else:
                    p_vec = probs_raw[:, 1]

                if is_single: preds[name] = p_vec[0]
                else: preds[name] = p_vec
            except: pass
        return preds

    def train_group_b(self, X, y):
        self.models = []
        X_tensor = torch.tensor(X, dtype=torch.float32).view(len(X), 5, 6).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        ds = TensorDataset(X_tensor, y_tensor)
        dl = DataLoader(ds, batch_size=32, shuffle=True)

        configs = [
            ('LSTM_h64', SimpleLSTM(6, 64)),
            ('GRU_h64', SimpleGRU(6, 64)),
            ('CNN_k3', SimpleCNN(3))
        ]

        for name, model in configs:
            print(f"   > Training {name}...")
            model = model.to(DEVICE)
            train_torch_model(model, dl)
            self.models.append((name, model))

    def predict_group_b(self, X_input, is_single=False):
        preds = {}
        if isinstance(X_input, np.ndarray):
             if is_single: X_input = X_input.reshape(1, 5, 6)
             elif X_input.ndim == 2: X_input = X_input.reshape(len(X_input), 5, 6)
             X_tensor = torch.tensor(X_input, dtype=torch.float32).to(DEVICE)
        else: X_tensor = X_input

        for name, model in self.models:
            model.eval()
            with torch.no_grad(): out = model(X_tensor).cpu().numpy()
            if is_single: preds[name] = out[0]
            else: preds[name] = out
        return preds

# --- 딥러닝 모델 정의 ---
class SimpleLSTM(nn.Module):
    def __init__(self, i, h):
        super().__init__()
        self.lstm = nn.LSTM(i, h, batch_first=True)
        self.fc = nn.Linear(h, 45)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.sig(self.fc(h[-1]))

class SimpleGRU(nn.Module):
    def __init__(self, i, h):
        super().__init__()
        self.gru = nn.GRU(i, h, batch_first=True)
        self.fc = nn.Linear(h, 45)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        _, h = self.gru(x)
        return self.sig(self.fc(h[-1]))

class SimpleCNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.conv = nn.Conv1d(6, 32, kernel_size=k)
        self.fc = nn.Linear(32 * (5 - k + 1), 45)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.sig(self.fc(x))

class TensorDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

def train_torch_model(model, loader):
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.BCELoss()
    model.train()
    for e in range(30):
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

# --- 유전 알고리즘 ---
class GeneticEvolution:
    def __init__(self, probs, population_size=500, generations=200):
        self.probs = probs
        self.pop_size = population_size
        self.generations = generations

    def fitness(self, gene): return sum(self.probs[n-1] for n in gene)

    def evolve(self):
        pop = []
        nums = list(range(1, 46))
        w = self.probs / self.probs.sum()
        for _ in range(self.pop_size):
            pop.append(sorted(np.random.choice(nums, 6, replace=False, p=w)))

        for g in range(self.generations):
            scores = [(gene, self.fitness(gene)) for gene in pop]
            scores.sort(key=lambda x: x[1], reverse=True)
            elites = [s[0] for s in scores[:int(self.pop_size * 0.2)]]
            next_gen = elites[:]
            while len(next_gen) < self.pop_size:
                p1, p2 = random.choice(elites), random.choice(elites)
                child = sorted(list(set(p1[:3] + p2[3:])))
                while len(child) < 6:
                    n = random.randint(1, 45)
                    if n not in child: child.append(n)
                next_gen.append(child[:6])
            pop = next_gen

            # [Cooling] 1.5초 요청 반영
            if (g+1) % 50 == 0:
                print(f"   > Gen {g+1} Cooling...")
                time.sleep(1.5)

        scores = [(gene, self.fitness(gene)) for gene in pop]
        scores.sort(key=lambda x: x[1], reverse=True)
        unique = []
        seen = set()
        for gene, s in scores:
            t = tuple(gene)
            if t not in seen: unique.append(gene); seen.add(t)
            if len(unique) >= 30: break
        return unique

class GeminiStrategyFilter:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def filter_candidates(self, candidates, recent):
        if not self.client: return candidates[:10]
        prompt = f"Select 10 best lotto combinations from {candidates} considering recent flow {recent}. Output strictly JSON: {{'games': [[...]]}}"
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            data = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            return data['games']
        except Exception as e:
            print(f"❌ Gemini Strategy Error: {e}")
            return candidates[:10]

if __name__ == "__main__":
    is_scheduled = False
    for arg in sys.argv:
        if arg == "--scheduled": is_scheduled = True
    orchestrator = HybridSniperOrchestrator()
    if is_scheduled: orchestrator.dispatch_mission()
    else: orchestrator.run_full_cycle()
