# -*- coding: utf-8 -*-
import os
import sys
import datetime
import random
import re
import json
import time

from model_selector import SniperArmory
from sheets_handler import SheetsHandler
from sync_engine import SyncEngine

# 🎯 M5 신경망 연동
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from models import SniperModel, get_device
except ImportError:
    print("❌ [치명적 오류] PyTorch가 설치되어 있지 않습니다. 터미널에 'pip install torch'를 입력하십시오.")
    sys.exit(1)

STATE_FILE = 'hybrid_sniper_v5_state.pth'
SNIPER_STATE_JSON = 'sniper_state.json'
REC_SHEET_NAME = '추천번호'

class SniperState:
    def __init__(self):
        self.state_file = SNIPER_STATE_JSON
        self.state = self.load_state()
        
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f: return json.load(f)
        return {}
        
    def update_metric(self, key, value):
        self.state[key] = value
        with open(self.state_file, 'w', encoding='utf-8') as f: json.dump(self.state, f, indent=4, ensure_ascii=False)

class LottoOrchestrator:
    def __init__(self):
        self.sheets = SheetsHandler()
        self.armory = SniperArmory(self.sheets, auto_scout=False)
        self.state_manager = SniperState()
        self.tactics = self._load_tactics_from_sheet()

    def _load_tactics_from_sheet(self):
        print("📡 [보급] Remote_Control 시트에서 사령관님의 전술 세팅을 로드합니다...")
        try:
            ws = self.sheets.get_ws("Remote_Control")
            
            def get_int(cell, default):
                val = ws.acell(cell).value
                return int(re.sub(r'[^0-9]', '', str(val))) if val else default
                
            def get_float(cell, default):
                val = ws.acell(cell).value
                try: return float(val) if val else default
                except: return default

            epochs = get_int('H5', 500)
            lr = get_float('H6', 0.0001) 
            hidden_size = get_int('H7', 256) 
            dropout = get_float('H8', 0.2)
            extract_count = get_int('H9', 20)
            gemini_prompt = ws.acell('H11').value or "로또 조합 10개 생성"
            window_size = get_int('H12', 10)

            print(f"   ✅ 전술 로드 완료: Epochs({epochs}), LR({lr}), 윈도우({window_size}주)")
            
            return {
                "epochs": epochs, "lr": lr, "hidden_size": hidden_size,
                "dropout": dropout, "extract_count": extract_count, 
                "gemini_prompt": gemini_prompt, "window_size": window_size
            }
        except Exception as e:
            print(f"   ⚠️ 전술 데이터 로드 실패. 기본값 무장: {e}")
            return {"epochs": 500, "lr": 0.0001, "hidden_size": 256, "dropout": 0.2, "extract_count": 20, "gemini_prompt": "로또 10조합 생성", "window_size": 10}

    def _parse_lotto_row(self, row):
        if not row or len(row) < 7: return None 
        nums = []
        for x in row[1:7]: 
            cleaned = re.sub(r'[^0-9]', '', str(x))
            if cleaned:
                v = int(cleaned)
                if 1 <= v <= 45: nums.append(v)
        return sorted(nums) if len(nums) == 6 else None

    def _extract_features(self, draw):
        base = [n / 45.0 for n in draw]
        sum_val = sum(draw) / 255.0  
        odd_ratio = sum(1 for n in draw if n % 2 != 0) / 6.0
        cons_count = sum(1 for i in range(5) if draw[i] + 1 == draw[i+1]) / 5.0
        return base + [sum_val, odd_ratio, cons_count]

    def _execute_ai_strike(self, prompt_text, target_tier="고급"):
        pipeline = self.armory.get_model_pipeline(target_tier)
        if not pipeline: return None
        for model_id in pipeline:
            try:
                time.sleep(1)
                res = self.armory.client.models.generate_content(model=model_id, contents=prompt_text)
                if res.text: return res.text
            except: continue 
        return None

    def run_sync(self):
        engine = SyncEngine(self.armory, self.sheets)
        engine.run()

    def train_m5_engine(self):
        epochs = self.tactics['epochs']
        lr = self.tactics['lr']
        hidden_size = self.tactics['hidden_size']
        window_size = self.tactics['window_size']
        
        print(f"\n🧠 [Phase 2] M5 가속 학습 개시 (어텐션 탑재 / 피처 엔지니어링 작동 중)...")
        
        ws = self.sheets.get_ws(0)
        all_records = ws.get_all_values()[1:]
        history = []
        
        for row in all_records:
            parsed = self._parse_lotto_row(row)
            if parsed: history.append(parsed)
                
        history.reverse() 
        
        if len(history) <= window_size:
            print("   ⚠️ 학습에 필요한 충분한 데이터(흐름)가 부족합니다.")
            return

        X, Y = [], []
        for i in range(len(history) - window_size):
            window = history[i : i + window_size]
            target = history[i + window_size]
            
            X.append([self._extract_features(draw) for draw in window])
            
            y_multi = [0.01] * 45 
            for n in target:
                y_multi[n - 1] = 0.95 
            Y.append(y_multi)

        device = get_device()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        
        model = SniperModel(input_size=9, hidden_size=hidden_size, dropout=self.tactics['dropout']).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float('inf')
        patience = 100 
        patience_counter = 0
        
        model.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, Y_tensor)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                torch.save(model.state_dict(), STATE_FILE)
            else:
                patience_counter += 1
                
            if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
                print(f"   🔥 정밀 타격 훈련 [{epoch}/{epochs}] 완료... (현재 오차: {current_loss:.4f})")
                
            if patience_counter >= patience:
                print(f"   🛑 [지능적 조기 종료] {epoch}회차에서 최적의 맥락을 확보했습니다.")
                break
                
        print(f"   ✅ [학습 완료] 초정밀 오차({best_loss:.4f}) 상태로 M5 엔진 무장을 마쳤습니다.")
        self.state_manager.update_metric('last_loss', best_loss)

    def _perform_clustering_analysis(self, count):
        print("   🧬 [비지도 학습] K-Means 알고리즘으로 과거 패턴 군집화 분석 중...")
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from collections import Counter
            
            ws = self.sheets.get_ws(0)
            all_records = ws.get_all_values()[1:201] 
            history = []
            
            for row in all_records:
                parsed = self._parse_lotto_row(row)
                if parsed: history.append(parsed)

            if len(history) < 10: raise ValueError("군집화 데이터 부족")

            X = np.array(history)
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
            target_cluster = kmeans.predict(X[0].reshape(1, -1))[0]
            
            cluster_numbers = []
            for idx in np.where(kmeans.labels_ == target_cluster)[0]:
                cluster_numbers.extend(history[idx])

            counter = Counter(cluster_numbers)
            best_nums = [num for num, count in counter.most_common(count)]
            print(f"   ✅ [패턴 식별 완료] 현재 트렌드는 군집 #{target_cluster}에 속합니다.")
            return sorted(best_nums[:count])
        except Exception as e:
            print(f"   ⚠️ 비지도 학습 실패 ({e}). 기본 추출로 대체합니다.")
            return sorted(random.sample(range(1, 46), count))

    def generate_hybrid_combinations(self):
        print("\n🔮 [Phase 3] 정예 번호 하이브리드 타격 및 표 정밀 기록...")
        extract_count = self.tactics['extract_count']
        window_size = self.tactics['window_size']
        
        m5_numbers = []
        try:
            device = get_device()
            model = SniperModel(input_size=9, hidden_size=self.tactics['hidden_size'], dropout=self.tactics['dropout']).to(device)
            if os.path.exists(STATE_FILE):
                model.load_state_dict(torch.load(STATE_FILE, map_location=device))
                model.eval()
                
                ws = self.sheets.get_ws(0)
                recent_history = []
                
                for row in ws.get_all_values()[1:]:
                    parsed = self._parse_lotto_row(row)
                    if parsed:
                        recent_history.append(parsed)
                        if len(recent_history) == window_size:
                            break
                            
                recent_history.reverse()
                
                if len(recent_history) == window_size:
                    x_input = [[self._extract_features(draw) for draw in recent_history]]
                    with torch.no_grad():
                        output = model(torch.tensor(x_input, dtype=torch.float32).to(device))[0]
                    top_k = torch.topk(output, extract_count).indices.cpu().numpy()
                    m5_numbers = sorted([int(idx) + 1 for idx in top_k])
                    print(f"   🤖 [M5 딥러닝 타겟 확보] AI 선별 {extract_count}수: {m5_numbers}")
        except Exception as e:
            print(f"   ⚠️ M5 예측 엔진 오류: {e}")
            
        if not m5_numbers: 
            m5_numbers = sorted(random.sample(range(1, 46), extract_count))

        kmeans_numbers = self._perform_clustering_analysis(extract_count)
        
        # 🎯 [신규 전술] 템플릿을 통째로 주어 제미나이의 실수를 원천 차단합니다.
        force_prompt = """
        [출력 양식 - 반드시 아래 양식을 복사해서 내용만 채워 넣으세요]
        시나리오 1: 1, 2, 3, 4, 5, 6
        시나리오 2: 번호 6개
        시나리오 3: 번호 6개
        시나리오 4: 번호 6개
        시나리오 5: 번호 6개
        시나리오 6: 번호 6개
        시나리오 7: 번호 6개
        시나리오 8: 번호 6개
        시나리오 9: 번호 6개
        시나리오 10: 번호 6개
        ---
        이번 추천 조합은 M5 딥러닝의 예측 가중치와 K-Means 알고리즘의 패턴을 종합하여 (여기에 3문장 이내로 어떤 전술과 홀짝 비중을 사용했는지 명확한 사유를 적어주세요).
        """
        full_command = f"{self.tactics['gemini_prompt']}\n{force_prompt}\n\n[M5 후보]: {m5_numbers}\n[K-Means 후보]: {kmeans_numbers}"
        
        result_text = self._execute_ai_strike(full_command, target_tier="고급")
        
        if result_text:
            try:
                ws = self.sheets.get_ws(REC_SHEET_NAME)
                now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                ws.batch_clear(["A2:G100"]) 
                ws.update(range_name='A2:C2', values=[[now_str, "초정밀 하이브리드 타격", "상세 내용은 하단 표 참조"]])
                
                scenarios = re.findall(r"시나리오\s*(\d+)[:\s.-]+([\d\s,]+)", result_text)
                table_data = []
                for idx, nums_str in scenarios:
                    nums = re.findall(r"\d+", nums_str)
                    if len(nums) >= 6:
                        row = [f"시나리오 {idx}"] + nums[:6]
                        table_data.append(row)
                
                next_row = 15 
                
                if table_data:
                    last_row = 3 + len(table_data) - 1
                    ws.update(range_name=f'A3:G{last_row}', values=table_data)
                    print(f"   ✅ [표 업데이트 완료] {len(table_data)}개의 새로운 시나리오가 각 셀(Cell)에 정확히 배치되었습니다.")
                    next_row = last_row + 2
                else:
                    print("   ⚠️ 제미나이의 답변에서 '시나리오' 형식을 추출하지 못했습니다.")
                
                # 🎯 [강력한 요약 추출기] '---' 가 없어도 시나리오 줄만 지우고 남은 모든 글자를 긁어옵니다.
                parts = result_text.split('---')
                if len(parts) > 1:
                    summary_text = parts[1].strip()
                else:
                    # 구분선이 없으면, '시나리오'라는 단어가 안 들어간 줄들만 모아서 요약으로 만듭니다.
                    summary_lines = [line.strip() for line in result_text.split('\n') if '시나리오' not in line and line.strip() != '']
                    summary_text = '\n'.join(summary_lines)
                
                if not summary_text:
                    summary_text = "M5 엔진과 K-Means 비지도 학습을 결합하여 최적의 조합을 산출했습니다."
                
                ws.update_acell(f'A{next_row}', f"📊 [AI 타격 전술 요약 및 사유]\n{summary_text}")
                print(f"   ✅ [요약 업데이트 완료] 핵심 전술 텍스트가 A{next_row} 위치에 기록되었습니다.")
                
            except Exception as e:
                print(f"   ⚠️ 시트 기록 중 오류: {e}")

    def evaluate_performance(self):
        print("\n🏅 [Phase 4] 성과 평가...")
        try:
            ws = self.sheets.get_ws(0)
            row = ws.row_values(2)
            parsed = self._parse_lotto_row(row)
            if parsed: print(f"   📊 최신 당첨 번호: {set(parsed)}")
        except: pass
        print(f"\n🏁 작전 종료.")

if __name__ == "__main__":
    orchestrator = LottoOrchestrator()
    orchestrator.run_sync()
    orchestrator.train_m5_engine()
    orchestrator.generate_hybrid_combinations()
    orchestrator.evaluate_performance()