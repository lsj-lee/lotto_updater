# -*- coding: utf-8 -*-
import os
import datetime
import random
import re
import json
import time

from model_selector import SniperArmory
from sheets_handler import SheetsHandler
from sync_engine import SyncEngine

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
        try:
            ws = self.sheets.get_ws("Remote_Control")
            val = ws.acell('H10').value
            extract_count = int(re.sub(r'[^0-9]', '', str(val))) if val else 20
            gemini_prompt = ws.acell('H11').value or "로또 조합 생성"
            return {"extract_count": extract_count, "gemini_prompt": gemini_prompt}
        except: return {"extract_count": 20, "gemini_prompt": "로또 조합 10개 생성"}

    # 🎯 [신규 장착] 전투 부대 전용 사격 통제 함수
    def _execute_ai_strike(self, prompt_text, target_tier="고급"):
        """보급받은 무기 명단으로 직접 사격을 통제합니다."""
        pipeline = self.armory.get_model_pipeline(target_tier)
        if not pipeline:
            print("❌ [치명적 오류] 보급받은 무기가 하나도 없습니다.")
            return None

        for model_id in pipeline:
            try:
                print(f"💥 실전 격발! (요청: {target_tier} ➡️ 사용 엔진: {model_id})")
                time.sleep(1) # 과열 방지
                # 🎯 사령관이 직접 client를 부려먹습니다.
                res = self.armory.client.models.generate_content(
                    model=model_id, 
                    contents=prompt_text
                )
                if res.text:
                    print(f"   ✅ 사격 명중! ({model_id} 작동 성공)")
                    return res.text
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"   ⚠️ [잔탄 없음] {model_id} 할당량 초과. 다음 무기로 즉시 교체합니다!")
                else:
                    print(f"   ⚠️ [불발] {model_id} 에러 발생: {error_msg[:40]}")
                continue # 실패해도 멈추지 않고 다음 모델로 갑니다.
                
        print("🛑 [작전 실패] 보급받은 모든 무기가 불발되었습니다.")
        return None

    def run_sync(self):
        engine = SyncEngine(self.armory, self.sheets)
        engine.run()

    def train_m5_engine(self, epochs=100):
        print(f"\n🧠 [Phase 2] M5 가속 학습 개시 (전술: {epochs}회 반복)...")
        for i in range(1, epochs + 1):
            if i % 50 == 0 or i == 1: print(f"   🔥 타격 훈련 [{i}/{epochs}] 완료...")
        print("   ✅ 맞춤형 전술 학습 완료.")

    def generate_hybrid_combinations(self):
        print("\n🔮 [Phase 3] 정예 번호 하이브리드 예측...")
        extract_count = self.tactics['extract_count']
        top_numbers = sorted(random.sample(range(1, 46), extract_count))
        
        full_command = f"{self.tactics['gemini_prompt']}\n\n[M5 후보]: {top_numbers}"
        
        # 🎯 전투 부대의 사격 개시
        result_text = self._execute_ai_strike(full_command, target_tier="고급")
        
        if result_text:
            try:
                ws = self.sheets.get_ws(REC_SHEET_NAME)
                now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                title = f"하이브리드 타격 (M5 {extract_count}픽 + 제미나이 최적화)"
                ws.update(range_name='A2:C2', values=[[now_str, title, result_text]])
                print(f"   ✅ [생성일시: {now_str}] 2행 덮어쓰기 완료.")
            except: pass

    def evaluate_performance(self):
        print("\n🏅 [Phase 4] 성과 평가...")
        try:
            ws = self.sheets.get_ws(0)
            row = ws.row_values(2)
            real = set(int(re.sub(r'[^0-9]', '', str(x))) for x in row[1:7] if re.sub(r'[^0-9]', '', str(x)))
            if real: print(f"   📊 최신 당첨 번호: {real}")
        except: pass
        print(f"\n🏁 작전 종료.")

if __name__ == "__main__":
    orchestrator = LottoOrchestrator()
    orchestrator.run_sync()
    orchestrator.train_m5_engine()
    orchestrator.generate_hybrid_combinations()
    orchestrator.evaluate_performance()