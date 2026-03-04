# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
from google import genai
from dotenv import load_dotenv

try:
    from sheets_handler import SheetsHandler
except ImportError:
    SheetsHandler = None

class SniperArmory:
    """
    📦 [보급 부대] 구글 서버를 정찰하여 무기를 파악하고,
    실전 부대에게 등급별 우선순위가 적용된 '무기 명단'만 제공합니다.
    """
    def __init__(self, sheets_handler=None, auto_scout=False):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(dotenv_path=os.path.join(base_dir, '.env'))
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ [오류] .env 파일에 키가 없습니다.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.sheets = sheets_handler
        self.weapons = {"고급": [], "중급": [], "하급": []}
        
        if auto_scout:
            self._execute_full_scout_and_verify()
            if self.sheets:
                self._update_simple_dashboard()
        else:
            self._load_weapons_from_dashboard()

    def _load_weapons_from_dashboard(self):
        print("📡 [보급] 대시보드 시트에서 최신 무기 리스트를 읽어옵니다...")
        if not self.sheets: return
        try:
            ws = self.sheets.get_ws("Model_Dashboard")
            row = ws.row_values(2)
            
            def get_models(start, end):
                return [row[i] for i in range(start, end) if i < len(row) and row[i] and row[i] != "없음"]
            
            self.weapons["고급"] = get_models(2, 5)
            self.weapons["중급"] = get_models(5, 8)
            self.weapons["하급"] = get_models(8, 11)
            print(f"   ✅ 보급 확인 - 고급: {len(self.weapons['고급'])}개, 중급: {len(self.weapons['중급'])}개, 하급: {len(self.weapons['하급'])}개")
        except Exception as e:
            print(f"   ⚠️ 대시보드 로드 실패: {e}")
            self.weapons["중급"] = ["gemini-2.5-flash"]

    def _execute_full_scout_and_verify(self):
        print("\n📡 [1단계] 구글 무기 창고 전수 조사 및 최신순 정렬 중...")
        try:
            all_models = list(self.client.models.list())
            scanned = {"고급": [], "중급": [], "하급": []}
            for m in all_models:
                m_id = m.name
                if "pro" in m_id.lower(): scanned["고급"].append(m_id)
                elif "flash" in m_id.lower() and "lite" not in m_id.lower() and "8b" not in m_id.lower(): scanned["중급"].append(m_id)
                elif any(x in m_id.lower() for x in ["lite", "8b", "nano"]): scanned["하급"].append(m_id)

            print("📊 [2단계] 등급별 최대 3개 확보 작전 (10초 냉각 적용)...")
            for tier in ["고급", "중급", "하급"]:
                candidates = sorted(scanned[tier], reverse=True)
                success_count = 0 
                for model_id in candidates:
                    time.sleep(10) 
                    try:
                        if self.client.models.generate_content(model=model_id, contents="1").text:
                            self.weapons[tier].append(model_id)
                            success_count += 1
                            print(f"   ✅ [검수 완료]: {model_id}")
                            if success_count >= 3: break 
                    except: pass
        except Exception as e: print(f"❌ 전수 조사 오류: {e}")

    def _update_simple_dashboard(self):
        try:
            ws = self.sheets.get_ws("Model_Dashboard")
            ws.clear()
            headers = ["날짜", "시간", "고급(1)", "고급(2)", "고급(3)", "중급(1)", "중급(2)", "중급(3)", "하급(1)", "하급(2)", "하급(3)"]
            ws.append_row(headers)
            def pad(lst): return lst[:3] + ["없음"] * (3 - len(lst[:3]))
            row_data = [datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S')] + pad(self.weapons["고급"]) + pad(self.weapons["중급"]) + pad(self.weapons["하급"])
            ws.append_row(row_data)
            print("📈 [3단계] 대시보드 기록 완료.")
        except: pass

    # 🎯 [신규 장착] 모델 명단(파이프라인) 반환 함수
    def get_model_pipeline(self, target_tier="고급"):
        if target_tier == "고급": tier_order = ["고급", "중급", "하급"]
        elif target_tier == "중급": tier_order = ["중급", "하급", "고급"]
        else: tier_order = ["하급", "중급", "고급"]

        pipeline = []
        for t in tier_order:
            if self.weapons.get(t): pipeline.extend(self.weapons[t])
        return pipeline

if __name__ == "__main__":
    print("🚀 [배치 작업] GitHub Actions 전용 스카우트 모드 가동!")
    handler = SheetsHandler() if SheetsHandler else None
    SniperArmory(handler, auto_scout=True)