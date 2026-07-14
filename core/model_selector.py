# -*- coding: utf-8 -*-
import os
import time
import json
from datetime import datetime
from google import genai
from dotenv import load_dotenv

try:
    from core.sheets_handler import SheetsHandler
except ImportError:
    SheetsHandler = None

class SniperArmory:
    """
    📦 [보급 부대] 구글 서버를 정찰하여 무기를 파악하고,
    실전 부대에게 '무기 명단'을 로컬 JSON(m5_armory.json)으로 실시간 동기화하여 제공합니다.
    - [NEW] 지능형 백오프(Exponential Backoff) 방어망 이식 완료
    """
    def __init__(self, sheets_handler=None, auto_scout=False):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(dotenv_path=os.path.join(base_dir, '..', '.env'))
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ [오류] .env 파일에 키가 없습니다.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.sheets = sheets_handler
        self.armory_file = "m5_armory.json"
        self.weapons = self._load_local_armory()
        
        if auto_scout:
            self._execute_full_scout_and_verify()
            if self.sheets:
                self._update_simple_dashboard()

    def _load_local_armory(self):
        """로컬에 저장된 무기 명단을 0.001초 만에 즉시 로드합니다."""
        if os.path.exists(self.armory_file):
            try:
                with open(self.armory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("weapons", {"고급": [], "중급": [], "하급": []})
            except Exception as e:
                print(f"   ⚠️ 무기고 로컬 스캔 오류: {e}")
        return {"고급": [], "중급": [], "하급": []}

    def _execute_full_scout_and_verify(self):
        print("\n📡 [1단계] 구글 무기 창고 전수 조사 및 최신순 정렬 중...")
        try:
            all_models = list(self.client.models.list())
            scanned = {"고급": [], "중급": [], "하급": []}
            for m in all_models:
                m_id = m.name
                m_id_lower = m_id.lower()
                
                # 불량 무기 입구 컷: 프리뷰, 구형, 비전 전용 제외
                if any(x in m_id_lower for x in ["preview", "exp", "vision", "001", "002", "tuning"]):
                    continue 

                if "pro" in m_id_lower: scanned["고급"].append(m_id)
                elif "flash" in m_id_lower and "lite" not in m_id_lower and "8b" not in m_id_lower: scanned["중급"].append(m_id)
                elif any(x in m_id_lower for x in ["lite", "8b", "nano"]): scanned["하급"].append(m_id)

            print("📊 [2단계] 등급별 최대 3개 확보 작전 (지능형 백오프 적용)...")
            self.weapons = {"고급": [], "중급": [], "하급": []}
            
            for tier in ["고급", "중급", "하급"]:
                candidates = sorted(scanned[tier], reverse=True)
                success_count = 0 
                consecutive_rate_limits = 0 # [NEW] 연속 트래픽 초과 카운터
                
                for model_id in candidates:
                    # [NEW] 429 연속 발생 시 지수 백오프 대기
                    if consecutive_rate_limits > 0:
                        backoff_sec = min(2 ** consecutive_rate_limits, 30)
                        print(f"   - ⏳ Rate limit 백오프 {backoff_sec}초 대기 중...")
                        time.sleep(backoff_sec)
                    else:
                        time.sleep(4) 
                        
                    try:
                        # 무의미한 "1" 대신 정상적인 단어 "Test"로 찔러 스팸 차단 우회
                        if self.client.models.generate_content(model=model_id, contents="Test").text:
                            self.weapons[tier].append(model_id)
                            success_count += 1
                            consecutive_rate_limits = 0 # 성공 시 카운터 초기화
                            print(f"   ✅ [검수 완료 - {tier}]: {model_id}")
                            
                            if success_count >= 3: 
                                break 
                    except Exception as e:
                        err_msg = str(e).lower()
                        # [NEW] 429 또는 quota 에러 발생 시 카운터 증가
                        if "429" in err_msg or "quota" in err_msg:
                            consecutive_rate_limits += 1
                            print(f"   ⚠️ [잔탄 없음 - 429 트래픽 초과]: {model_id} (연속 {consecutive_rate_limits}회)")
                        else:
                            clean_err = str(e).replace('\n', ' ')[:70]
                            print(f"   ❌ [검수 탈락 - {model_id}]: {clean_err}...")
            
            data = {"update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "weapons": self.weapons}
            with open(self.armory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print("📈 [보급] 로컬 무기고 JSON 동기화 완료.")

        except Exception as e: 
            print(f"❌ 전수 조사 오류: {e}")

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

    def get_model_pipeline(self, target_tier="고급"):
        if target_tier == "고급": tier_order = ["고급", "중급", "하급"]
        elif target_tier == "중급": tier_order = ["중급", "하급", "고급"]
        else: tier_order = ["하급", "중급", "고급"]

        pipeline = []
        for t in tier_order:
            if self.weapons.get(t): pipeline.extend(self.weapons[t])
            
        if not pipeline:
            fallback = os.getenv("DEFAULT_AI_MODEL", "gemini-2.5-flash")
            pipeline.append(fallback)
            
        return pipeline

if __name__ == "__main__":
    print("🚀 [배치 작업] GitHub Actions 전용 스카우트 모드 가동!")
    handler = SheetsHandler() if SheetsHandler else None
    SniperArmory(handler, auto_scout=True)