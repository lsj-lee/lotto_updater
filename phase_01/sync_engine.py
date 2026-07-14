# -*- coding: utf-8 -*-
import re
import time
import datetime
import asyncio
import os
from phase_01.vision_scraper import VisionScraper
from phase_01.vision_analyzer import VisionAnalyzer

class SyncEngine:
    """
    🕹️ [전술 통제기] Scraper(캡처)와 Analyzer(해독)를 지휘하여 시트 동기화를 완수합니다.
    - WAF 원천 차단을 위해 네이버 포털 우회 검색 전술 사용
    """
    def __init__(self, armory, sheets):
        self.armory = armory
        self.sheets = sheets
        self.ws = self.sheets.get_ws(0)
        self.scraper = VisionScraper()
        self.analyzer = VisionAnalyzer(self.armory)

    def run(self):
        print("\n🔄 [Phase 1] 시각 지능(Vision AI) 기반 스텔스 동기화 개시...")
        
        sheet_latest = self._get_sheet_latest_ep()
        print(f"   📊 현재 시트에 기록된 최신 회차: {sheet_latest}회")

        real_latest = self._get_real_latest_ep()
        print(f"   🌐 현재 시간 기준 실제 최신 회차: {real_latest}회")

        if sheet_latest >= real_latest:
            print("   ✅ 이미 최신 데이터를 보유하고 있습니다. 동기화를 생략합니다.")
            return

        missing_eps = list(range(sheet_latest + 1, real_latest + 1))
        print(f"   🚀 총 {len(missing_eps)}개 회차 누락 발견. 네이버 우회 정찰기를 띄웁니다...")

        capture_path = "lotto_capture.png"
        
        for ep in missing_eps:
            extracted_data = {}
            try:
                # 1. 시각 정찰기 출동 (각 누락 회차별 검색 캡처)
                asyncio.run(self.scraper.capture(ep, capture_path))

                if not os.path.exists(capture_path):
                    print(f"   ❌ {ep}회차 스크린샷 확보 실패. 작전을 중지합니다.")
                    break

                print("   👁️ 제미나이 시각 지능으로 암호 해독 중...")
                
                # 2. 암호 해독조 가동
                extracted_data, error_msg = self.analyzer.analyze_image(capture_path, [ep])
                
                if not extracted_data:
                    print(f"   ❌ {ep}회차 시각 지능 분석 실패: {error_msg}")
                    print(f"   📸 [블랙박스 확보] 원인 파악을 위해 '{capture_path}' 사진을 보존합니다.")
                    break

                # 3. 해독 결과 시트 기록
                ep_str = str(ep)
                if ep_str in extracted_data:
                    data = extracted_data[ep_str]
                    nums = data.get('numbers', [0]*6)
                    row_data = [
                        int(ep),
                        nums[0], nums[1], nums[2], nums[3], nums[4], nums[5],
                        data.get('bonus', 0), data.get('winners', 0), data.get('prize', 0)
                    ]
                    self._insert_to_sheet(row_data)
                    print(f"   ✅ {ep}회차 시트 삽입 완료: {nums} + 보너스 {data.get('bonus')}")
                else:
                    print(f"   ⚠️ {ep}회차 데이터를 사진에서 찾지 못했습니다.")
                    
            finally:
                # 분석 성공 시에만 임시 사진 깔끔하게 소각
                if os.path.exists(capture_path) and extracted_data:
                    try:
                        os.remove(capture_path)
                        print("   🧹 임시 정찰 사진 소각 완료. (기지 청결 유지)")
                    except Exception as e:
                        print(f"   ⚠️ 임시 사진 소각 실패: {e}")
            
            # 다중 회차 누락 시 포털 사이트 과부하 방지를 위한 냉각 시간 (1초)
            time.sleep(1)

    def _get_sheet_latest_ep(self):
        try:
            col_a = self.ws.col_values(1)
            eps = [int(re.sub(r'[^0-9]', '', str(val))) for val in col_a[1:] if re.sub(r'[^0-9]', '', str(val))]
            return max(eps) if eps else 0
        except: return 0

    def _get_real_latest_ep(self):
        today = datetime.datetime.now()
        first_draw = datetime.datetime(2002, 12, 7, 20, 45) 
        return ((today - first_draw).days // 7) + 1

    def _insert_to_sheet(self, data):
        try:
            self.ws.insert_row(data, 2)
        except: pass

if __name__ == "__main__":
    print("🚀 [비전 모드] 시각 지능 동기화 엔진 단독 가동!")
    try:
        from core.sheets_handler import SheetsHandler
        from dotenv import load_dotenv
        load_dotenv()
        
        handler = SheetsHandler()
        engine = SyncEngine(armory=None, sheets=handler)
        engine.run()
        print("\n✅ 비전 작전 종료.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")