# -*- coding: utf-8 -*-
import re
import time
import datetime
import json
import asyncio
import os
from playwright.async_api import async_playwright
from PIL import Image
from google import genai

class SyncEngine:
    """
    👁️ [정보국] Playwright로 화면을 캡처하고, 시트에 기록된 중급 모델을 꺼내와 즉시 판독합니다.
    작전 완료 후 임시 스크린샷은 자동으로 소각됩니다.
    """
    def __init__(self, armory, sheets):
        self.armory = armory
        self.sheets = sheets
        self.ws = self.sheets.get_ws(0)

    def run(self):
        print("\n🔄 [Phase 1] 시각 지능(Vision AI) 기반 스텔스 동기화 개시...")
        
        sheet_latest = self._get_sheet_latest_ep()
        print(f"   📊 현재 시트에 기록된 최신 회차: {sheet_latest}회")

        real_latest = self._get_real_latest_ep()
        print(f"   🌐 오늘 날짜 기준 실제 최신 회차: {real_latest}회")

        if sheet_latest >= real_latest:
            print("   ✅ 이미 최신 데이터를 보유하고 있습니다. 동기화를 마칩니다.")
            return

        missing_eps = list(range(sheet_latest + 1, real_latest + 1))
        print(f"   🚀 총 {len(missing_eps)}개 회차 누락 발견. 스텔스 정찰기를 띄웁니다...")

        capture_path = "lotto_capture.png"
        
        # 🎯 [핵심 개조] 작전 성공 여부와 관계없이 반드시 증거를 인멸하기 위한 방어벽
        try:
            # 1. Playwright로 사이트 접속 및 스크린샷 촬영
            asyncio.run(self._capture_screenshot("https://www.lotto.co.kr/article/list/AC01", capture_path))

            if not os.path.exists(capture_path):
                print("   ❌ 스크린샷 확보 실패. 작전을 중지합니다.")
                return

            # 2. Gemini Vision AI로 스크린샷 해독
            print("   👁️ 제미나이 시각 지능으로 암호 해독 중...")
            extracted_data = self._analyze_image(capture_path, missing_eps)
            
            if not extracted_data:
                print("   ❌ 시각 지능 분석 실패. (API 응답 오류 또는 표 인식 불가)")
                return

            # 3. 해독된 데이터를 시트에 주입 (과거 회차부터 순서대로)
            for ep in missing_eps:
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
                    time.sleep(0.5)
                else:
                    print(f"   ⚠️ {ep}회차 데이터를 사진에서 찾지 못했습니다.")
                    
        finally:
            # 🧹 [증거 인멸] 캡처 파일이 존재하면 즉시 소각합니다.
            if os.path.exists(capture_path):
                try:
                    os.remove(capture_path)
                    print("   🧹 임시 정찰 사진 소각 완료. (기지 청결 유지)")
                except Exception as e:
                    print(f"   ⚠️ 임시 사진 소각 실패: {e}")

    def _get_sheet_latest_ep(self):
        try:
            col_a = self.ws.col_values(1)
            eps = [int(re.sub(r'[^0-9]', '', str(val))) for val in col_a[1:] if re.sub(r'[^0-9]', '', str(val))]
            return max(eps) if eps else 0
        except: return 0

    def _get_real_latest_ep(self):
        today = datetime.datetime.now()
        first_draw = datetime.datetime(2002, 12, 7) 
        return ((today - first_draw).days // 7) + 1

    async def _capture_screenshot(self, url, output_path):
        print("   📸 목표 지점 침투 및 고해상도 촬영 중...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1500)
                await page.screenshot(path=output_path, full_page=True)
                print("   ✅ 적 요새(웹페이지) 표면 촬영 완료.")
            except Exception as e:
                print(f"   ⚠️ 촬영 중 오류: {e}")
            finally:
                await browser.close()

    def _analyze_image(self, image_path, missing_eps):
        try:
            img = Image.open(image_path)
            prompt = f"""
            당신은 최고의 시각 데이터 추출 AI입니다.
            첨부된 이미지는 로또 당첨번호 표입니다.
            이미지에서 다음 누락된 회차 {missing_eps}의 당첨 정보를 찾아서 정확히 추출하세요.
            
            반드시 아래 JSON 형식으로만 대답하고, 마크다운(```json 등)은 제외하세요:
            {{
                "1213": {{
                    "numbers": [5, 11, 25, 27, 36, 38],
                    "bonus": 2,
                    "winners": 18,
                    "prize": 1740011646
                }}
            }}
            
            [주의사항]
            1. '명', '원', 쉼표(,)는 절대 넣지 말고 순수 숫자형(int)으로만 추출하세요.
            2. 회차 번호를 Key(문자열)로 사용하세요.
            3. 이미지에 해당 회차가 보이지 않으면 빈 JSON {{}} 을 반환하세요.
            """
            
            try:
                dashboard_ws = self.sheets.get_ws("Model_Dashboard")
                target_model = dashboard_ws.cell(2, 6).value 
                if not target_model or target_model == "없음":
                    target_model = "gemini-2.5-flash"
            except Exception:
                target_model = "gemini-2.5-flash"
                
            print(f"   🎯 대시보드 연동 완료! 사용 모델: {target_model}")
            
            if self.armory:
                client = self.armory.client
            else:
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                client = genai.Client(api_key=api_key)
                
            response = client.models.generate_content(
                model=target_model,
                contents=[img, prompt]
            )
            
            if response.text:
                res_text = response.text.replace('```json', '').replace('```', '').strip()
                return json.loads(res_text)
            return {}
        except Exception as e:
            print(f"   ⚠️ 제미나이 시각 분석 오류: {e}")
            return {}

    def _insert_to_sheet(self, data):
        try:
            self.ws.insert_row(data, 2)
        except: pass

# ==========================================
# 🎯 [실행 버튼] 무기고 스캔을 원천 차단했습니다.
# ==========================================
if __name__ == "__main__":
    print("🚀 [비전 모드] 시각 지능 동기화 엔진 단독 가동!")
    try:
        from sheets_handler import SheetsHandler
        from dotenv import load_dotenv
        load_dotenv()
        
        handler = SheetsHandler()
        
        engine = SyncEngine(armory=None, sheets=handler)
        engine.run()
        print("\n✅ 비전 작전 종료.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")