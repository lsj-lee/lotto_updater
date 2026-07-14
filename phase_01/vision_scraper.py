# -*- coding: utf-8 -*-
import asyncio
from playwright.async_api import async_playwright

class VisionScraper:
    """
    📸 [시각 정찰기] 네이버 포털 우회 정찰 및 뷰포트(Viewport) 캡처 전담
    - 봇 탐지가 없는 민간 포털 검색 결과를 활용하여 WAF 원천 회피
    """
    async def capture(self, ep, output_path):
        print(f"   📸 목표 지점 침투 중: 네이버 검색망 우회 (타겟 회차: {ep}회)...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                # 네이버 검색창으로 직행하여 결과 출력
                url = f"https://search.naver.com/search.naver?query={ep}회+로또+당첨번호"
                await page.goto(url, wait_until="networkidle", timeout=15000)
                
                # 렌더링 안정화를 위한 대기
                await page.wait_for_timeout(2000) 
                
                # 전체 화면(full_page=True)이 아닌, 화면에 보이는 검색 결과 최상단 위젯만 캡처
                await page.screenshot(path=output_path, full_page=False)
                print(f"   ✅ [포털 우회 성공] 제 {ep}회차 당첨 정보 시각 데이터 확보 완료.")
            except Exception as e:
                print(f"   ⚠️ 캡처 실패 사유: {e}")
            finally:
                await browser.close()