# -*- coding: utf-8 -*-
import json
import os
from PIL import Image
from google import genai

class VisionAnalyzer:
    """
    👁️ [암호 해독조] 무기고에서 렌즈(AI 모델)를 꺼내와 캡처된 표의 번호를 해독합니다.
    """
    def __init__(self, armory):
        self.armory = armory

    def analyze_image(self, image_path, missing_eps):
        try:
            img = Image.open(image_path)
            prompt = f"""
            당신은 최고의 시각 데이터 추출 AI입니다.
            첨부된 이미지는 네이버 로또 당첨번호 검색 결과입니다.
            이미지에서 {missing_eps}회차의 당첨 정보를 찾아서 정확히 추출하세요.
            
            반드시 아래 JSON 형식으로만 대답하고, 마크다운(```json 등)은 제외하세요:
            {{
                "1230": {{
                    "numbers": [3, 8, 9, 22, 28, 42],
                    "bonus": 45,
                    "winners": 16,
                    "prize": 1771357196
                }}
            }}
            
            [주의사항]
            1. '명', '원', 쉼표(,)는 절대 넣지 말고 순수 숫자형(int)으로만 추출하세요.
            2. 회차 번호를 Key(문자열)로 사용하세요.
            3. 이미지에 당첨 번호가 보이지 않으면 빈 JSON {{}} 을 반환하세요.
            4. 네이버 위젯 화면에 당첨자 수(winners)나 1등 당첨금(prize)이 보이지 않는다면 억지로 찾지 말고 0 으로 반환하세요.
            """
            
            if not self.armory:
                from core.model_selector import SniperArmory
                self.armory = SniperArmory()
                
            client = self.armory.client
            pipeline = self.armory.get_model_pipeline(target_tier="중급")
            target_model = pipeline[0] if pipeline else os.getenv("DEFAULT_AI_MODEL", "gemini-2.5-flash")
            
            print(f"   🎯 로컬 무기고 연동 완료! 시각 지능 모델 장착: {target_model}")
            
            response = client.models.generate_content(
                model=target_model,
                contents=[img, prompt]
            )
            
            raw_text = response.text
            if not raw_text:
                return {}, "AI의 응답 텍스트가 완전히 비어있습니다."
                
            res_text = raw_text.replace('```json', '').replace('```', '').strip()
            print(f"      [AI 해독 결과]: {res_text}")
            
            try:
                parsed_json = json.loads(res_text)
                if not parsed_json:
                    return {}, "AI가 당첨 번호를 찾지 못해 빈칸을 반환했습니다."
                return parsed_json, "성공"
            except json.JSONDecodeError as e:
                return {}, f"JSON 파싱 실패 사유: {e}"
                
        except Exception as e:
            return {}, f"API 통신 오류 및 시스템 장애: {str(e)}"