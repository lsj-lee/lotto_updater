# -*- coding: utf-8 -*-
import json
import os
import time
from PIL import Image
from google import genai

class VisionAnalyzer:
    """
    👁️ [암호 해독조] 무기고에서 렌즈(AI 모델)를 꺼내와 캡처된 표의 번호를 해독합니다.
    - [NEW] 서버 과부하(503) 및 트래픽 초과(429) 대비 지능형 재시도(Backoff) 방어망 탑재
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
            
            max_retries = 3
            raw_text = ""
            success = False
            
            for model_name in pipeline:
                print(f"   🎯 로컬 무기고 연동 완료! 시각 지능 모델 장착 시도: {model_name}")
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[img, prompt]
                        )
                        raw_text = response.text
                        success = True
                        break # 성공 시 재시도 루프 즉시 탈출
                    except Exception as e:
                        error_msg = str(e)
                        if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                            if attempt < max_retries - 1:
                                wait_time = 5 * (attempt + 1)
                                print(f"      ⚠️ 시각 지능 서버 과부하. {wait_time}초 대기 후 재시도... (시도 {attempt+1}/{max_retries})")
                                time.sleep(wait_time)
                            else:
                                print(f"      🚨 {max_retries}회 재시도 실패. 모델({model_name}) 통신 불가.")
                        else:
                            print(f"      🚨 알 수 없는 시각 지능 API 오류 ({model_name}): {error_msg}")
                            break # 서버 지연이 아닌 치명적 에러면 즉시 다음 무기로 교체
                
                if success:
                    break # 현재 무기로 성공했으면 다음 차순위 모델은 시도하지 않음
                    
            if not success or not raw_text:
                return {}, "가용한 모든 AI 모델과의 시각 지능 통신에 실패했습니다."
                
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
            return {}, f"이미지 처리 및 시스템 장애: {str(e)}"