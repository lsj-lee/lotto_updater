# -*- coding: utf-8 -*-
import json
import os
import time
import sys
from PIL import Image
from google import genai

class VisionAnalyzer:
    """
    👁️ [암호 해독조] 무기고에서 렌즈(AI 모델)를 꺼내와 캡처된 표의 번호를 해독합니다.
    - [NEW] 통신 실패 시 명확한 에러 사유를 출력하고 시스템을 대기 상태로 전환
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
            
            target_model = pipeline[0] if pipeline else "models/gemini-3.5-flash"
            max_retries = 3
            raw_text = ""
            success = False
            last_error_msg = "알 수 없는 통신 오류" # [NEW] 에러 메시지 포획용 변수
            
            print(f"   🔄 [최상위 모델 지정]: {target_model} (서버 부하 시 3회 재시도 후 작전 정지)")
            
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=target_model,
                        contents=[img, prompt]
                    )
                    raw_text = response.text
                    success = True
                    break 
                except Exception as e:
                    last_error_msg = str(e) # [NEW] 에러 원인 저장
                    if any(err in last_error_msg for err in ["503", "UNAVAILABLE", "429", "quota"]):
                        if attempt < max_retries - 1:
                            wait_time = 5 * (attempt + 1)
                            print(f"      ⚠️ 시각 지능 서버 과부하. {wait_time}초 대기 후 재시도... (시도 {attempt+1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            print(f"      🚨 {max_retries}회 재시도 실패. 모델({target_model}) 통신 불가.")
                    else:
                        print(f"      🚨 알 수 없는 시각 지능 API 오류 ({target_model}): {last_error_msg}")
                        break 
                
            # [NEW] 3회 재시도 실패 시 에러 사유 출력
            if not success or not raw_text:
                print("\n" + "="*65)
                print(f"🚨 [작전 중단] 최상위 모델({target_model}) 시각 지능 통신 3회 연속 실패로 인해 시스템을 정지합니다.")
                print(f"   ▶ 차단 사유 (Error): {last_error_msg}")
                print("   ▶ 조치 권고사항:")
                print("      - 일시적인 구글 API 트래픽 초과 현상일 확률이 높습니다.")
                print("      - 시스템을 수동으로 재가동하려면 잠시 후 메뉴 '5번(ALL)'을 다시 입력하십시오.")
                print("="*65)
                sys.exit(0)
                
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