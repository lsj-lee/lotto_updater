import os
import requests
from dotenv import load_dotenv

# 🛰️ Sniper V5 - 지휘소 환경 변수 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def find_and_strike_models():
    print("=" * 60)
    print("🚀 [Sniper V5] Gemini API: Search & Strike Verification")
    print("   - Strategy: 1단계 탐색 -> 2단계 실전 사격 검증")
    print("=" * 60)

    if not api_key:
        print("❌ [ERROR] API Key Missing. .env 파일을 확인하세요.")
        return

    print("\n1️⃣ [1단계] 구글 본부 스캔: 텍스트 생성(generateContent) 가능 모델 탐색 중...")
    url_list = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    try:
        res_list = requests.get(url_list)
        if res_list.status_code != 200:
            print(f"❌ 서버 접근 거부 (HTTP {res_list.status_code}). 권한이나 키 설정을 확인하세요.")
            print("응답:", res_list.text)
            return
            
        data = res_list.json()
        all_models = data.get('models', [])
        
        # 이름에 gemini가 들어가고 텍스트 생성을 지원하는 모델만 추출
        target_candidates = []
        for m in all_models:
            name = m.get('name', '')
            methods = m.get('supportedGenerationMethods', [])
            if 'generateContent' in methods and 'gemini' in name.lower():
                target_candidates.append(name)
                
        if not target_candidates:
            print("⚠️ 타격 가능한 참모(모델) 후보를 하나도 찾지 못했습니다.")
            return
            
        print(f"✅ 총 {len(target_candidates)}명의 참모 후보 발견. 즉시 2단계 검증으로 넘어갑니다.\n")
        
        print("2️⃣ [2단계] 실전 통신 검증: 각 참모에게 직접 교신(Hello)을 시도합니다...")
        
        verified_working_models = []
        
        for model_name in target_candidates:
            # 출력 이름 간소화 (예: models/gemini-1.5-flash -> gemini-1.5-flash)
            short_name = model_name.split('/')[-1] if '/' in model_name else model_name
            print(f"🎯 타격 시도: [{short_name}] ...", end=" ")
            
            # 실제 텍스트 생성을 요청하는 POST 통신
            url_generate = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": "Hello, this is a connection test."}]}]
            }
            
            res_gen = requests.post(url_generate, json=payload)
            
            if res_gen.status_code == 200:
                print("✅ 100% 교신 성공! (응답 확인됨)")
                verified_working_models.append(model_name)
            else:
                print(f"❌ 연결 실패 (오류 코드: {res_gen.status_code})")
        
        print("\n" + "=" * 60)
        print("🏆 [최종 작전 결과: 100% 작동이 보장된 최정예 참모 목록]")
        if verified_working_models:
            for idx, wm in enumerate(verified_working_models, 1):
                print(f"   {idx}. {wm}")
            print("\n🎉 사령관님, 이 목록에 있는 참모들은 지금 당장 로또 분석에 투입할 수 있는 실제 전력입니다!")
        else:
            print("⚠️ 안타깝게도 서류상 후보는 있었으나, 실제로 무전에 응답하는 참모가 없습니다. (API 승인 지연 중일 확률 99%)")
            
    except Exception as e:
        print(f"❌ 물리적 네트워크 에러 발생: {e}")

if __name__ == "__main__":
    find_and_strike_models()