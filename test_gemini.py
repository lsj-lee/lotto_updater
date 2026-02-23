import os
import sys
import time
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    print("❌ 라이브러리 설치 필요: pip install -U google-genai python-dotenv")
    sys.exit(1)

def main():
    print("\n" + "="*85)
    print("🛰️  [Sniper V5] 무차별 전수 조사: 필터 없이 모든 모델 실전 투입 테스트")
    print("="*85 + "\n")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ .env 파일에서 API 키를 찾을 수 없습니다.")
        return

    client = genai.Client(api_key=api_key)
    
    # 1. 필터 없이 모든 모델 확보
    print("🔍 [Step 1] 전체 모델 리스트 무조건 확보 중...")
    all_models = []
    try:
        for m in client.models.list():
            clean_name = m.name.replace('models/', '')
            # 임베딩(embedding) 모델은 텍스트 생성이 안 되므로 이름으로만 제외
            if 'embedding' not in clean_name and 'aqa' not in clean_name:
                all_models.append(clean_name)
        print(f"✅ 총 {len(all_models)}개의 후보 모델을 식별했습니다.\n")
    except Exception as e:
        print(f"❌ 목록 가져오기 실패: {e}")
        return

    # 2. 무차별 사격 테스트 (Blind Fire)
    print("🚀 [Step 2] 무차별 사격 개시 (응답 여부만 확인)")
    print("-" * 85)
    print(f"{'#':<3} | {'Model ID':<40} | {'Status':<15}")
    print("-" * 85)

    working_models = []

    for i, model_id in enumerate(all_models, 1):
        try:
            # 기능(Methods) 정보 무시하고 일단 호출 시도
            response = client.models.generate_content(
                model=model_id,
                contents="hi"
            )
            print(f"{i:<3} | {model_id:<40} | ✅ ONLINE")
            working_models.append(model_id)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg:
                status = "⚠️ QUOTA FULL"
            elif "404" in err_msg:
                status = "❌ NOT FOUND"
            else:
                status = "❌ ERROR"
            print(f"{i:<3} | {model_id:<40} | {status}")
        
        # 서버 과부하 및 차단 방지용 지연 (M5 안정성 확보)
        time.sleep(0.7)

    # 3. 작전 리포트
    print("-" * 85)
    print(f"\n📊 [최종 정찰 보고서]")
    if working_models:
        print(f"🟢 즉시 가동 가능 모델: {', '.join(working_models)}")
        print(f"\n🎯 [사령관님을 위한 제언]: '{working_models[0]}' 모델을 메인 지휘관으로 추천합니다.")
    else:
        print("⚠️ 모든 모델이 할당량 초과이거나 가동 불능 상태입니다.")
        print("💡 팁: 약 1분 후 다시 시도하거나, AI Studio에서 새 API Key 발급을 고려하세요.")

    print("\n" + "="*85)

if __name__ == "__main__":
    main()