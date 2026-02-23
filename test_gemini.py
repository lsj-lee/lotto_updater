import os
import sys
import time
from dotenv import load_dotenv

# [필수 라이브러리]
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ 'google-genai' 라이브러리가 필요합니다. pip install google-genai를 실행하세요.")
    sys.exit(1)

def main():
    print("\n" + "="*60)
    print("🚀 [Sniper V5] Gemini 모델 탐색 및 진단 도구 (Enhanced)")
    print("="*60 + "\n")

    # 1. 환경 변수 로드
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ .env 파일에 GEMINI_API_KEY가 없습니다.")
        sys.exit(1)

    masked_key = f"{api_key[:5]}...{api_key[-5:]}"
    print(f"🔑 API Key 확인됨: {masked_key}")

    # 2. 클라이언트 초기화
    try:
        client = genai.Client(api_key=api_key)
        print("✅ Gemini Client 초기화 성공.\n")
    except Exception as e:
        print(f"❌ Client 초기화 실패: {e}")
        sys.exit(1)

    # 3. 전체 모델 스캔 (Model Discovery)
    print("🔍 [Step 1] 전체 모델 목록 스캔 중...")
    print("-" * 80)
    print(f"{'모델 ID (Model Name)':<40} | {'기능 (Methods)':<30}")
    print("-" * 80)

    available_models = []

    try:
        # Paging을 통해 모든 모델 가져오기
        # page_size=1000으로 설정하여 한 번에 최대한 많이 가져옴
        for model in client.models.list(config={'page_size': 1000}):
            methods = getattr(model, 'supported_generation_methods', [])

            # 생성 기능(generateContent)이 있는 모델만 필터링
            if 'generateContent' in methods:
                # 모델 이름 정제 (models/ 접두사 제거)
                clean_name = model.name.replace('models/', '')
                print(f"{clean_name:<40} | {'generateContent'}")
                available_models.append(clean_name)
            else:
                # 생성 기능이 없는 모델은 로그에만 남김 (Embeddings 등)
                pass

    except Exception as e:
        print(f"⚠️ 모델 목록 조회 실패: {e}")
        print("   -> API 키 권한 문제이거나, 'List Models' API가 비활성화된 상태일 수 있습니다.")
        print("   -> 하지만 'Generate Content'는 작동할 수 있으므로 강제 테스트를 진행합니다.")

    print("-" * 80)

    # 4. 최적 모델 자동 선택 (Auto Selection)
    target_model = None

    if available_models:
        print(f"\n✅ 총 {len(available_models)}개의 사용 가능 모델을 발견했습니다.")
        # 우선순위: gemini-1.5-pro > gemini-1.5-flash > gemini-1.0-pro
        priority_order = [
            'gemini-1.5-pro',
            'gemini-1.5-flash',
            'gemini-1.0-pro',
            'gemini-pro'
        ]

        for p_model in priority_order:
            # 정확히 일치하거나 최신 버전(001, 002 등) 포함하는지 확인
            matched = [m for m in available_models if p_model in m]
            if matched:
                # 가장 최신 버전(이름이 긴 것 or 사전순 뒤쪽) 선택
                target_model = sorted(matched)[-1]
                print(f"🎯 [Auto Select] 최적 모델 선택됨: {target_model}")
                break

        if not target_model:
            target_model = available_models[0]
            print(f"⚠️ 우선순위 모델이 없어 첫 번째 모델을 선택합니다: {target_model}")
    else:
        print("\n⚠️ 목록에서 사용 가능한 모델을 찾지 못했습니다.")
        print("🚀 [Force Fire] 기본 모델(gemini-1.5-flash)로 강제 테스트를 시도합니다.")
        target_model = 'gemini-1.5-flash'

    # 5. 발사 테스트 (Firing Test)
    print(f"\n💥 [Step 2] Firing Test 시작: {target_model}")

    try:
        response = client.models.generate_content(
            model=target_model,
            contents="Hello! Are you operational? Please respond with 'System Online'."
        )

        print("\n📝 [Response]")
        print(f"> {response.text.strip()}")

        print("\n" + "="*60)
        print(f"✅ 테스트 성공! [{target_model}] 정상 작동 중.")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ 테스트 실패 ({target_model}):")
        print(f"   에러 메시지: {e}")

        print("\n💡 [Troubleshooting 가이드]")
        error_msg = str(e)
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            print("   1. 모델명 오류: 해당 모델이 존재하지 않거나 접근 권한이 없습니다.")
            print("   2. API 키 권한: 현재 키로는 이 모델을 사용할 수 없습니다.")
        elif "400" in error_msg or "INVALID_ARGUMENT" in error_msg:
            print("   1. API 키가 유효하지 않습니다. (.env 파일 확인)")
            print("   2. 결제 계정(Billing)이 연결되지 않았을 수 있습니다.")
        elif "429" in error_msg:
            print("   1. 할당량 초과(Quota Exceeded). 잠시 후 다시 시도하세요.")
        else:
            print("   -> 알 수 없는 오류입니다. 구글 클라우드 콘솔을 확인하세요.")

if __name__ == "__main__":
    main()
