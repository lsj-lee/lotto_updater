import os
import time
from google import genai
from dotenv import load_dotenv

def find_every_available_model():
    # 1. 환경 변수 및 클라이언트 설정
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY_1")
    if not api_key:
        print("❌ .env에서 키를 찾을 수 없습니다.")
        return

    client = genai.Client(api_key=api_key)

    print("\n" + "="*70)
    print("📡 [실시간] 상진 님의 API 키로 접근 가능한 모든 모델 리스트")
    print("="*70)

    try:
        # 구글 서버에서 사용 가능한 모델 목록 전체 수신
        available_models = client.models.list()
        
        valid_models = []
        for model in available_models:
            # 텍스트 생성이 가능한 모델만 필터링
            if 'generateContent' in model.supported_actions:
                valid_models.append(model.name)
                print(f"📍 발견: {model.name:<40} | 버전: {model.version}")

        print("\n" + "="*70)
        print(f"🔎 총 {len(valid_models)}개의 생성 모델 발견. 실제 가동 테스트를 시작합니다.")
        print("="*70)

        # 2. 발견된 모델들 실제 가용성 테스트
        for m_path in valid_models:
            # 모델 경로에서 'models/' 접두사 처리
            m_id = m_path.split('/')[-1]
            print(f"🧪 {m_id:<35} ->", end=" ", flush=True)
            
            try:
                # 할당량 및 응답 테스트
                res = client.models.generate_content(model=m_id, contents="ping")
                print(f" ✅ [사용 가능] (응답: {res.text.strip()})")
            except Exception as e:
                err = str(e).lower()
                if "429" in err:
                    print(" ⚠️ [429] 할당량 초과 (오늘 한도 도달)")
                elif "403" in err:
                    print(" 🚫 [403] 권한 없음 (계정 제한)")
                else:
                    print(f" ❌ [에러] {err[:40]}...")
            
            time.sleep(0.5) # 서버 매너 대기

    except Exception as e:
        print(f"❌ 목록 호출 실패: {e}")

    print("="*70)
    print("🏁 모든 조사가 완료되었습니다.")

if __name__ == "__main__":
    find_every_available_model()