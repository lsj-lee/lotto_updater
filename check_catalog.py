# -*- coding: utf-8 -*-
import os
from google import genai
from dotenv import load_dotenv

def scout_google_warehouse():
    # 1. 환경 설정 로드
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(base_dir, '.env'))
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ [오류] .env 파일에 GEMINI_API_KEY가 없습니다.")
        return

    print("📡 [정찰 시작] 구글 클라우드 무기 창고에 접속 중...")
    client = genai.Client(api_key=api_key)

    try:
        # 2. 모든 모델 리스트 가져오기
        # 구글 API는 반복자(Iterator)를 반환하므로 리스트로 변환하여 확인합니다.
        models = list(client.models.list())
        
        print(f"✅ 총 {len(models)}개의 모델이 발견되었습니다.\n")
        print("=" * 60)
        print(f"{'등급/종류':<15} | {'모델 식별자 (ID)':<30}")
        print("-" * 60)

        pro_models = []
        flash_models = []
        other_models = []

        for m in models:
            m_id = m.name # 예: models/gemini-1.5-pro
            if "pro" in m_id.lower():
                pro_models.append(m_id)
            elif "flash" in m_id.lower():
                flash_models.append(m_id)
            else:
                other_models.append(m_id)

        # 3. 결과 분류 출력
        for name in sorted(pro_models):
            print(f"{'💎 고급(Pro)':<15} | {name:<30}")
        for name in sorted(flash_models):
            print(f"{'⚡ 중급(Flash)':<15} | {name:<30}")
        for name in sorted(other_models):
            print(f"{'📦 기타/임베딩':<15} | {name:<30}")

        print("=" * 60)
        
        # 4. 사령관님이 찾으시는 핵심 모델 존재 여부 체크
        target = "gemini-1.5-pro"
        found = any(target in m for m in pro_models)
        if found:
            print(f"\n💡 확인 결과: '{target}'이 목록에 존재합니다! 왜 이전 코드에서 못 찾았는지 분석이 필요합니다.")
        else:
            print(f"\n⚠️ 확인 결과: '{target}'이 목록에 없습니다. 구글이 사령관님의 지역이나 계정에서 이 모델을 숨긴 것 같습니다.")

    except Exception as e:
        print(f"❌ [정찰 실패] 구글 서버 응답 오류: {e}")

if __name__ == "__main__":
    scout_google_warehouse()