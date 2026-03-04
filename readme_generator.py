# -*- coding: utf-8 -*-
import os
import time
from model_selector import SniperArmory

def generate_readme():
    print("🧬 [README Factory] AI가 시스템을 분석하여 설명서를 작성합니다...")
    
    try:
        # 1. 보급 부대에서 명단 요청
        armory = SniperArmory(auto_scout=False)
        pipeline = armory.get_model_pipeline("고급")
        
        if not pipeline:
            print("❌ 보급받은 무기 명단이 없습니다. 작전을 취소합니다.")
            return
        
        # 2. 분석할 파일 리스트 확보
        files = [f for f in os.listdir('.') if f.endswith('.py')]
        file_list_str = "\n".join([f"- {f}" for f in files])
        
        prompt = f"""
        당신은 로또 분석 시스템 'Sniper V7'의 기술 문서 작성자입니다.
        현재 프로젝트에 포함된 파일 리스트를 바탕으로 멋진 README.md 내용을 작성해 주세요.
        
        [프로젝트 구성 파일]
        {file_list_str}
        
        [포함할 내용]
        1. 프로젝트 명: Sniper V7 (로또 분석 자동화 시스템)
        2. 핵심 기술: M5 가속 학습, Playwright 시각 지능(Vision AI), GitHub Actions 무인 정비.
        3. 파일별 기능 요약.
        4. 사령관(사용자)을 위한 경고: "이 시스템은 확률 기반 분석 도구입니다."
        
        마크다운 형식으로 출력하고, 다른 설명 없이 오직 파일 내용만 출력하세요.
        """
        
        # 3. 행정 부대의 직접 사격 루프
        readme_content = None
        for model_id in pipeline:
            try:
                print(f"💥 문서 작성 요청 (엔진: {model_id})")
                time.sleep(1)
                res = armory.client.models.generate_content(model=model_id, contents=prompt)
                
                if res.text:
                    readme_content = res.text
                    print(f"   ✅ 작성 완료! ({model_id} 작동 성공)")
                    break # 성공하면 즉시 루프 탈출
            except Exception as e:
                print(f"   ⚠️ {model_id} 응답 실패: {str(e)[:40]}")
                continue # 실패 시 다음 모델로 교체
        
        # 4. 파일 저장
        if readme_content:
            with open("README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            print("✅ README.md 파일이 성공적으로 갱신되었습니다.")
        else:
            print("❌ 가용한 모든 AI 엔진이 문서 작성을 실패했습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    generate_readme()