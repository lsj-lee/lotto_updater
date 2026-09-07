# -*- coding: utf-8 -*-
import os
import io
import json
from datetime import datetime
from contextlib import redirect_stdout
from dotenv import load_dotenv

from google import genai 
from phase_02.m5_ultimate import M5UltimateEngine
from core.model_selector import SniperArmory  # [NEW] 로컬 무기고 연동

class GeminiTactician:
    """
    📊 [Phase 03] AI 데이터 분석 및 리포트 생성 모듈
    - Phase 04의 오답노트(m5_memory.json) 지시사항을 반영
    - 무기고(Armory) 연동을 통해 하드코딩 없이 유연한 모델 통신망 구축
    """
    def __init__(self, sheets_handler):
        self.sheets = sheets_handler
        
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("🚨 .env 파일에 GEMINI_API_KEY가 설정되지 않았습니다!")
        
        self.client = genai.Client(api_key=api_key)
        self.armory = SniperArmory() # 무기고 장착
        print("   ✅ Google GenAI API 및 로컬 무기고(Armory) 연동 완료.")

    def _load_memory_directives(self):
        memory_file = "m5_memory.json"
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("tactical_directives", {})
            except Exception as e:
                print(f"   ⚠️ 로컬 메모리 스캔 중 오류 발생: {e}")
                return {}
        return {}

    def call_m5_and_get_results(self):
        print("   ⚙️ M5 예측 모델 구동 중... (10세트 데이터 조합 모드)")
        engine = M5UltimateEngine(self.sheets) 
        
        f = io.StringIO()
        with redirect_stdout(f):
            final_sets, hot_nums, m5_stats = engine.execute_strike()
            
        m5_output = f.getvalue()
        
        # 가로챈 M5 엔진의 연산 로그를 터미널 화면에도 실시간으로 쏴줍니다.
        print(m5_output) 
        
        combined_data = f"{m5_stats}\n\n[M5 최종 10세트 예측 조합 산출 결과]\n{m5_output}"
        
        print("   ✅ M5 모델 10세트 예측 및 3단계 중요도 지표 추출 완료. AI 모델에 전송합니다.")
        return final_sets, hot_nums, combined_data

    def request_tactical_briefing(self, m5_data):
        print("\n" + "="*60)
        print(" 🧠 [AI 리포트 생성] 계층형 모델 호출 프로세스를 시작합니다.")
        print("="*60 + "\n")
        
        directives = self._load_memory_directives()
        directives_str = json.dumps(directives, ensure_ascii=False, indent=2) if directives else "수신된 오답노트 지시사항 없음 (기본 시스템 룰 적용)"

        prompt = f"""
        당신은 데이터 과학과 통계 분석에 능통한 AI 데이터 분석가입니다. 
        아래 M5 모델이 산출한 데이터와, '지난 회차 오답노트 지시사항'을 종합적으로 분석하여 리포트를 작성하십시오.
        
        [이전 회차 오답노트 지시사항 (m5_memory.json)]
        {directives_str}

        [M5 연산 데이터 (최종 산출물)]
        {m5_data}

        위 데이터를 분석하여 다음 형식으로 객관적이고 명료한 데이터 분석 보고서 어투(~합니다, ~분석됩니다)로 리포트를 작성하십시오:
        
        1. 📊 [3단계 중요도 분석]: M5 모델이 선정한 Tier 1/2/3 핵심 번호들의 통계적 추천 사유를 지표 데이터를 들어 설명하십시오.
        2. 🎯 [10세트 조합 패턴 해설]: 생성된 10세트 조합의 통계적 특징과 가중치 분배 패턴을 분석하십시오.
        3. 💡 [핵심 요약]: 이번 주 가장 주목해야 단 하나의 '결정적 번호(Key Number)'를 지목하고 그 사유를 명시하십시오.
        4. 🔄 [자가 학습 반영 브리핑]: 이번 M5 엔진의 최종 산출 결과가, 상단에 제공된 '이전 회차 오답노트 지시사항'을 어떻게 수용하고 반영했는지 그 인과관계를 지휘관에게 명확하게 보고하십시오.
        """

        pipeline = self.armory.get_model_pipeline(target_tier="고급")

        for idx, model_name in enumerate(pipeline):
            print(f"   🔄 [API 요청 {idx+1}/{len(pipeline)}] 모델({model_name}) 호출 중...")
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                print(f"   🎯 [처리 완료] 모델({model_name}) 분석 리포트 생성 성공.\n")
                return response.text 
            except Exception as e:
                print(f"   ⚠️ [오류] {model_name} 통신 실패 (사유: {e}). 차순위 모델로 전환합니다.")

        return "🚨 [프로세스 실패] 가용한 모든 AI 모델 통신에 실패했습니다."

    def save_to_spreadsheet(self, final_sets, hot_nums, briefing_text):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sh = self.sheets.doc 
        print(f"\n📡 [데이터 저장] {now} 기준 시트 기록을 시작합니다...")

        try:
            df = self.sheets.get_history_data()
            latest_draw = int(df.index.max() if df.index.name == '회차' else df['회차'].max())
            target_draw = latest_draw + 1

            ws_rec = sh.worksheet("추천번호")
            ws_rec.clear()
            ws_rec.update('A1', [[f"제 {target_draw}회차 M5 예측 10세트 조합"]])
            
            # ========================================================
            # [버그 수정] numpy.int64 타입을 구글 시트가 읽을 수 있도록 순수 Python int로 강제 변환
            # ========================================================
            clean_final_sets = [[int(num) for num in lotto_set] for lotto_set in final_sets]
            ws_rec.update('A2', clean_final_sets)
            
            print("   ✅ [추천번호] 탭 최신 데이터 갱신 완료.")
        except Exception as e:
            print(f"   ⚠️ 추천번호 탭 저장 실패: {e}")
            target_draw = "알수없음"

        try:
            ws_log = sh.worksheet("기록")
            hot_nums_str = ", ".join(map(str, sorted(hot_nums)))
            
            # ========================================================
            # [수정 완료] 기존 데이터 중복 방지 (삭제 후 덮어쓰기)
            # ========================================================
            all_logs = ws_log.get_all_values()
            for i, row in enumerate(all_logs):
                # 1번 인덱스(B열: 회차)가 동일한 기록을 찾으면 행 삭제
                if len(row) > 1 and str(row[1]).strip() == str(target_draw):
                    ws_log.delete_rows(i + 1)
                    break
            
            ws_log.insert_row([now, target_draw, hot_nums_str, briefing_text[:4000]], 2)
            print(f"   ✅ [기록] 탭 2번 행에 제 {target_draw}회차 분석 리포트 저장(덮어쓰기) 완료.")
        except Exception as e:
            print(f"   ⚠️ 기록 탭 저장 실패: {e}")