# -*- coding: utf-8 -*-
from phase_02.data_processor import DataProcessor
from phase_02.ai_core import AICore
from phase_02.ai_core_seq import AICoreSeq  # [NEW] 제2 코어 장착
from phase_02.tactical_matrix import TacticalMatrix

class M5UltimateEngine:
    """
    🚀 [Phase 02] M5 통합 컨트롤러 (듀얼 코어 융합 모드)
    - Alpha Engine (정통 통계 분류기 + 조립조) 5세트
    - Beta Engine (시퀀스 생성형 AI) 5세트
    """
    def __init__(self, sheets_handler):
        self.sheets = sheets_handler
        self.dp = DataProcessor(self.sheets)
        
        # 듀얼 코어(두 개의 뇌) 이식
        self.ai_alpha = AICore(self.dp)
        self.ai_beta = AICoreSeq(self.dp)
        
        self.matrix = TacticalMatrix()

    def execute_strike(self):
        print("\n🚀 [M5 통합 컨트롤러] 듀얼 코어(Alpha & Beta) 융합 작전 개시")
        
        # 1. 정보 지원조: 데이터 로드
        print("   📡 [정보 지원조] 전장 데이터 로드 및 텐서망 구축 중...")
        data = self.dp.load_data()
        
        # ====================================================
        # [제1 코어: Alpha Engine] 정통 퀀트 분석
        # ====================================================
        print("\n   [제1 코어: Alpha Engine 가동 (통계적 확률 + 물리적 통제)]")
        self.ai_alpha.run_evolutionary_training(data)
        final_probs = self.ai_alpha.predict_final_probs(data)
        
        # 기존 매트릭스(조립조)에서 10개를 뽑지만, 상위 5개만 Alpha 세트로 편입
        alpha_full_sets, hot_nums, stats_log = self.matrix.build_matrix(final_probs)
        alpha_sets = alpha_full_sets[:5]
        
        # ====================================================
        # [제2 코어: Beta Engine] 조건부 시퀀스 자율 생성
        # ====================================================
        print("\n   [제2 코어: Beta Engine 가동 (미니 GPT + 조건부 자율 발권)]")
        self.ai_beta.run_sequence_training(data)
        beta_sets = self.ai_beta.generate_beta_sets(data, num_sets=5)
        
        # ====================================================
        # [최종 매트릭스 융합]
        # ====================================================
        final_sets = alpha_sets + beta_sets
        
        print("\n   ✅ [듀얼 코어 융합 완료] Alpha(5세트) + Beta(5세트) 총 10세트 구축 성공.")
        
        return final_sets, hot_nums, stats_log