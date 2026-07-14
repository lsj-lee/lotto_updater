# -*- coding: utf-8 -*-
from phase_02.data_processor import DataProcessor
from phase_02.ai_core import AICore
from phase_02.tactical_matrix import TacticalMatrix

class M5UltimateEngine:
    """
    🚀 [Phase 02] M5 통합 컨트롤러
    - 데이터 프로세서, AI 코어, 전술 매트릭스를 지휘하여 최종 번호 세트를 산출합니다.
    """
    def __init__(self, sheets_handler):
        self.sheets = sheets_handler
        self.dp = DataProcessor(self.sheets)
        self.ai = AICore(self.dp)
        self.matrix = TacticalMatrix()

    def execute_strike(self):
        print("\n🚀 [M5 통합 컨트롤러] 작전 개시")
        
        # 1. 정보 지원조: 구글 시트에서 과거 당첨 데이터 로드 및 진화 규칙(Rule) 생성
        print("   📡 [정보 지원조] 전장 데이터 로드 및 텐서망 구축 중...")
        data = self.dp.load_data()
        
        # 2. 전략 훈련소: 다중 세대 진화 훈련 및 상위 5대 공식 선발
        self.ai.run_evolutionary_training(data)
        
        # 3. 전술 예측: 5대 앙상블 공식을 결합하여 1~45번 최종 확률표 도출
        final_probs = self.ai.predict_final_probs(data)
        
        # 4. 무기 조립조: 도출된 확률을 바탕으로 15개 핫존을 나누고 10세트 매트릭스 완성
        print("   ⚙️ [무기 조립조] 10대 전술 매트릭스 조립 중...")
        final_sets, hot_nums, stats_log = self.matrix.build_matrix(final_probs)
        
        return final_sets, hot_nums, stats_log