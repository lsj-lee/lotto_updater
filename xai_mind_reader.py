# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

from core.sheets_handler import SheetsHandler
from phase_02.data_processor import DataProcessor
from phase_02.ai_core import AICore
from phase_02.ai_core_seq import AICoreSeq

def extract_importance(model):
    if hasattr(model, 'steps'):
        model = model.steps[-1][1]
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_)
        if coefs.ndim > 1: coefs = np.mean(coefs, axis=0)
        return coefs
    return None

def force_scan_alpha_memory(alpha_obj):
    """Alpha 객체의 메모리를 전수 조사하여 숨겨진 모델을 찾아내는 강제 스캔 함수"""
    for attr_name, attr_value in vars(alpha_obj).items():
        if isinstance(attr_value, list) and len(attr_value) > 0:
            for item in attr_value:
                # [수정 완료] 딕셔너리(Dictionary) 형태로 저장된 실제 ai_core.py 구조 완벽 탐지
                if isinstance(item, dict) and 'model' in item:
                    sub_item = item['model']
                    if hasattr(sub_item, 'fit') and hasattr(sub_item, 'predict'):
                        return sub_item, [f"지표_{i+1}" for i in range(100)]
                
                # 튜플(Tuple) 형태로 저장된 경우 (기존 로직 유지)
                elif isinstance(item, tuple):
                    for sub_item in item:
                        if hasattr(sub_item, 'fit') and hasattr(sub_item, 'predict'):
                            features = [x for x in item if isinstance(x, list) and len(x) > 0 and isinstance(x[0], str)]
                            return sub_item, (features[0] if features else [f"지표_{i+1}" for i in range(100)])
    return None, None

def run_mind_reader():
    print("="*60)
    print(" 🔍 [XAI Mind Reader] M5 듀얼 코어 사고 과정 투시 스캐너 가동")
    print("="*60)
    
    try:
        print("\n📡 구글 시트 전장 데이터 로드 중...")
        sheets = SheetsHandler()
        dp = DataProcessor(sheets)
        data = dp.load_data()
        
        # ==========================================
        # 🧠 제1 코어 (Alpha) 생각 읽기
        # ==========================================
        print("\n" + "-"*60)
        print(" 🧠 [제1 코어: Alpha Engine] 사고 과정 분석 (특성 중요도)")
        print("-" * 60)
        alpha = AICore(dp)
        alpha.run_evolutionary_training(data)
        alpha.predict_final_probs(data) 
        
        print("\n🔍 [Alpha 뇌파 스캔]: 예측을 위해 가장 중요하게 평가한 통계 지표 Top 5")
        
        # 메모리 강제 전수 조사 실행
        target_model, target_features = force_scan_alpha_memory(alpha)
                
        if target_model:
            importances = extract_importance(target_model)
            if importances is not None:
                importances = importances / np.sum(importances) # 100% 정규화
                min_len = min(len(target_features), len(importances))
                
                importance_df = pd.DataFrame({
                    'Feature': target_features[:min_len],
                    'Importance': importances[:min_len]
                }).sort_values(by='Importance', ascending=False)
                
                for idx, row in enumerate(importance_df.head(5).itertuples()):
                    print(f"  {idx+1}위: {row.Feature:<20} (영향력: {row.Importance*100:.2f}%)")
            else:
                print("  ⚠️ Alpha 에이스 모델이 속을 알 수 없는 블랙박스 알고리즘(KNN 등)입니다.")
        else:
            print("  ⚠️ 메모리 전수 조사 실패: Alpha 객체 안에 훈련된 모델이 존재하지 않습니다.")

        # ==========================================
        # 🧬 제2 코어 (Beta) 생각 읽기
        # ==========================================
        print("\n" + "-"*60)
        print(" 🧬 [제2 코어: Beta Engine] 사고 과정 분석 (시퀀스 추론 맵)")
        print("-" * 60)
        beta = AICoreSeq(dp)
        beta.run_sequence_training(data)
        
        print("\n🔍 [Beta 뇌파 스캔]: 실시간 문맥(Context)에 따른 확률 저울질")
        
        def print_beta_thought(context_state, situation_name):
            probs = beta.model.predict_proba([context_state])[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            
            print(f"\n  ▶ [상황: {situation_name}] 현재 문맥 {context_state}")
            print("     기계의 뇌 속 확률 저울질 결과:")
            for i, idx in enumerate(top3_idx):
                lotto_num = beta.model.classes_[idx]
                print(f"      - {i+1}순위 채택 후보: [{lotto_num:02d}번] (수학적 확률: {probs[idx]*100:.2f}%)")

        print_beta_thought([0, 0, 0, 0, 0], "맨 처음 첫 번째 공을 뽑을 때")
        print_beta_thought([7, 0, 0, 0, 0], "방금 7번을 뽑고, 두 번째 공을 고를 때")
        print_beta_thought([7, 8, 9, 0, 0], "7, 8, 9번 (3연번)이 뽑혀버린 최악의 상태일 때")

        print("\n" + "="*60)
        print(" ✅ XAI 투시 스캔 및 뇌파 해독 완료.")
        
    except Exception as e:
        print(f"\n🚨 스캐너 가동 중 치명적 오류 발생: {e}")

if __name__ == "__main__":
    run_mind_reader()