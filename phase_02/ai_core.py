# -*- coding: utf-8 -*-
import time
import numpy as np
import copy
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')

class AICore:
    """
    🧠 [Phase 02] 전략 훈련소 (Self-Evolving & Command Directed)
    - [NEW] 클래스 불균형(Class Imbalance) 해결을 위한 샘플 가중치(6.5배) 훈련 도입
    """
    def __init__(self, data_processor):
        self.dp = data_processor
        self.top_models = []

    def run_evolutionary_training(self, data, start_learn=200, step=50):
        print("\n   🤖 [전략 훈련소] 다중 세대 진화 및 모델 학습 개시...")
        latest_idx = len(data)
        model_bank = []
        
        master_model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42, warm_start=True 
        )
        checkpoints = list(range(start_learn, latest_idx - 50, step))
        
        for cp in checkpoints:
            print(f"      📡 데이터 {cp}개 기준 모델 훈련 및 미래 구간 검증 중...", end='\r')
            past_data = data.iloc[:cp]
            X_train, y_train = self._prepare_xy(past_data)
            
            if len(X_train) == 0: continue
            
            # [NEW] 실제 당첨(1) 번호를 맞출 경우 낙첨(0)보다 6.5배 높은 가중치 보상 부여
            sample_weights = np.where(y_train == 1, 6.5, 1.0)
                
            master_model.n_estimators += 10
            master_model.fit(X_train, y_train, sample_weight=sample_weights)
            current_era_model = copy.deepcopy(master_model)
            
            survival_score = self._evaluate_model(current_era_model, data, start_idx=cp, test_length=50)
            model_bank.append({'model': current_era_model, 'score': survival_score, 'generation': cp})

        if not model_bank:
             raise ValueError("❌ 훈련된 모델이 없습니다.")

        model_bank.sort(key=lambda x: x['score'], reverse=True)
        self.top_models = model_bank[:5]
        
        avg_score = np.mean([m['score'] for m in self.top_models])
        print(f"\n   ✅ 훈련 종료. 상위 5개 우수 모델 선발 완료. (평균 성능 점수: {avg_score:.2f}%)")
        return self.top_models

    def _prepare_xy(self, history_df):
        X, y = [], []
        for i in range(50, len(history_df)):
            feat = self.dp.extract_features(history_df.iloc[:i])
            actual = history_df.iloc[i].values
            for j, f in enumerate(feat):
                X.append(f)
                y.append(1 if (j + 1) in actual else 0)
        return np.array(X), np.array(y)

    def _evaluate_model(self, model, data, start_idx, test_length):
        hits, test_count = 0, 0
        end_idx = min(start_idx + test_length, len(data))
        for i in range(start_idx, end_idx):
            feat = self.dp.extract_features(data.iloc[:i])
            probs = model.predict_proba(feat)[:, 1]
            top_15 = np.argsort(probs)[-15:] + 1
            actual = data.iloc[i].values
            hits += len(set(top_15) & set(actual))
            test_count += 1
        return (hits / (test_count * 6)) * 100 if test_count > 0 else 0

    def predict_final_probs(self, current_data):
        print("\n   ⚔️ [앙상블 예측 및 지휘관 가중치 적용] 오답노트 명령을 확률에 주입합니다.")
        feat = self.dp.extract_features(current_data)
        
        combined_probs = np.zeros(45)
        total_weight = sum([m['score'] for m in self.top_models]) or 1
            
        for entry in self.top_models:
            probs = entry['model'].predict_proba(feat)[:, 1]
            combined_probs += (probs * entry['score'])
            
        base_probs = combined_probs / total_weight

        directives = self.dp.memory_data.get("tactical_directives", {})
        if directives:
            zone_w = directives.get("zone_weights", {})
            hot_digits = directives.get("hot_last_digits", [])
            carry_w = directives.get("carryover_weight", 1.0)
            
            last_draw_nums = current_data.iloc[-1].values if len(current_data) > 0 else []

            for i in range(45):
                num = i + 1
                
                zone_idx = (num - 1) // 10 + 1
                if zone_idx > 5: zone_idx = 5
                z_key = f"zone_{zone_idx}"
                base_probs[i] *= zone_w.get(z_key, 1.0)
                
                if (num % 10) in hot_digits:
                    base_probs[i] *= 1.15
                    
                if num in last_draw_nums:
                    base_probs[i] *= carry_w

        return base_probs