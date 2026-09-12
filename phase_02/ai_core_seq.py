# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

class AICoreSeq:
    """
    🧠 [Phase 02] 제2 코어: 조건부 시퀀스 생성 AI (Beta Engine)
    - [NEW] 제로 패딩 착시를 제거하기 위한 45차원 원-핫 인코딩 텐서 매트릭스 적용
    """
    def __init__(self, data_processor):
        self.dp = data_processor
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            max_iter=500,
            random_state=42
        )
        self.is_trained = False

    def run_sequence_training(self, history_data):
        print("      📡 [제2 코어] 시퀀스 뇌신경망(MLP) 가동 및 문맥 학습 중...", end='\r')
        X_train, y_train = [], []
        
        recent_data = history_data.tail(500).values.tolist()
        
        for draw in recent_data:
            draw = sorted(draw)
            # [NEW] 45개의 0으로 이루어진 원-핫 인코딩 텐서망 생성
            context = np.zeros(45, dtype=int)
            for i in range(6):
                X_train.append(context.copy())
                y_train.append(draw[i])
                if i < 5:
                    context[draw[i] - 1] = 1 # 등장한 번호의 인덱스 스위치만 1로 활성화 (ON)
                    
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("\n   ✅ [제2 코어] 시퀀스 딥러닝 훈련 완료.")

    def generate_beta_sets(self, current_data, num_sets=5):
        print("   🧬 [제2 코어] Soft-Weighting이 적용된 시퀀스 자율 발권을 시작합니다 (Beta 5세트)...")
        if not self.is_trained:
            return []
            
        beta_sets = []
        
        directives = self.dp.memory_data.get("tactical_directives", {})
        zone_w = directives.get("zone_weights", {})
        hot_digits = directives.get("hot_last_digits", [])
        carry_w = directives.get("carryover_weight", 1.0)
        last_draw_nums = current_data.iloc[-1].values if len(current_data) > 0 else []

        for _ in range(num_sets):
            # [NEW] 초기 문맥은 모든 스위치가 0(OFF)인 45차원 텐서
            context = np.zeros(45, dtype=int)
            ticket = []
            
            for i in range(6):
                raw_probs = self.model.predict_proba([context])[0]
                
                for num_idx in range(45):
                    num = num_idx + 1
                    
                    if num in ticket:
                        raw_probs[num_idx] = 0.0
                        continue
                        
                    z_idx = (num - 1) // 10 + 1
                    z_idx = 5 if z_idx > 5 else z_idx
                    z_key = f"zone_{z_idx}"
                    original_w = zone_w.get(z_key, 1.0)
                    soft_w = 1.0 + (original_w - 1.0) * 0.5
                    raw_probs[num_idx] *= soft_w
                    
                    if (num % 10) in hot_digits:
                        raw_probs[num_idx] *= 1.075
                        
                    if num in last_draw_nums:
                        soft_cw = 1.0 + (carry_w - 1.0) * 0.5
                        raw_probs[num_idx] *= soft_cw
                
                prob_sum = np.sum(raw_probs)
                if prob_sum > 0:
                    normalized_probs = raw_probs / prob_sum
                else:
                    normalized_probs = np.ones(45) / 45.0
                    
                top_k_indices = np.argsort(normalized_probs)[-5:]
                top_k_probs = normalized_probs[top_k_indices]
                top_k_probs = top_k_probs / np.sum(top_k_probs) 
                
                next_num = np.random.choice(top_k_indices, p=top_k_probs) + 1
                
                ticket.append(next_num)
                if i < 5:
                    context[next_num - 1] = 1 # 발권된 번호 텐서 활성화 (ON)
                    
            beta_sets.append(sorted(ticket))
            
        return beta_sets