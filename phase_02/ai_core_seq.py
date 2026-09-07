# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

class AICoreSeq:
    """
    🧠 [Phase 02] 제2 코어: 조건부 시퀀스 생성 AI (Beta Engine)
    - 과거 당첨 데이터를 '순서와 문맥'으로 학습하는 Autoregressive 신경망
    - 확률 샘플링(Top-K) 및 로짓 바이어스(오답노트 개입)를 통한 하드코딩 없는 자율 발권
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
        
        # 500회차만 사용하여 과적합 방지 및 최신 트렌드 반영
        recent_data = history_data.tail(500).values.tolist()
        
        for draw in recent_data:
            draw = sorted(draw)
            context = [0, 0, 0, 0, 0] # 앞서 뽑힌 번호를 기억하는 5칸의 메모리
            for i in range(6):
                X_train.append(context.copy())
                y_train.append(draw[i])
                if i < 5:
                    context[i] = draw[i]
                    
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("\n   ✅ [제2 코어] 시퀀스 딥러닝 훈련 완료.")

    def generate_beta_sets(self, current_data, num_sets=5):
        print("   🧬 [제2 코어] 하드코딩 필터 없는 시퀀스 자율 발권을 시작합니다 (Beta 5세트)...")
        if not self.is_trained:
            return []
            
        beta_sets = []
        
        # 오답노트(조건부 개입) 스캔
        directives = self.dp.memory_data.get("tactical_directives", {})
        zone_w = directives.get("zone_weights", {})
        hot_digits = directives.get("hot_last_digits", [])
        carry_w = directives.get("carryover_weight", 1.0)
        last_draw_nums = current_data.iloc[-1].values if len(current_data) > 0 else []

        for _ in range(num_sets):
            context = [0, 0, 0, 0, 0]
            ticket = []
            
            for i in range(6):
                # 1. 문맥을 바탕으로 1~45번 순수 다음 번호 확률 예측
                raw_probs = self.model.predict_proba([context])[0]
                
                # 2. 조건부 개입 (Conditional Decoding) - 지휘관의 오답노트를 확률에 직접 주입
                for num_idx in range(45):
                    num = num_idx + 1
                    
                    # 이미 뽑힌 번호는 확률 0으로 파괴 (중복 불가)
                    if num in ticket:
                        raw_probs[num_idx] = 0.0
                        continue
                        
                    # 구간 가중치 주입
                    z_idx = (num - 1) // 10 + 1
                    z_idx = 5 if z_idx > 5 else z_idx
                    z_key = f"zone_{z_idx}"
                    raw_probs[num_idx] *= zone_w.get(z_key, 1.0)
                    
                    # 끝수 저격 주입
                    if (num % 10) in hot_digits:
                        raw_probs[num_idx] *= 1.15
                        
                    # 이월수 가중치 주입
                    if num in last_draw_nums:
                        raw_probs[num_idx] *= carry_w
                
                # 3. 확률 정규화 (전체 합을 1로 맞춤)
                prob_sum = np.sum(raw_probs)
                if prob_sum > 0:
                    normalized_probs = raw_probs / prob_sum
                else:
                    normalized_probs = np.ones(45) / 45.0
                    
                # 4. Top-K 확률 기반 샘플링 (매번 똑같은 조합이 앵무새처럼 나오는 것을 방지)
                # 가장 확률이 높은 상위 5개 번호 중, 각자의 확률 비율에 맞춰 자연스럽게 하나를 선택
                top_k_indices = np.argsort(normalized_probs)[-5:]
                top_k_probs = normalized_probs[top_k_indices]
                top_k_probs = top_k_probs / np.sum(top_k_probs) # K개 안에서 재정규화
                
                next_num = np.random.choice(top_k_indices, p=top_k_probs) + 1
                
                ticket.append(next_num)
                if i < 5:
                    context[i] = next_num # 뽑힌 번호를 다음 추론을 위한 문맥에 추가
                    
            beta_sets.append(sorted(ticket))
            
        return beta_sets