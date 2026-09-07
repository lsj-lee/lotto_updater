# -*- coding: utf-8 -*-
import numpy as np
import random
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# [NEW] 독립된 검증 모듈 수입
from core.entropy_validator import EntropyValidator

def _monte_carlo_worker(t1, n1, t2, n2, t3, n3, cold, nc, req_consec, max_retries=50000):
    for _ in range(max_retries):
        selected = []
        selected.extend(random.sample(t1, n1))
        selected.extend(random.sample(t2, n2))
        selected.extend(random.sample(t3, n3))
        
        if cold and nc > 0:
            selected.extend(random.sample(cold, nc))
            
        combo = sorted(selected)
        
        # [NEW] 유연한 전술 매트릭스: 연번 필수 포함 조건을 50% 확률로만 적용 (엔트로피 상승 유도)
        if req_consec:
            if random.random() < 0.5:
                has_consecutive = any(combo[i] + 1 == combo[i+1] for i in range(len(combo)-1))
                if not has_consecutive: continue 

        total_sum = sum(combo)
        if total_sum < 90 or total_sum > 180: continue
        
        odds = sum(1 for n in combo if n % 2 != 0)
        if odds == 0 or odds == 6: continue
        
        lows = sum(1 for n in combo if n <= 22)
        if lows == 0 or lows == 6: continue
        
        last_digits = [n % 10 for n in combo]
        if any(last_digits.count(d) >= 3 for d in range(10)): continue

        # [NEW] 독립 모듈을 호출하여 검증
        if not EntropyValidator.is_natural_gap(combo):
            continue
            
        return combo
    return sorted(selected)

class TacticalMatrix:
    def __init__(self):
        self.memory_file = "m5_memory.json"
        self.directives = self._load_directives()

    def _load_directives(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("tactical_directives", {})
            except:
                return {}
        return {}

    def build_matrix(self, final_probs):
        print("   ⚙️ [무기 조립조] 분석된 확률을 바탕으로 전술 매트릭스 조립을 시작합니다...")
        
        req_consecutive = self.directives.get("require_consecutive", False)
        if req_consecutive:
            print("      ⚠️ [품질 통제] '연번' 필수 조건을 50% 확률로 유연하게 적용합니다 (엔트로피 완화).")
        print("      🛡️ [방어망 가동] CPU 멀티프로세싱 기반 몬테카를로 무작위성(Randomness) 판별망 활성화.")
            
        ranked_indices = np.argsort(final_probs)[::-1]
        ranked_numbers = [int(idx + 1) for idx in ranked_indices]
        
        hot_zone = ranked_numbers[:15]
        cold_zone = ranked_numbers[15:]
        
        tier1, tier2, tier3 = hot_zone[0:5], hot_zone[5:10], hot_zone[10:15]
        
        tasks = [
            (tier1, 3, tier2, 2, tier3, 1, None, 0, req_consecutive),
            (tier1, 2, tier2, 2, tier3, 2, None, 0, req_consecutive),
            (tier1, 2, tier2, 3, tier3, 1, None, 0, req_consecutive),
            (tier1, 1, tier2, 2, tier3, 3, None, 0, req_consecutive),
            (tier1, 4, tier2, 1, tier3, 1, None, 0, req_consecutive),
            (tier1, 2, tier2, 1, tier3, 2, cold_zone, 1, req_consecutive),
            (tier1, 1, tier2, 2, tier3, 2, cold_zone, 1, req_consecutive),
            (tier1, 2, tier2, 2, tier3, 1, cold_zone, 1, req_consecutive),
            (tier1, 1, tier2, 1, tier3, 2, cold_zone, 2, req_consecutive),
            (tier1, 2, tier2, 1, tier3, 1, cold_zone, 2, req_consecutive)
        ]
        
        final_sets = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_monte_carlo_worker, *task) for task in tasks]
            for future in as_completed(futures):
                final_sets.append(future.result())
                
        final_sets = sorted(final_sets)

        # [NEW] 모듈을 통해 검증 후 통계 로그 저장
        system_entropy = EntropyValidator.evaluate_and_print(final_sets)

        stats_log = {
            "tier1": tier1, "tier2": tier2, "tier3": tier3,
            "cold_sample": cold_zone[:5],
            "entropy_score": system_entropy
        }
        print("   ✅ 10대 전술 매트릭스 병렬 조립 및 거시적 품질 검열 완료. 타격 준비 끝.")
        return final_sets, hot_zone, stats_log