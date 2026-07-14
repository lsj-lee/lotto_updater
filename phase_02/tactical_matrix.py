# -*- coding: utf-8 -*-
import numpy as np
import random
import json
import os

class TacticalMatrix:
    """
    ⚙️ [Phase 02] 무기 조립조 (Tactical Matrix Builder & Macro Filter)
    - AI가 도출한 확률을 바탕으로 번호를 계급화(Tier 1, 2, 3)합니다.
    - 오답노트(m5_memory.json)의 지시에 따라 불량 조합을 강제 폐기하고 다시 뽑습니다(Re-roll).
    """
    def __init__(self):
        self.memory_file = "m5_memory.json"
        self.directives = self._load_directives()

    def _load_directives(self):
        """Phase 04가 작성한 오답노트 작전 지시서를 로컬에서 은밀히 스캔합니다."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("tactical_directives", {})
            except:
                return {}
        return {}

    def build_matrix(self, final_probs):
        print("   ⚙️ [무기 조립조] 분석된 확률을 바탕으로 15개 핫존 및 전술 매트릭스 조립을 시작합니다...")
        
        # [NEW] 오답노트 지시사항 확인
        req_consecutive = self.directives.get("require_consecutive", False)
        if req_consecutive:
            print("      ⚠️ [품질 통제] 지휘관 오답노트 지시 수신: '연번(연속된 번호)' 필수 포함 조합만 합격시킵니다.")
            
        # 1. 🥇 번호 계급화 (Ranking)
        ranked_indices = np.argsort(final_probs)[::-1]
        ranked_numbers = [int(idx + 1) for idx in ranked_indices]
        
        # 2. 🎯 핫존(15개) 및 콜드존(30개) 분리
        hot_zone = ranked_numbers[:15]
        cold_zone = ranked_numbers[15:]
        
        tier1 = hot_zone[0:5]   # 1~5위 (최우선 타격 목표)
        tier2 = hot_zone[5:10]  # 6~10위 (주력 예비대)
        tier3 = hot_zone[10:15] # 11~15위 (후방 지원대)
        
        # 3. ⚔️ 10대 전술 매트릭스 조립 (필터링 적용)
        final_sets = []
        
        # [전술 1~5] 핫존 집중형
        final_sets.append(self._draft(tier1, 3, tier2, 2, tier3, 1, req_consec=req_consecutive))
        final_sets.append(self._draft(tier1, 2, tier2, 2, tier3, 2, req_consec=req_consecutive))
        final_sets.append(self._draft(tier1, 2, tier2, 3, tier3, 1, req_consec=req_consecutive))
        final_sets.append(self._draft(tier1, 1, tier2, 2, tier3, 3, req_consec=req_consecutive))
        final_sets.append(self._draft(tier1, 4, tier2, 1, tier3, 1, req_consec=req_consecutive))
        
        # [전술 6~10] 외곽 우회 및 이변 대비형 (콜드존 포함)
        final_sets.append(self._draft(tier1, 2, tier2, 1, tier3, 2, cold_zone, 1, req_consec=req_consecutive))
        final_sets.append(self._draft(tier1, 1, tier2, 2, tier3, 2, cold_zone, 1, req_consec=req_consecutive))
        final_sets.append(self._draft(tier1, 2, tier2, 2, tier3, 1, cold_zone, 1, req_consec=req_consecutive))
        final_sets.append(self._draft(tier1, 1, tier2, 1, tier3, 2, cold_zone, 2, req_consec=req_consecutive))
        final_sets.append(self._draft(tier1, 2, tier2, 1, tier3, 1, cold_zone, 2, req_consec=req_consecutive))

        # 4. 📊 참모장 브리핑용 통계 로그 작성
        stats_log = {
            "tier1": tier1,
            "tier2": tier2,
            "tier3": tier3,
            "cold_sample": cold_zone[:5] 
        }
        
        print("   ✅ 10대 전술 매트릭스 조립 및 거시적 품질 검열 완료. 타격 준비 끝.")
        return final_sets, hot_zone, stats_log

    def _draft(self, t1, n1, t2, n2, t3, n3, cold=None, nc=0, req_consec=False):
        """지정된 개수만큼 번호를 차출하되, 지시된 필터(예: 연번)를 통과할 때까지 다시 뽑습니다."""
        max_retries = 300 # 무한 루프(시스템 뻗음)를 막기 위한 안전장치
        
        for attempt in range(max_retries):
            selected = []
            selected.extend(random.sample(t1, n1))
            selected.extend(random.sample(t2, n2))
            selected.extend(random.sample(t3, n3))
            
            if cold and nc > 0:
                selected.extend(random.sample(cold, nc))
                
            selected = sorted(selected)
            
            # ==========================================
            # [거시적 품질 통제망] 불량 조합 필터링
            # ==========================================
            if req_consec:
                # 리스트 안에 i번째 번호에 1을 더한 값이 i+1번째 번호와 같은지(연번인지) 확인
                has_consecutive = any(selected[i] + 1 == selected[i+1] for i in range(len(selected)-1))
                if not has_consecutive:
                    continue # 불합격: 조합 폐기하고 맨 위 for문으로 돌아가 다시 뽑기 (Re-roll)
            
            # 모든 검열을 무사히 통과했다면 세트 반환
            return selected
            
        # 300번을 넘게 돌려도 연번이 안 나왔다면 (핫존 번호 분포가 심하게 띄엄띄엄 있을 경우)
        # 시스템 과부하를 막기 위해 마지막으로 뽑힌 조합을 그대로 반환하며 타협
        return sorted(selected)