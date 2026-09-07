# -*- coding: utf-8 -*-
import math
import numpy as np

class EntropyValidator:
    """
    🧬 [자연계 무작위성 검증 전담 모듈]
    - 조합의 간격 표준편차를 통한 기계적 인위성(패턴) 배제
    - 10세트 전체의 샤논 엔트로피(Shannon Entropy)를 측정하여 편향성 경고
    """
    @staticmethod
    def is_natural_gap(combo):
        """개별 조합의 번호 간 간격(Gap) 표준편차를 계산하여 인위성을 걸러냅니다."""
        gaps = [combo[i+1] - combo[i] for i in range(5)]
        gap_std = np.std(gaps)
        return 2.0 <= gap_std <= 15.0

    @staticmethod
    def calculate_matrix_entropy(final_sets):
        """전체 10세트 매트릭스의 정보 엔트로피를 계산합니다."""
        all_numbers = [num for combo in final_sets for num in combo]
        counts = [all_numbers.count(i) for i in range(1, 46)]
        probabilities = [c / 60.0 for c in counts if c > 0]
        return -sum(p * math.log2(p) for p in probabilities)

    @staticmethod
    def evaluate_and_print(final_sets):
        """엔트로피 지수를 평가하고 콘솔에 경고/통과 메시지를 출력합니다."""
        entropy = EntropyValidator.calculate_matrix_entropy(final_sets)
        print(f"      🧬 [자연계 무작위성 검증] 시스템 정보 엔트로피 지수: {entropy:.2f} (적정 수준: 4.8 ~ 5.4)")
        if entropy < 4.8:
            print("         -> ⚠️ 경고: 조합이 인위적으로 너무 편향되어 있습니다. (기계적 패턴 과다)")
        elif entropy > 5.4:
            print("         -> ⚠️ 경고: 조합이 너무 골고루 퍼져 있습니다. (인간의 인위적 분산 의도 감지)")
        else:
            print("         -> ✅ 통과: 완벽한 자연 무작위성(Chaos)과 통계적 필터가 결합된 최적의 상태입니다.")
        return entropy