# -*- coding: utf-8 -*-
import os
import sys
import logging
import subprocess
import time
from lotto_predict import LottoOrchestrator

# 🎯 [관제탑 로그 설정]
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 [관제탑] Sniper V7 수동 타격 모드 (리셋 스위치 탑재) 가동...")
    
    try:
        orchestrator = LottoOrchestrator()
        logger.info("📡 [작전 개시] 사령관님의 직접 명령 수신.")
        
        # ==========================================
        # 🔄 [Phase 0] 과거 기록 소각 (리셋 스위치)
        # ==========================================
        logger.info("🔄 [초기화 명령 감지] 과거의 작전 완료 기록을 모두 소각합니다.")
        try:
            ws = orchestrator.sheets.get_ws("Remote_Control")
            ws.update_acell('D10', 'FALSE') # 시트의 체크박스 해제
            logger.info("✅ 기록 소각 완료. 시트의 체크박스를 자동으로 껐습니다.")
        except Exception as e:
            logger.warning(f"⚠️ 체크박스 초기화 실패: {e}")

        # ==========================================
        # 🎯 [Phase 1~4] 본대 작전 (안전 구역)
        # ==========================================
        orchestrator.run_sync()
        orchestrator.train_m5_engine()
        orchestrator.generate_hybrid_combinations()
        orchestrator.evaluate_performance()

        logger.info("🏁 명령하신 4개의 임무를 마쳤습니다.")

        # ==========================================
        # 🧬 [Phase 5] 자가 진화 프로토콜 (실험 구역)
        # ==========================================
        evolution_file = "evolution_manager.py"

        if os.path.exists(evolution_file):
            logger.info("⏳ 시트 동기화 대기 중... (3초 딜레이 적용)")
            time.sleep(3) # 🎯 시트에 데이터가 완전히 안착할 시간을 벌어줍니다.
            
            logger.info("🧬 [Phase 5] 자가 진화 (Evolution) 프로토콜 개시...")
            try:
                subprocess.run([sys.executable, evolution_file], check=True)
                logger.info("✅ 자가 진화 완료. 다음 작전에 새로운 전술이 적용됩니다.")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ 자가 진화 오류 발생 (로또 예측은 정상 보존됨): {e}")
            except Exception as e:
                logger.warning(f"⚠️ 알 수 없는 오류: {e}")
        else:
            logger.warning(f"⚠️ {evolution_file} 파일이 존재하지 않아 진화 단계를 건너뜁니다.")

        logger.info("🏁 관제탑: 모든 작전이 성공적으로 종료되었습니다.")

    except Exception as e:
        logger.error(f"❌ 통합 작전 실패: {e}")

if __name__ == "__main__":
    main()