# -*- coding: utf-8 -*-
import sys
import logging

# ==========================================
# 📋 [System] 로깅 설정 (화면에 깔끔하게 보여주기)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def run_all_manual():
    logging.info("🚀 [수동 타격 명령] 지휘관의 명령으로 M5 기지의 모든 작전을 즉시 연속 실행합니다.")

    # 1. 코어 엔진 불러오기
    try:
        from lotto_predict import LottoOrchestrator
        logging.info("✅ 코어 엔진(lotto_predict.py) 장착 완료.")
    except ImportError:
        logging.error("❌ 'lotto_predict.py' 파일을 찾을 수 없습니다. 같은 폴더에 있는지 확인하십시오.")
        sys.exit(1)

    # 2. 사령탑 객체 생성
    orchestrator = LottoOrchestrator()

    # 3. 4단계 연속 타격 실시
    try:
        logging.info("==================================================")
        logging.info("▶️ [Phase 1] 데이터 무중단 동기화 (Sync) 개시...")
        orchestrator.sync_data()

        logging.info("==================================================")
        logging.info("▶️ [Phase 2] M5 가속 딥러닝 훈련 (Train) 개시...")
        orchestrator.train_brain()

        logging.info("==================================================")
        logging.info("▶️ [Phase 3] 제미나이 정예 번호 예측 (Predict) 개시...")
        orchestrator.load_and_predict()

        logging.info("==================================================")
        logging.info("▶️ [Phase 4] 성과 평가 (Evaluate) 개시...")
        orchestrator.evaluate_performance()
        
        logging.info("==================================================")
        logging.info("🎉 [작전 완료] 모든 수동 타격이 성공적으로 끝났습니다! 구글 시트를 확인하십시오.")

    except Exception as e:
        logging.error(f"❌ 작전 수행 중 치명적 오류 발생: {e}")
        logging.info("⚠️ 오류가 발생한 지점에서 강제 타격을 중지합니다. 기지 안전을 확보했습니다.")

if __name__ == "__main__":
    run_all_manual()