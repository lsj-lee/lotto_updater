# -*- coding: utf-8 -*-
import schedule
import time
import logging
import sys
import os
import torch
import gc
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# -----------------------------------------------------------------------------
# 🧩 모듈 로딩 (lotto_predict.py 및 evolution_manager.py)
# -----------------------------------------------------------------------------
try:
    from lotto_predict import LottoOrchestrator
    print("✅ 'lotto_predict.py' 모듈 로드 성공")
except ImportError:
    logging.error("❌ 'lotto_predict.py'를 찾을 수 없습니다. 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

try:
    from evolution_manager import EvolutionManager
    print("✅ 'evolution_manager.py' 모듈 로드 성공")
except ImportError:
    logging.warning("⚠️ 'evolution_manager.py'가 아직 없습니다. 자율 진화 기능이 비활성화됩니다.")
    EvolutionManager = None

# -----------------------------------------------------------------------------
# ⚙️ M5 하드웨어 안전장치 및 설정
# -----------------------------------------------------------------------------
USED_CORES = 6
torch.set_num_threads(USED_CORES)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"🚀 [System] M5 Neural Engine Activated (MPS/Metal). Cores: {USED_CORES}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ [System] MPS 가속을 사용할 수 없습니다. CPU 모드로 실행합니다.")

# -----------------------------------------------------------------------------
# 🛰️ 메인 스케줄러 클래스 (Orchestrator)
# -----------------------------------------------------------------------------
class LottoScheduler:
    def __init__(self):
        self.orchestrator = LottoOrchestrator()
        self.evolution_manager = EvolutionManager() if EvolutionManager else None
        logging.info("🤖 Hybrid Sniper V5 OrchestratorInitialized.")

    def run_safe(self, task_name, func, *args):
        """작업을 안전하게 실행하고 예외를 처리하는 래퍼 함수"""
        logging.info(f"▶️ [작업 시작] {task_name}")
        try:
            # 메모리 정리 (M5 리소스 보호)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            func(*args)
            logging.info(f"✅ [작업 완료] {task_name}")
        except Exception as e:
            logging.error(f"❌ [작업 실패] {task_name}: {str(e)}")
            # 치명적 오류 발생 시 관리자 알림 로직 추가 가능

    # --- 개별 작업 정의 ---

    def job_sync(self):
        """데이터 동기화 (토요일 밤)"""
        self.run_safe("Data Synchronization", self.orchestrator.sync_data)

    def job_train(self):
        """모델 학습 (월요일 밤)"""
        # train_brain은 모델과 데이터를 반환하므로 래퍼 필요
        def _train():
            model, data = self.orchestrator.train_brain()
            if model:
                logging.info("🧠 모델 학습 완료 및 저장됨.")
            else:
                logging.warning("⚠️ 학습 데이터 부족으로 모델 학습 건너뜀.")
        self.run_safe("Model Training", _train)

    def job_predict(self):
        """번호 예측 및 보고서 생성 (수요일 저녁)"""
        def _predict():
            # 예측을 위해 데이터를 다시 로드하거나 상태를 확인
            # lotto_predict.py의 구조상 train_brain이 데이터를 리턴하지만,
            # 여기서는 예측만 수행하려면 데이터를 다시 로드해야 함.
            # orchestrator에 데이터 로드 기능이 통합되어 있다고 가정하거나 추가 구현 필요.
            # 현재 lotto_predict.py의 train_brain에서 데이터를 로드함.
            # 효율성을 위해 predict_only 모드를 lotto_predict.py에 추가하는 것이 좋음.
            # 임시로 train_brain을 호출하여 데이터를 얻거나, 별도 로드 함수 사용.

            # (수정 예정인 lotto_predict.py에 load_data 메소드 추가 필요)
            # 여기서는 편의상 train_brain을 호출하여 최신 모델로 예측 (또는 저장된 모델 로드)
            logging.info("🔮 예측 시나리오 생성 중...")
            model, data = self.orchestrator.train_brain() # 재학습 또는 로드
            if model and data:
                self.orchestrator.generate_report(model, data)

        self.run_safe("Prediction & Reporting", _predict)

    def job_evaluate(self):
        """성과 평가 (목요일 아침) - Reward System"""
        if hasattr(self.orchestrator, 'evaluate_performance'):
            self.run_safe("Performance Evaluation", self.orchestrator.evaluate_performance)
        else:
            logging.warning("⚠️ 'evaluate_performance' 메소드가 구현되지 않았습니다.")

    def job_evolution(self):
        """자율 진화 제안 (금요일 저녁) - Self-Evolution"""
        if self.evolution_manager:
            logging.info("🧬 [Self-Evolution] 코드 분석 및 진화 제안 시작...")
            # 터미널 상호작용을 위해 메인 스레드에서 실행
            # 백그라운드 실행 중이라면 로그만 남기고, 사용자가 직접 실행하도록 유도
            if sys.stdin.isatty():
                self.evolution_manager.execute_evolution_cycle('lotto_predict.py')
            else:
                logging.info("ℹ️ 백그라운드 실행 중입니다. 진화 제안은 'python evolution_manager.py'를 수동 실행하세요.")
        else:
            logging.warning("⚠️ Evolution Manager가 로드되지 않았습니다.")

# -----------------------------------------------------------------------------
# 🕒 스케줄 설정
# -----------------------------------------------------------------------------
def start_scheduler():
    bot = LottoScheduler()

    # 1. 데이터 동기화 (매주 토요일 21:00) - 추첨 직후
    schedule.every().saturday.at("21:00").do(bot.job_sync)

    # 2. 모델 학습 (매주 월요일 21:00) - 데이터 분석 및 학습
    schedule.every().monday.at("21:00").do(bot.job_train)

    # 3. 예측 보고서 (매주 수요일 18:00) - 목요일 구매 전
    schedule.every().wednesday.at("18:00").do(bot.job_predict)

    # 4. 성과 평가 (매주 목요일 09:00) - 지난주 결과 복기
    schedule.every().thursday.at("09:00").do(bot.job_evaluate)

    # 5. 자율 진화 (매주 금요일 20:00) - 주말 전 시스템 점검 및 업데이트
    schedule.every().friday.at("20:00").do(bot.job_evolution)

    logging.info("🚀 [Scheduler] Hybrid Sniper V5 자동화 시스템이 시작되었습니다.")
    logging.info("   - 토 21:00: 데이터 동기화")
    logging.info("   - 월 21:00: 모델 학습")
    logging.info("   - 수 18:00: 번호 예측")
    logging.info("   - 목 09:00: 성과 평가")
    logging.info("   - 금 20:00: 자율 진화 제안")
    logging.info("   (Ctrl+C로 종료)")

    while True:
        schedule.run_pending()
        time.sleep(60) # 1분마다 체크

if __name__ == "__main__":
    start_scheduler()
