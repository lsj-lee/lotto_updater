# -*- coding: utf-8 -*-
import schedule
import time
import logging
import sys
import os
import torch
import gc
import pytz
from datetime import datetime

# ==========================================
# 📋 [System] 로깅 설정
# ==========================================
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
    logging.error("❌ 'lotto_predict.py'를 찾을 수 없습니다. 파일 경로를 확인하세요.")
    sys.exit(1)

try:
    from evolution_manager import EvolutionManager
    print("✅ 'evolution_manager.py' 모듈 로드 성공")
except ImportError:
    logging.warning("⚠️ 'evolution_manager.py'가 없습니다. 자율 진화 기능이 비활성화됩니다.")
    EvolutionManager = None

# -----------------------------------------------------------------------------
# ⚙️ M5 하드웨어 안전장치 및 설정
# -----------------------------------------------------------------------------
USED_CORES = 6
torch.set_num_threads(USED_CORES)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"🚀 [System] M5 Neural Engine (MPS/Metal) 가속 활성화. (Core: {USED_CORES})")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ [System] MPS 가속 불가. CPU 모드로 실행합니다.")

# -----------------------------------------------------------------------------
# 🛰️ 메인 스케줄러 클래스 (Orchestrator)
# -----------------------------------------------------------------------------
class LottoScheduler:
    """
    [Phase 1~4] 전체 파이프라인을 시간표에 맞춰 지휘하는 오케스트레이터입니다.
    """
    def __init__(self):
        self.orchestrator = LottoOrchestrator()
        self.evolution_manager = EvolutionManager() if EvolutionManager else None
        logging.info("🤖 Hybrid Sniper V5 OrchestratorInitialized.")

    def run_safe(self, task_name, func, *args):
        """
        작업 실행 중 오류가 발생해도 스케줄러가 죽지 않도록 보호하는 래퍼 함수입니다.
        작업 전후로 메모리 청소(GC)를 수행하여 M5 시스템을 보호합니다.
        """
        logging.info(f"▶️ [작업 시작] {task_name}")
        try:
            # 메모리 정리 (리소스 보호)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            func(*args)
            logging.info(f"✅ [작업 완료] {task_name}")
        except Exception as e:
            logging.error(f"❌ [작업 실패] {task_name}: {str(e)}")

    # --- 개별 작업 정의 ---

    def job_sync(self):
        """Phase 1: 데이터 동기화 (일요일 02:00)"""
        logging.info("📅 Phase 1: 데이터 동기화 시작 (Naver -> Gemini -> Sheet)")
        self.run_safe("Data Synchronization", self.orchestrator.sync_data)

    def job_train(self):
        """Phase 2: 모델 학습 (월요일 02:00) - 예측 없음"""
        logging.info("📅 Phase 2: AI 모델 학습 시작 (Only Training)")
        self.run_safe("Model Training", self.orchestrator.train_brain)

    def job_predict(self):
        """Phase 3: 하이브리드 예측 (수요일 02:00) - Top 20 -> LLM"""
        logging.info("📅 Phase 3: 번호 예측 및 시트 기록 시작")
        if hasattr(self.orchestrator, 'load_and_predict'):
            self.run_safe("Prediction Only", self.orchestrator.load_and_predict)
        else:
            logging.error("❌ 'load_and_predict' 함수가 없습니다.")

    def job_evaluate(self):
        """Phase 4: 성과 평가 (목요일 02:00) - Reward Log"""
        logging.info("📅 Phase 4: 지난 작전 성과 평가 시작")
        self.run_safe("Performance Evaluation", self.orchestrator.evaluate_performance)

    def job_evolution(self):
        """Phase 4+: 자율 진화 제안 (금요일 02:00)"""
        if self.evolution_manager:
            logging.info("🧬 [Self-Evolution] 코드 분석 및 진화 제안 시작...")
            if sys.stdin.isatty():
                self.evolution_manager.execute_evolution_cycle('lotto_predict.py')
            else:
                logging.info("ℹ️ 백그라운드 모드입니다. 진화 제안은 수동으로 실행하세요.")
        else:
            logging.warning("⚠️ Evolution Manager가 로드되지 않았습니다.")

# -----------------------------------------------------------------------------
# 🕒 KST (한국 시간) 기반 스케줄링 로직
# -----------------------------------------------------------------------------
def run_kst_schedule():
    bot = LottoScheduler()

    print("🚀 [Scheduler] Hybrid Sniper V5 KST(한국 시간) 스케줄러 시작...")
    print("   - 일요일 02:00 : Phase 1 (데이터 동기화)")
    print("   - 월요일 02:00 : Phase 2 (모델 학습)")
    print("   - 수요일 02:00 : Phase 3 (하이브리드 예측)")
    print("   - 목요일 02:00 : Phase 4 (성과 평가)")
    print("   - 금요일 02:00 : Phase 4+ (자율 진화)")

    # 타임존 설정: 대한민국 (KST)
    kst = pytz.timezone('Asia/Seoul')
    last_run_minute = -1

    while True:
        # 현재 한국 시간 확인
        now = datetime.now(kst)
        current_day_str = now.strftime("%A") # Sunday, Monday...
        current_hour = now.hour
        current_minute = now.minute

        # 1분 단위로 작업 체크 (중복 실행 방지)
        if current_minute != last_run_minute:

            # 1. 일요일 02:00 -> Phase 1 (Sync)
            if current_day_str == "Sunday" and current_hour == 2 and current_minute == 0:
                bot.job_sync()

            # 2. 월요일 02:00 -> Phase 2 (Train)
            elif current_day_str == "Monday" and current_hour == 2 and current_minute == 0:
                bot.job_train()

            # 3. 수요일 02:00 -> Phase 3 (Predict)
            elif current_day_str == "Wednesday" and current_hour == 2 and current_minute == 0:
                bot.job_predict()

            # 4. 목요일 02:00 -> Phase 4 (Evaluate)
            elif current_day_str == "Thursday" and current_hour == 2 and current_minute == 0:
                bot.job_evaluate()

            # 5. 금요일 02:00 -> Phase 4+ (Evolution)
            elif current_day_str == "Friday" and current_hour == 2 and current_minute == 0:
                bot.job_evolution()

            last_run_minute = current_minute

        # CPU 점유율을 낮추기 위해 10초 대기
        time.sleep(10)

if __name__ == "__main__":
    try:
        run_kst_schedule()
    except KeyboardInterrupt:
        print("\n🛑 스케줄러가 사용자에 의해 중단되었습니다.")
