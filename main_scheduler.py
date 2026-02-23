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

    # --- 개별 작업 정의 ---

    def job_sync(self):
        """Phase 1: 데이터 동기화"""
        self.run_safe("Data Synchronization", self.orchestrator.sync_data)

    def job_train(self):
        """Phase 2: 모델 학습 (데이터 로드 -> 학습 -> 가중치 저장)"""
        # train_brain()은 모델을 반환하지만 스케줄러에서는 저장만 하면 되므로 반환값 무시
        self.run_safe("Model Training (Phase 2)", self.orchestrator.train_brain)

    def job_predict(self):
        """Phase 3: 번호 예측 (학습 없이 가중치 로드 -> Top 20 -> 1만개 -> 50개 -> LLM)"""
        if hasattr(self.orchestrator, 'load_and_predict'):
            self.run_safe("Prediction Only (Phase 3)", self.orchestrator.load_and_predict)
        else:
            logging.error("❌ 'load_and_predict' 메소드가 없습니다. lotto_predict.py를 확인하세요.")

    def job_evaluate(self):
        """Phase 4: 성과 평가 (Reward System)"""
        self.run_safe("Performance Evaluation (Reward)", self.orchestrator.evaluate_performance)

    def job_evolution(self):
        """Phase 4+: 자율 진화 제안"""
        if self.evolution_manager:
            logging.info("🧬 [Self-Evolution] 코드 분석 및 진화 제안 시작...")
            # 터미널 상호작용이 필요하므로, 실제 자동화 시에는 로그만 남기거나
            # 별도의 알림을 보내는 방식으로 처리하는 것이 좋습니다.
            if sys.stdin.isatty():
                self.evolution_manager.execute_evolution_cycle('lotto_predict.py')
            else:
                logging.info("ℹ️ 백그라운드 실행 중입니다. 진화 제안은 수동으로 실행하세요.")
        else:
            logging.warning("⚠️ Evolution Manager가 로드되지 않았습니다.")

# -----------------------------------------------------------------------------
# 🕒 KST (한국 시간) 기반 스케줄링 로직
# -----------------------------------------------------------------------------
def run_kst_schedule():
    bot = LottoScheduler()

    # 작업 실행 상태를 추적하여 1분 동안 중복 실행 방지
    last_run_minute = -1

    print("🚀 [Scheduler] Hybrid Sniper V5 KST(한국 시간) 스케줄러 시작...")
    print("   - 일요일 02:00 : Phase 1 (데이터 동기화)")
    print("   - 월요일 02:00 : Phase 2 (모델 학습)")
    print("   - 수요일 02:00 : Phase 3 (Top 20 기반 번호 예측)")
    print("   - 목요일 02:00 : Phase 4 (성과 평가)")
    print("   - 금요일 02:00 : Phase 4+ (자율 진화)")

    # 타임존 설정: 대한민국 (KST)
    kst = pytz.timezone('Asia/Seoul')

    while True:
        # 현재 한국 시간 확인
        now = datetime.now(kst)
        current_day_str = now.strftime("%A") # Sunday, Monday...
        current_hour = now.hour
        current_minute = now.minute

        # 디버깅용 로그 (매시 정각에만 출력)
        # if current_minute == 0 and current_minute != last_run_minute:
        #     print(f"🕒 [Tick] 현재 한국 시간: {now.strftime('%Y-%m-%d %H:%M:%S')} ({current_day_str})")

        # 1분 단위 체크 (중복 실행 방지)
        if current_minute != last_run_minute:

            # 1. 일요일 02:00 -> Phase 1 (Sync)
            if current_day_str == "Sunday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 데이터 동기화 시작")
                bot.job_sync()

            # 2. 월요일 02:00 -> Phase 2 (Train)
            elif current_day_str == "Monday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 모델 학습 시작")
                bot.job_train()

            # 3. 수요일 02:00 -> Phase 3 (Predict)
            elif current_day_str == "Wednesday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 번호 예측 시작")
                bot.job_predict()

            # 4. 목요일 02:00 -> Phase 4 (Evaluate)
            elif current_day_str == "Thursday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 성과 평가 시작")
                bot.job_evaluate()

            # 5. 금요일 02:00 -> Phase 4+ (Evolution)
            elif current_day_str == "Friday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 자율 진화 제안 시작")
                bot.job_evolution()

            # 실행 완료 후 현재 분 기록
            last_run_minute = current_minute

        # CPU 점유율을 낮추기 위해 10초 대기
        time.sleep(10)

if __name__ == "__main__":
    try:
        run_kst_schedule()
    except KeyboardInterrupt:
        print("\n🛑 스케줄러가 사용자에 의해 중단되었습니다.")
