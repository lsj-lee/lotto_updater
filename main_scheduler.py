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
    [압축된 스케줄러]
    일요일: 데이터 수집 -> 모델 학습 (연속 실행)
    월요일: 번호 예측 -> 성과 평가 (연속 실행)
    화요일: 자율 진화 (단독 실행)
    """
    def __init__(self):
        self.orchestrator = LottoOrchestrator()
        self.evolution_manager = EvolutionManager() if EvolutionManager else None
        logging.info("🤖 Hybrid Sniper V5 OrchestratorInitialized.")

    def _cleanup_memory(self):
        """M5 메모리 누수 방지를 위한 강제 청소"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logging.info("🧹 [System] 메모리 정리 완료 (Garbage Collection)")

    def run_safe(self, task_name, func, *args):
        """작업 안전 실행 래퍼"""
        logging.info(f"▶️ [작업 시작] {task_name}")
        try:
            self._cleanup_memory()
            func(*args)
            logging.info(f"✅ [작업 완료] {task_name}")
        except Exception as e:
            logging.error(f"❌ [작업 실패] {task_name}: {str(e)}")

    def run_sequence(self, tasks):
        """
        여러 작업을 연속해서 실행하며, 사이사이에 안전 휴식(Sleep)과 메모리 정리를 수행합니다.
        tasks: [(task_name, func), (task_name, func), ...]
        """
        for i, (name, func) in enumerate(tasks):
            self.run_safe(name, func)

            # 마지막 작업이 아니면 휴식 및 정리
            if i < len(tasks) - 1:
                logging.info("💤 [System] 과열 방지를 위해 10초간 대기합니다...")
                time.sleep(10)
                self._cleanup_memory()

    # --- 개별 작업 정의 ---

    def job_sync(self):
        self.orchestrator.sync_data()

    def job_train(self):
        self.orchestrator.train_brain()

    def job_predict(self):
        if hasattr(self.orchestrator, 'load_and_predict'):
            self.orchestrator.load_and_predict()
        else:
            logging.error("❌ 'load_and_predict' 함수가 없습니다.")

    def job_evaluate(self):
        self.orchestrator.evaluate_performance()

    def job_evolution(self):
        if self.evolution_manager:
            logging.info("🧬 [Self-Evolution] 코드 분석 및 진화 제안 시작...")
            if sys.stdin.isatty():
                self.evolution_manager.execute_evolution_cycle('lotto_predict.py')
            else:
                logging.info("ℹ️ 백그라운드 모드입니다. 진화 제안은 수동으로 실행하세요.")
        else:
            logging.warning("⚠️ Evolution Manager가 로드되지 않았습니다.")

# -----------------------------------------------------------------------------
# 🕒 KST (한국 시간) 기반 압축 스케줄링 로직
# -----------------------------------------------------------------------------
def run_kst_schedule():
    bot = LottoScheduler()

    print("🚀 [Scheduler] Hybrid Sniper V5 압축 스케줄러 (High-Speed Mode) 시작...")
    print("   - 일요일 02:00 (KST): [기초 공사] 데이터 수집 -> (10초 휴식) -> 모델 학습")
    print("   - 월요일 02:00 (KST): [실전 사격] 정예 번호 예측 -> (10초 휴식) -> 성과 평가")
    print("   - 화요일 02:00 (KST): [자가 진화] 코드 분석 및 개선 제안")
    print("   (이후 수~토요일은 휴식하며 다음 작전을 준비합니다)")

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

            # 1. 일요일 02:00 -> 기초 공사 (Sync + Train)
            if current_day_str == "Sunday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 기초 공사 시작")
                bot.run_sequence([
                    ("Phase 1: 데이터 동기화", bot.job_sync),
                    ("Phase 2: 모델 학습", bot.job_train)
                ])

            # 2. 월요일 02:00 -> 실전 사격 (Predict + Evaluate)
            elif current_day_str == "Monday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 실전 사격 시작")
                bot.run_sequence([
                    ("Phase 3: 정예 번호 예측", bot.job_predict),
                    ("Phase 4: 성과 평가", bot.job_evaluate)
                ])

            # 3. 화요일 02:00 -> 자가 진화 (Evolution)
            elif current_day_str == "Tuesday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 자가 진화 시작")
                bot.run_safe("Phase 4+: 자율 진화", bot.job_evolution)

            last_run_minute = current_minute

        # CPU 점유율을 낮추기 위해 10초 대기
        time.sleep(10)

if __name__ == "__main__":
    try:
        run_kst_schedule()
    except KeyboardInterrupt:
        print("\n🛑 스케줄러가 사용자에 의해 중단되었습니다.")
