# -*- coding: utf-8 -*-
import schedule
import time
import logging
import sys
import os
import torch
import gc
import pytz
import psutil
from datetime import datetime, timedelta

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
# 🧩 모듈 로딩
# -----------------------------------------------------------------------------
try:
    from lotto_predict import LottoOrchestrator, SniperState
    print("✅ 'lotto_predict.py' 모듈 로드 성공")
except ImportError:
    logging.error("❌ 'lotto_predict.py'를 찾을 수 없습니다.")
    sys.exit(1)

try:
    from evolution_manager import EvolutionManager
    print("✅ 'evolution_manager.py' 모듈 로드 성공")
except ImportError:
    logging.warning("⚠️ 'evolution_manager.py' 없음. 진화 기능 제한됨.")
    EvolutionManager = None

# -----------------------------------------------------------------------------
# 🛰️ 메인 스케줄러 클래스
# -----------------------------------------------------------------------------
class LottoScheduler:
    """
    [자율 기지 스케줄러]
    - 정규 작전 수행 (KST 02:00)
    - 자원 감시 (Resource Awareness)
    - 지능형 재시도 (Smart Retry)
    - 자동 잠자기 (Auto-Sleep)
    """
    def __init__(self):
        self.orchestrator = LottoOrchestrator()
        self.state_manager = SniperState()
        self.evolution_manager = EvolutionManager() if EvolutionManager else None
        self.retry_queue = []
        logging.info("🤖 Sniper V5 Scheduler Initialized.")

    def _cleanup_memory(self):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def check_resource_safety(self):
        """[시스템 감시] CPU 점유율이 80%를 초과하면 작전 중단"""
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 80:
            logging.warning(f"⚠️ [High Load] CPU {cpu_usage}% > 80%. M5 보호를 위해 작전 이월.")
            self.execute_auto_sleep()
            return False
        return True

    def execute_auto_sleep(self):
        """[자동 잠자기] 작전 종료 후 30초 유예 후 시스템 절전"""
        logging.info("🏁 작전 종료. 30초 후 잠자기 모드로 진입합니다.")
        time.sleep(30)
        try:
            os.system("osascript -e 'tell application \"System Events\" to sleep'")
        except Exception as e:
            logging.error(f"❌ 잠자기 실패: {e}")

    def run_safe(self, task_name, func, *args):
        logging.info(f"▶️ [작업 시작] {task_name}")
        try:
            self._cleanup_memory()
            func(*args)
            logging.info(f"✅ [작업 완료] {task_name}")
            return True
        except Exception as e:
            logging.error(f"❌ [작업 실패] {task_name}: {e}")
            return False

    def run_sequence_with_retry(self, tasks):
        """
        연속 작전 실행. 실패 시 다음 날 02:00 재시도 예약.
        tasks: [(name, func), ...]
        """
        # 1. 자원 점검
        if not self.check_resource_safety():
            return

        all_success = True
        failed_task = None

        for i, (name, func) in enumerate(tasks):
            success = self.run_safe(name, func)
            if not success:
                all_success = False
                failed_task = (name, func)
                break

            if i < len(tasks) - 1:
                time.sleep(10) # 쿨다운

        if all_success:
            logging.info("✨ 모든 작전 성공.")
            self.retry_queue = [] # 성공 시 재시도 큐 클리어
        else:
            logging.warning(f"⚠️ '{failed_task[0]}' 실패. 내일 02:00 재시도 예약.")
            self.retry_queue.append(failed_task)

        self.execute_auto_sleep()

    def retry_failed_tasks(self):
        """재시도 큐 실행"""
        if not self.retry_queue: return
        logging.info(f"🔄 재시도 작전 {len(self.retry_queue)}건 시작...")
        tasks = self.retry_queue[:]
        self.retry_queue = []
        self.run_sequence_with_retry(tasks)

    # --- Job Wrappers ---
    def job_sync(self): self.orchestrator.sync_data()
    def job_train(self): self.orchestrator.train_brain()
    def job_predict(self): self.orchestrator.load_and_predict()
    def job_evaluate(self): self.orchestrator.evaluate_performance()

    def job_evolution(self):
        if self.evolution_manager:
            if sys.stdin.isatty():
                self.evolution_manager.execute_evolution_cycle('lotto_predict.py', self.state_manager)
            else:
                # 백그라운드 모드에서는 프롬프트 진화만 수행 (코드 수정 X)
                # execute_evolution_cycle 내부에서 처리됨
                pass

# -----------------------------------------------------------------------------
# 🕒 KST 기반 메인 루프
# -----------------------------------------------------------------------------
def run_kst_schedule():
    bot = LottoScheduler()
    print("🚀 Sniper V5 자율 스케줄러 가동 (KST 02:00)")

    kst = pytz.timezone('Asia/Seoul')
    last_run_minute = -1

    while True:
        now = datetime.now(kst)
        if now.minute != last_run_minute:
            # 매일 02:00 정각
            if now.hour == 2 and now.minute == 0:
                day = now.strftime("%A")
                logging.info(f"🕒 [Schedule] {day} 02:00 작전 개시")

                # 1. 재시도 우선 처리
                if bot.retry_queue:
                    bot.retry_failed_tasks()
                    last_run_minute = now.minute
                    continue

                # 2. 요일별 정규 작전
                if day == "Sunday":
                    bot.run_sequence_with_retry([
                        ("Phase 1: Sync", bot.job_sync),
                        ("Phase 2: Train", bot.job_train)
                    ])
                elif day == "Monday":
                    bot.run_sequence_with_retry([
                        ("Phase 3: Predict", bot.job_predict),
                        ("Phase 4: Eval", bot.job_evaluate)
                    ])
                elif day == "Tuesday":
                    bot.run_sequence_with_retry([
                        ("Phase 4+: Evolution", bot.job_evolution)
                    ])
                else:
                    logging.info("💤 휴식일. 시스템 점검 후 절전.")
                    bot.execute_auto_sleep()

            last_run_minute = now.minute
        time.sleep(10)

if __name__ == "__main__":
    try:
        run_kst_schedule()
    except KeyboardInterrupt:
        print("\n🛑 시스템 종료.")
