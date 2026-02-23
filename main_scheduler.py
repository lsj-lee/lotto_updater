# -*- coding: utf-8 -*-
import schedule
import time
import logging
import sys
import os
import torch
import gc
import pytz
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
# 🧩 모듈 로딩 (lotto_predict.py 및 evolution_manager.py)
# -----------------------------------------------------------------------------
try:
    from lotto_predict import LottoOrchestrator, SniperState, SystemMonitor
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
    [지능형 스케줄러]
    - 정규 작전 수행 (일/월/화 02:00)
    - 실패 시 익일 02:00 자동 재시도 (Dynamic Retry)
    - 누락된 작전 자동 감지 및 따라잡기 (Smart Catch-up)
    - 작전 종료 후 Mac 자동 잠자기 (Auto-Sleep)
    """
    def __init__(self):
        self.orchestrator = LottoOrchestrator()
        self.state_manager = SniperState()
        self.evolution_manager = EvolutionManager() if EvolutionManager else None
        self.retry_queue = [] # 재시도 작업 목록
        logging.info("🤖 Hybrid Sniper V5 OrchestratorInitialized.")

    def _cleanup_memory(self):
        """M5 메모리 누수 방지를 위한 강제 청소"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logging.info("🧹 [System] 메모리 정리 완료 (Garbage Collection)")

    def execute_auto_sleep(self):
        """작전 종료 후 시스템 보호를 위해 Mac 잠자기 모드 진입"""
        logging.info("🏁 모든 작전 종료. 시스템 보호를 위해 30초 후 잠자기 모드로 진입합니다.")
        time.sleep(30) # 안전 유예 시간
        try:
            # macOS 전용 잠자기 명령
            os.system("osascript -e 'tell application \"System Events\" to sleep'")
        except Exception as e:
            logging.error(f"❌ 잠자기 모드 진입 실패: {e}")

    def run_safe(self, task_name, func, *args):
        """단일 작업 안전 실행"""
        logging.info(f"▶️ [작업 시작] {task_name}")
        try:
            self._cleanup_memory()
            func(*args)
            logging.info(f"✅ [작업 완료] {task_name}")
            return True
        except Exception as e:
            logging.error(f"❌ [작업 실패] {task_name}: {str(e)}")
            return False

    def check_system_load(self):
        """시스템 부하 확인 (과열 방지)"""
        is_healthy, cpu, mem = SystemMonitor.check_health()
        if not is_healthy:
            logging.warning(f"⚠️ [System Alert] 과부하 감지 (CPU: {cpu}%, MEM: {mem}%). 작전을 이월합니다.")
            self.execute_auto_sleep() # 즉시 잠자기
            return False
        return True

    def run_sequence_with_retry(self, tasks):
        """
        연속 작업 실행 및 실패 시 재시도 예약
        tasks: [(task_name, func), ...]
        """
        all_success = True
        failed_task = None

        # 시스템 상태 점검
        if not self.check_system_load():
            return # 과부하로 중단

        for i, (name, func) in enumerate(tasks):
            success = self.run_safe(name, func)

            if not success:
                all_success = False
                failed_task = (name, func)
                break # 이후 작업 중단하고 재시도 예약

            # 마지막 작업이 아니면 휴식 및 정리
            if i < len(tasks) - 1:
                logging.info("💤 [System] 과열 방지를 위해 10초간 대기합니다...")
                time.sleep(10)
                self._cleanup_memory()

        if all_success:
            logging.info("✨ [Mission Complete] 모든 작전이 성공적으로 완료되었습니다.")
            self.retry_queue = []
        else:
            logging.warning(f"⚠️ [Mission Failed] '{failed_task[0]}' 실패. 내일 02:00 재시도 예약됨.")
            self.retry_queue.append(failed_task)

        # 작전 종료 후 잠자기
        self.execute_auto_sleep()

    def smart_catch_up(self):
        """
        [지능형 작전 이어서 하기]
        누락된 이전 단계가 있다면 현재 단계 실행 전에 수행합니다.
        """
        state = self.state_manager.load_state()
        today_str = datetime.now().strftime("%Y-%m-%d")

        # 1. 동기화 (Phase 1) 누락 확인 (최근 3일 내 기록 없음)
        if not state.get('last_sync_date') or (datetime.now() - datetime.strptime(state['last_sync_date'], "%Y-%m-%d")).days > 3:
            logging.info("🔄 [Catch-up] 누락된 데이터 동기화 수행 중...")
            self.run_safe("Phase 1: Sync (Catch-up)", self.job_sync)
            time.sleep(5)

        # 2. 학습 (Phase 2) 누락 확인 (Sync보다 오래됨)
        last_train = state.get('last_train_date')
        if not last_train or last_train < state.get('last_sync_date', ''):
             logging.info("🧠 [Catch-up] 누락된 모델 학습 수행 중...")
             self.run_safe("Phase 2: Train (Catch-up)", self.job_train)
             time.sleep(5)

    def retry_failed_tasks(self):
        """재시도 큐에 있는 작업 실행 (익일 02:00)"""
        if not self.retry_queue:
            return

        logging.info(f"🔄 [Retry] 재시도 작업 {len(self.retry_queue)}건 실행 시작...")
        tasks_to_retry = self.retry_queue[:]
        self.retry_queue = []

        self.run_sequence_with_retry(tasks_to_retry)

    # --- 개별 작업 정의 ---

    def job_sync(self):
        self.orchestrator.sync_data()

    def job_train(self):
        self.orchestrator.train_brain()

    def job_predict(self):
        if hasattr(self.orchestrator, 'load_and_predict'):
            self.orchestrator.load_and_predict()
        else:
            raise AttributeError("'load_and_predict' 함수가 없습니다.")

    def job_evaluate(self):
        self.orchestrator.evaluate_performance()

    def job_evolution(self):
        if self.evolution_manager:
            if sys.stdin.isatty():
                result = self.evolution_manager.execute_evolution_cycle('lotto_predict.py')
            else:
                logging.info("ℹ️ 백그라운드 모드: 진화 제안 생성만 시도합니다.")
                # 실제로는 제안 생성 로직을 호출해야 함
                result = {"success": False, "detail": "Background mode"}

            # 진화 결과 기록
            self.orchestrator.log_operation("Phase 4+: Evolution",
                                            "SUCCESS" if result.get("success") else "SKIP",
                                            result.get("detail", ""))
        else:
            logging.warning("⚠️ Evolution Manager 로드 실패")

# -----------------------------------------------------------------------------
# 🕒 KST (한국 시간) 기반 지능형 스케줄링 로직
# -----------------------------------------------------------------------------
def run_kst_schedule():
    bot = LottoScheduler()

    print("🚀 [Scheduler] Hybrid Sniper V5 지능형 스케줄러 (Smart Catch-up Enabled) 시작...")
    print("   - 매일 02:00 (KST): 누락된 작전 확인 및 수행 (Catch-up)")
    print("   - 일요일 02:00 (KST): [기초 공사] 데이터 수집 -> 모델 학습")
    print("   - 월요일 02:00 (KST): [실전 사격] 정예 번호 예측 -> 성과 평가")
    print("   - 화요일 02:00 (KST): [자가 진화] 코드 분석 및 개선 제안")

    kst = pytz.timezone('Asia/Seoul')
    last_run_minute = -1

    while True:
        now = datetime.now(kst)
        current_day_str = now.strftime("%A")
        current_hour = now.hour
        current_minute = now.minute

        if current_minute != last_run_minute:

            # 02:00 정각 스케줄 시작
            if current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 작전 개시")

                # 0. 시스템 상태 점검
                if not bot.check_system_load():
                    last_run_minute = current_minute
                    continue

                # 1. 누락 작전 수행 (Catch-up)
                bot.smart_catch_up()

                # 2. 재시도 작업 수행
                if bot.retry_queue:
                    bot.retry_failed_tasks()
                    last_run_minute = current_minute
                    continue

                # 3. 요일별 정규 작전
                if current_day_str == "Sunday":
                    logging.info("📅 [Sunday Mission] 기초 공사")
                    bot.run_sequence_with_retry([
                        ("Phase 1: 데이터 동기화", bot.job_sync),
                        ("Phase 2: 모델 학습", bot.job_train)
                    ])

                elif current_day_str == "Monday":
                    logging.info("📅 [Monday Mission] 실전 사격")
                    bot.run_sequence_with_retry([
                        ("Phase 3: 정예 번호 예측", bot.job_predict),
                        ("Phase 4: 성과 평가", bot.job_evaluate)
                    ])

                elif current_day_str == "Tuesday":
                    logging.info("📅 [Tuesday Mission] 자가 진화")
                    bot.run_sequence_with_retry([
                        ("Phase 4+: 자율 진화", bot.job_evolution)
                    ])

                else:
                    logging.info("💤 [Rest Day] 오늘은 휴식일입니다. 시스템 점검 후 잠자기 모드로 진입합니다.")
                    bot.execute_auto_sleep()

            last_run_minute = current_minute

        time.sleep(10)

if __name__ == "__main__":
    try:
        run_kst_schedule()
    except KeyboardInterrupt:
        print("\n🛑 스케줄러가 사용자에 의해 중단되었습니다.")
