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
    [지능형 스케줄러]
    - 정규 작전 수행 (일/월/화 02:00)
    - 실패 시 익일 02:00 자동 재시도 (Dynamic Retry)
    - 작전 종료 후 Mac 자동 잠자기 (Auto-Sleep)
    """
    def __init__(self):
        self.orchestrator = LottoOrchestrator()
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

    def run_sequence_with_retry(self, tasks):
        """
        연속 작업 실행 및 실패 시 재시도 예약
        tasks: [(task_name, func), ...]
        """
        all_success = True
        failed_task = None

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
            # 성공 시 재시도 큐 초기화 (혹시 남아있다면)
            self.retry_queue = []
        else:
            logging.warning(f"⚠️ [Mission Failed] '{failed_task[0]}' 실패. 내일 02:00 재시도 예약됨.")
            self.retry_queue.append(failed_task)

        # 작전 종료 후 잠자기 (성공하든 실패해서 재시도 예약하든 일단 시스템 종료)
        self.execute_auto_sleep()

    def retry_failed_tasks(self):
        """재시도 큐에 있는 작업 실행 (익일 02:00)"""
        if not self.retry_queue:
            return

        logging.info(f"🔄 [Retry] 재시도 작업 {len(self.retry_queue)}건 실행 시작...")
        # 큐 복사 후 비움 (실행 중 다시 실패하면 다시 추가됨)
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
            # 진화는 인터랙티브 작업이므로 자동화에서는 로그만 남기거나,
            # 백그라운드 분석만 수행하도록 변경 가능. 여기선 실행 시도.
            if sys.stdin.isatty():
                self.evolution_manager.execute_evolution_cycle('lotto_predict.py')
            else:
                logging.info("ℹ️ 백그라운드 모드: 진화 제안 생성만 시도합니다.")
                # (EvolutionManager에 비대화형 분석 메소드가 있다면 호출)
        else:
            logging.warning("⚠️ Evolution Manager 로드 실패")

# -----------------------------------------------------------------------------
# 🕒 KST (한국 시간) 기반 지능형 스케줄링 로직
# -----------------------------------------------------------------------------
def run_kst_schedule():
    bot = LottoScheduler()

    print("🚀 [Scheduler] Hybrid Sniper V5 지능형 스케줄러 (Auto-Sleep & Retry Enabled) 시작...")
    print("   - 일요일 02:00 (KST): [기초 공사] 데이터 수집 -> 모델 학습")
    print("   - 월요일 02:00 (KST): [실전 사격] 정예 번호 예측 -> 성과 평가")
    print("   - 화요일 02:00 (KST): [자가 진화] 코드 분석 및 개선 제안")
    print("   - 매일 02:00 (KST): 실패한 작전이 있다면 자동 재시도")

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

            # 0. 재시도 작업 우선 확인 (매일 02:00)
            if current_hour == 2 and current_minute == 0 and bot.retry_queue:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 재시도 작업 실행")
                bot.retry_failed_tasks()
                last_run_minute = current_minute
                continue # 재시도 실행했으면 정규 스케줄은 건너뜀 (중복 방지)

            # 1. 일요일 02:00 -> 기초 공사 (Sync + Train)
            if current_day_str == "Sunday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 기초 공사 시작")
                bot.run_sequence_with_retry([
                    ("Phase 1: 데이터 동기화", bot.job_sync),
                    ("Phase 2: 모델 학습", bot.job_train)
                ])

            # 2. 월요일 02:00 -> 실전 사격 (Predict + Evaluate)
            elif current_day_str == "Monday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 실전 사격 시작")
                bot.run_sequence_with_retry([
                    ("Phase 3: 정예 번호 예측", bot.job_predict),
                    ("Phase 4: 성과 평가", bot.job_evaluate)
                ])

            # 3. 화요일 02:00 -> 자가 진화 (Evolution)
            elif current_day_str == "Tuesday" and current_hour == 2 and current_minute == 0:
                logging.info(f"🕒 [Schedule] {current_day_str} 02:00 - 자가 진화 시작")
                bot.run_sequence_with_retry([
                    ("Phase 4+: 자율 진화", bot.job_evolution)
                ])

            last_run_minute = current_minute

        # CPU 점유율을 낮추기 위해 10초 대기
        time.sleep(10)

if __name__ == "__main__":
    try:
        run_kst_schedule()
    except KeyboardInterrupt:
        print("\n🛑 스케줄러가 사용자에 의해 중단되었습니다.")
