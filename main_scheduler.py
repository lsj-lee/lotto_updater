# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import logging
import psutil
import datetime
from datetime import timedelta

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
# 🛰️ 단발성(Run-Once) 사령관 클래스
# -----------------------------------------------------------------------------
class SniperCommander:
    """
    [Hit & Run 사령관]
    - 매일 1회 실행 (crontab 연동)
    - 누락된 작전(Catch-up) 우선 수행
    - CPU 부하 감지 (Absolute Safety Mode)
    - 작전 수행 후 시스템 자동 잠자기
    """
    def __init__(self):
        self.orchestrator = LottoOrchestrator()
        self.state_manager = SniperState()
        self.evolution_manager = EvolutionManager() if EvolutionManager else None
        logging.info("🤖 Sniper V5 Commander Initialized (Run-Once Mode).")

    def _cleanup_memory(self):
        try:
            import gc
            import torch
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except: pass

    def check_resource_safety(self):
        """[시스템 감시] CPU 점유율이 60%를 초과하면 작전 중단"""
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 60:
            logging.warning(f"⚠️ M5 절대 안전 모드 발동: CPU 부하 {cpu_usage}% > 60%. 작전을 취소하고 종료합니다.")
            self.execute_auto_sleep()
            sys.exit(0)
        return True

    def execute_auto_sleep(self):
        """[자동 잠자기] 작전 종료 후 30초 유예 후 시스템 절전"""
        logging.info("🏁 작전 종료. 30초 후 맥북을 수면 상태로 전환합니다.")
        time.sleep(30)
        try:
            os.system("osascript -e 'tell application \"System Events\" to sleep'")
        except Exception as e:
            logging.error(f"❌ 잠자기 명령 실패: {e}")

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

    # --- Job Wrappers ---
    def job_sync(self): self.orchestrator.sync_data()
    def job_train(self): self.orchestrator.train_brain()
    def job_predict(self): self.orchestrator.load_and_predict()
    def job_evaluate(self): self.orchestrator.evaluate_performance()

    def job_evolution(self):
        if self.evolution_manager:
            # 단발성 실행에서는 자동 모드로 가정하거나, 로그만 남김
            # 여기서는 기존 로직 유지
            try:
                self.evolution_manager.execute_evolution_cycle('lotto_predict.py', self.state_manager)
            except Exception as e:
                logging.error(f"진화 실패: {e}")
        else:
            logging.info("진화 모듈 없음. 패스.")

    def get_tasks_for_day(self, day_name):
        """요일별 작전 정의 (기존 스케줄 유지)"""
        if day_name == "Sunday":
            return [("Phase 1: Sync", self.job_sync), ("Phase 2: Train", self.job_train)]
        elif day_name == "Monday":
            return [("Phase 3: Predict", self.job_predict), ("Phase 4: Eval", self.job_evaluate)]
        elif day_name == "Tuesday":
            return [("Phase 4+: Evolution", self.job_evolution)]
        else:
            return [] # Rest Days (Wed-Sat)

    def execute_mission(self):
        # 1. 자원 안전 점검
        self.check_resource_safety()

        # 2. 날짜 계산
        today = datetime.datetime.now().date()
        last_run_str = self.state_manager.state.get("last_scheduler_run", None)

        if last_run_str:
            last_run_date = datetime.datetime.strptime(last_run_str, "%Y-%m-%d").date()
        else:
            # 최초 실행 시 어제 실행한 것으로 간주하여 오늘 것만 수행
            last_run_date = today - timedelta(days=1)

        logging.info(f"📅 오늘: {today}, 마지막 실행: {last_run_date}")

        # 3. Catch-up 로직 (누락된 작전 수행)
        # last_run_date + 1 부터 today - 1 까지 확인
        target_date = last_run_date + timedelta(days=1)

        while target_date < today:
            day_name = target_date.strftime("%A")
            tasks = self.get_tasks_for_day(day_name)

            if tasks:
                logging.info(f"🚀 [Catch-up] 누락된 작전 수행: {target_date} ({day_name})")
                for task_name, task_func in tasks:
                    self.run_safe(task_name, task_func)

                # 상태 업데이트 및 종료 (1회 타격 원칙)
                self.state_manager.update_metric("last_scheduler_run", target_date.strftime("%Y-%m-%d"))
                logging.info(f"✨ Catch-up 완료 ({target_date}). 내일 계속됩니다.")
                self.execute_auto_sleep()
                sys.exit(0)

            # Rest Day인 경우 그냥 스킵하고 날짜만 업데이트?
            # 아니면 굳이 상태 업데이트 안하고 루프 계속?
            # 상태 업데이트를 해야 "확인했다"는 기록이 남음.
            logging.info(f"💤 [Pass] {target_date} ({day_name}) - 휴식일 (Skip)")
            # 휴식일이라도 상태는 업데이트하여 중복 체크 방지
            # 하지만 '1회 타격' 원칙상 '작전'이 없으면 루프를 돌아도 됨.
            # 다만, 너무 오래전 날짜에서 시작하면 무한루프 위험? -> while target_date < today 조건 있음.
            target_date += timedelta(days=1)

        # 4. 오늘 작전 수행 (Catch-up이 없었거나 모두 휴식일이었을 경우)
        day_name = today.strftime("%A")
        tasks = self.get_tasks_for_day(day_name)

        if tasks:
            logging.info(f"🚀 [Regular] 정규 작전 수행: {today} ({day_name})")
            for task_name, task_func in tasks:
                self.run_safe(task_name, task_func)
        else:
            logging.info(f"💤 [Rest] {today} ({day_name}) - 오늘은 작전이 없습니다.")

        # 5. 최종 상태 업데이트 및 종료
        self.state_manager.update_metric("last_scheduler_run", today.strftime("%Y-%m-%d"))
        self.execute_auto_sleep()

# -----------------------------------------------------------------------------
# 🚀 메인 실행부
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        commander = SniperCommander()
        commander.execute_mission()
    except KeyboardInterrupt:
        logging.warning("🛑 사용자 중단.")
    except Exception as e:
        logging.error(f"❌ 치명적 오류: {e}")
        # 오류 발생 시에도 잠자기는 시도 (안전)
        try:
            os.system("osascript -e 'tell application \"System Events\" to sleep'")
        except: pass
