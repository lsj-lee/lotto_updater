# -*- coding: utf-8 -*-
import os
import sys
import datetime
import logging
from lotto_predict import LottoOrchestrator, SniperState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SniperCommander:
    """
    🎮 [수동 관제탑] 명령 즉시 작전을 1회 수행하며, 리셋 스위치를 지원합니다.
    """
    def __init__(self):
        logging.info("🚀 [관제탑] Sniper V7 수동 타격 모드 (리셋 스위치 탑재) 가동...")
        self.orc = LottoOrchestrator()
        self.state_manager = SniperState()
        self.sheet_name = "Remote_Control"

    def get_last_sunday(self):
        today = datetime.date.today()
        idx = (today.weekday() + 1) % 7 
        last_sunday = today - datetime.timedelta(days=idx)
        return last_sunday.strftime("%Y-%m-%d")

    def update_dashboard(self, phase_name, status, note=""):
        try:
            ws = self.orc.sheets.get_ws(self.sheet_name)
            row_map = {"Phase 1": 5, "Phase 2": 6, "Phase 3": 7, "Phase 4": 8}
            row_idx = row_map.get(phase_name.split(":")[0], 9)
            ws.update_cell(row_idx, 3, status)
            ws.update_cell(row_idx, 5, note)
        except: pass

    def run_now(self):
        logging.info("📡 [작전 개시] 사령관님의 직접 명령 수신.")
        try:
            target_sunday = self.get_last_sunday()
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            ws = self.orc.sheets.get_ws(self.sheet_name)
            
            is_reset = str(ws.cell(2, 4).value).upper() == 'TRUE'
            if is_reset:
                logging.info("🔄 [초기화 명령 감지] 과거의 작전 완료 기록을 모두 소각합니다.")
                phases = ["last_sync_date", "last_train_date", "last_predict_date", "last_eval_date"]
                for p in phases: self.state_manager.state[p] = "2000-01-01" 
                self.state_manager.update_metric("last_reset", today_str) 
                ws.update_cell(2, 4, False)
                logging.info("✅ 기록 소각 완료. 시트의 체크박스를 자동으로 껐습니다.")

            ws.update_cell(2, 3, "⏳ 수동 작전 수행 중...")
            
            # 🎯 핵심 개조: lotto_predict.py의 최신 함수명으로 링크를 완벽히 수정했습니다!
            all_tasks = [
                ("Phase 1: Sync", self.orc.run_sync, "last_sync_date", target_sunday),
                ("Phase 2: Train", self.orc.train_m5_engine, "last_train_date", today_str),
                ("Phase 3: Predict", self.orc.generate_hybrid_combinations, "last_predict_date", target_sunday),
                ("Phase 4: Eval", self.orc.evaluate_performance, "last_eval_date", target_sunday)
            ]

            tasks_executed = 0

            for name, func, state_key, target_date in all_tasks:
                if self.state_manager.state.get(state_key, "") >= target_date:
                    self.update_dashboard(name, "✅ 이미 완료됨 (스킵)")
                    continue

                self.update_dashboard(name, "⏳ 실행 중...")
                try:
                    func() 
                    self.state_manager.update_metric(state_key, today_str)
                    self.update_dashboard(name, "✅ 완료")
                    tasks_executed += 1
                except Exception as e:
                    logging.error(f"❌ {name} 실패: {e}")
                    self.update_dashboard(name, "❌ 오류 발생", str(e))
                    sys.exit(1)

            if tasks_executed == 0:
                logging.info("🏁 오늘은 더 이상 수행할 임무가 없습니다. (강제 실행은 시트 D2 체크)")
                ws.update_cell(2, 3, "✅ 잔여 임무 없음")
            else:
                logging.info(f"🏁 명령하신 {tasks_executed}개의 임무를 마쳤습니다.")
                ws.update_cell(2, 3, f"✅ 완료 ({today_str})")
            
        except Exception as e:
            logging.error(f"❌ 실행 중 치명적 오류: {e}")
            sys.exit(1)

if __name__ == "__main__":
    commander = SniperCommander()
    commander.run_now() 