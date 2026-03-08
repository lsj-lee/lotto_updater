# -*- coding: utf-8 -*-
import json
import re
import datetime
import time
from sheets_handler import SheetsHandler
from model_selector import SniperArmory

class EvolutionManager:
    """
    🧬 [진화 관제소 - 최종 완성형] 
    1. '추천번호' 탭의 복잡한 표(시나리오 1~10)를 완벽히 해독하여 모든 추천 번호를 수집합니다.
    2. 메타데이터(Epochs, LR, 프롬프트)와 함께 '작전기록' 탭에 대기 상태로 영구 기록합니다.
    3. 다음 작전 시, 과거의 미완성 기록을 찾아 실제 당첨 결과를 '자동으로 업데이트' 합니다.
    """
    def __init__(self):
        print("\n🧬 [진화 관제소] 전투 일지 보존 및 사후 자동 업데이트 시스템 가동...")
        self.sheets = SheetsHandler()
        self.armory = SniperArmory(self.sheets, auto_scout=False)
        
    def _setup_headers(self, ws):
        if not ws.get_all_values():
            headers = [
                "기록 일시", "대상 회차", "학습 횟수(Epochs)", "학습 속도(LR)", 
                "M5 후보수", "적용된 프롬프트", "군집(K-Means)", "추천 번호 총합", 
                "AI 요약/사유", "실제 당첨", "적중 수", "AI 반성문 및 다음 전술"
            ]
            ws.append_row(headers)
            print("   📝 [시스템] '작전기록' 탭에 12개의 정밀 데이터 헤더를 생성했습니다.")

    def run_evolution(self):
        try:
            # 1. 시트 데이터 확보
            pred_ws = self.sheets.get_ws("추천번호")
            real_ws = self.sheets.get_ws(0) # 시트1
            history_ws = self.sheets.get_ws("작전기록")
            remote_ws = self.sheets.get_ws("Remote_Control")
            
            self._setup_headers(history_ws)

            # ==========================================
            # 🔄 [고급 전술] 과거 기록 사후 당첨 자동 업데이트
            # ==========================================
            self._update_past_records(history_ws, real_ws)

            # ==========================================
            # 🎯 [신규 기록] 표 형식의 추천 번호 정밀 타격 및 수집
            # ==========================================
            print("   ⏳ '추천번호' 시트에서 표 형식의 데이터를 스캔합니다...")
            all_pred_data = []
            for attempt in range(1, 6):
                all_pred_data = pred_ws.get_all_values()
                if len(all_pred_data) >= 3:
                    print(f"   ✅ [스캔 성공] {attempt}번의 시도 끝에 전체 표 데이터를 확보했습니다.")
                    break
                print(f"   ⚠️ [수색 대기] 구글 서버 동기화 지연 중... ({attempt}/5회)")
                time.sleep(2)
            
            if not all_pred_data or len(all_pred_data) < 3:
                print("   ❌ 분석할 예측 데이터가 부족하여 기록을 보류합니다.")
                return

            # 데이터 파싱 (스크린샷 구조 완벽 대응)
            record_time = all_pred_data[1][0] if len(all_pred_data[1]) > 0 else datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            pred_nums_set = set()
            summary_text = ""
            
            for row in all_pred_data:
                if not row: continue
                row_str = " ".join(row)
                
                # 1. '시나리오'가 포함된 행에서 로또 번호만 남김없이 수집
                if "시나리오" in str(row[0]):
                    for cell in row[1:7]: # B열 ~ G열
                        nums = re.findall(r'\b(?:[1-9]|[1-3][0-9]|4[0-5])\b', str(cell))
                        for n in nums:
                            pred_nums_set.add(int(n))
                
                # 2. 하단에 적힌 '요약'이나 '사유' 텍스트 수집
                elif "요약" in str(row[0]) or "사유" in str(row[0]) or "타격" in str(row[0]):
                    summary_text += row_str.strip() + "\n"

            if not pred_nums_set:
                print("   ⚠️ 표 내부에서 '시나리오' 번호를 찾지 못했습니다. 기록을 중단합니다.")
                return
                
            pred_nums_list = sorted(list(pred_nums_set))
            
            # 메타데이터 추출 (Remote_Control)
            epochs = remote_ws.acell('H5').value or "알수없음"
            lr = remote_ws.acell('H6').value or "알수없음"
            m5_count = remote_ws.acell('H9').value or "알수없음"
            current_prompt = remote_ws.acell('H11').value or "기본 프롬프트"
            
            # 회차 계산 (시트1의 최신 회차 + 1 = 대상 회차)
            real_row = real_ws.row_values(2)
            real_ep = int(re.sub(r'[^0-9]', '', str(real_row[0]))) if real_row else 0
            target_ep = str(real_ep + 1)
            
            kmeans_cluster = "자동 분류됨"

            # ==========================================
            # 📝 데이터베이스 기록 (대기 상태로 저장)
            # ==========================================
            real_nums_display = "데이터 수집 전"
            hit_count = "대기 중"
            reflection = "미래 회차 추첨 대기 중"
            
            record = [
                record_time, target_ep, epochs, lr, m5_count, 
                current_prompt, kmeans_cluster, str(pred_nums_list), 
                summary_text.strip(), real_nums_display, 
                str(hit_count), reflection
            ]
            
            history_ws.append_row(record)
            print(f"   📝 [작전기록] {target_ep}회차 전투 일지가 영구 보존되었습니다. (당첨 확인 대기 중)")

        except Exception as e:
            print(f"❌ 진화 프로토콜 치명적 오류: {e}")

    def _update_past_records(self, history_ws, real_ws):
        """과거 기록 중 '데이터 수집 전'인 항목을 찾아 당첨 결과를 업데이트합니다."""
        print("   🔍 [사후 확인] 과거 작전의 실제 당첨 여부를 대조합니다...")
        try:
            history_data = history_ws.get_all_values()
            if len(history_data) <= 1: return
            
            # 실제 당첨 번호 데이터베이스 (회차 -> 당첨번호)
            real_all = real_ws.get_all_values()
            real_db = {}
            for r in real_all[1:]:
                if not r: continue
                ep = re.sub(r'[^0-9]', '', str(r[0]))
                if ep:
                    nums = [int(re.sub(r'[^0-9]', '', str(x))) for x in r[1:7] if re.sub(r'[^0-9]', '', str(x))]
                    if len(nums) == 6:
                        real_db[ep] = nums

            updates_made = False
            for idx, row in enumerate(history_data):
                if idx == 0: continue 
                if len(row) > 9 and "데이터 수집 전" in row[9]:
                    target_ep = str(row[1])
                    if target_ep in real_db:
                        real_nums = real_db[target_ep]
                        pred_nums = [int(x) for x in re.findall(r'\d+', row[7])]
                        hits = set(pred_nums) & set(real_nums)
                        
                        row_num = idx + 1
                        history_ws.update_cell(row_num, 10, str(real_nums)) # 실제 당첨
                        history_ws.update_cell(row_num, 11, str(len(hits))) # 적중 수
                        
                        print(f"   🧬 {target_ep}회차 당첨 데이터 확인! (적중: {len(hits)}개). AI 반성문을 요청합니다...")
                        reflection = self._generate_reflection(target_ep, real_nums, pred_nums, hits)
                        history_ws.update_cell(row_num, 12, reflection) # 반성문
                        
                        updates_made = True
            
            if updates_made:
                print("   ✅ 과거 작전 기록의 사후 업데이트 및 훈장 수여가 완료되었습니다.")
            else:
                print("   ✔️ 사후 업데이트가 필요한 과거 기록이 없습니다.")
        except Exception as e:
            print(f"   ⚠️ 사후 업데이트 중 오류 발생 (무시하고 진행): {e}")

    def _generate_reflection(self, target_ep, real_nums, pred_nums, hits):
        prompt = f"""
        당신은 로또 예측 시스템 'Sniper V7'입니다.
        [과거 작전({target_ep}회) 결과]
        - 실제 당첨: {real_nums}
        - 나의 추천: {pred_nums[:15]}...
        - 적중: {list(hits)} (총 {len(hits)}개)
        
        실패 원인(어떤 번호대를 놓쳤는지 등)을 2문장으로 분석하고, 
        다음 전술 개선 방향을 1문장으로 제시하세요. (JSON 형태 말고 그냥 텍스트로만 출력하세요)
        """
        pipeline = self.armory.get_model_pipeline("고급")
        for model_id in pipeline:
            try:
                res = self.armory.client.models.generate_content(model=model_id, contents=prompt)
                if res.text: return res.text.strip()
            except: continue
        return "사후 분석에 실패했습니다."

if __name__ == "__main__":
    EvolutionManager().run_evolution()