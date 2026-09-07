# -*- coding: utf-8 -*-
import os
import io
import sys
import datetime
import time
import json
import re
from dotenv import load_dotenv
from google import genai 
from core.model_selector import SniperArmory

class TacticalReviewer:
    """
    🔍 [Phase 04] 예측 결과 피드백 및 자기 학습(Self-Correction) 데이터 생성
    - [NEW] 최상위 모델 통신 3회 실패 시 차순위 전환 없이 즉시 작전 중단
    """
    def __init__(self, sheets_handler):
        self.sheets = sheets_handler
        self.doc = self.sheets.doc
        
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("🚨 .env 파일에 GEMINI_API_KEY가 설정되지 않았습니다!")
        
        self.client = genai.Client(api_key=api_key)
        self.armory = SniperArmory()

    def execute_review(self):
        print("\n" + "="*60)
        print(" 🧠 [Phase 04] M5 예측 성과 분석 및 자가 학습 데이터 생성")
        print("="*60)
        
        try:
            history_df = self.sheets.get_history_data()
            if history_df.index.name == '회차':
                latest_actual_draw = int(history_df.index.max())
                actual_row = history_df.loc[latest_actual_draw]
            else:
                latest_actual_draw = int(history_df['회차'].max())
                actual_row = history_df[history_df['회차'] == latest_actual_draw].iloc[0]

            try:
                ws_review = self.doc.worksheet("오답노트")
                reviewed_draws = [int(r[0]) for r in ws_review.get_all_values()[1:] if str(r[0]).isdigit()]
                if latest_actual_draw in reviewed_draws:
                    print(f"   ⚠️ {latest_actual_draw}회차 분석 피드백이 이미 존재합니다. 기존 데이터를 삭제하고 [새로 계산하여 덮어쓰기]를 진행합니다.")
            except Exception:
                pass 

            ws_log = self.doc.worksheet("기록")
            all_logs = ws_log.get_all_values()
            
            target_pred_row = None
            pred_date = ""
            
            print(f"   📡 제 {latest_actual_draw}회차 실제 당첨 결과와 예측 기록을 대조합니다...")
            for row in all_logs[1:]:
                if len(row) < 3: continue
                if str(row[1]).strip() == str(latest_actual_draw):
                    target_pred_row = row
                    pred_date = row[0]
                    break

            if not target_pred_row:
                print(f"   ⚠️ {latest_actual_draw}회차에 해당하는 시스템 예측 기록이 존재하지 않습니다.")
                return

            pred_nums_str = target_pred_row[2].replace('"', '')
            pred_nums = [int(n.strip()) for n in pred_nums_str.split(',') if n.strip().isdigit()]
            
            actual_nums = [int(actual_row[f'{i}번']) for i in range(1, 7)]
            bonus_num = int(actual_row['보너스'])

            hits = set(pred_nums) & set(actual_nums)

            print(f"   🎯 분석 대상: 제 {latest_actual_draw}회차")
            print(f"   📊 적중 결과: {len(hits)}개 일치 {sorted(list(hits))}")
            print("   💬 AI 모델에 오답노트 및 [기계용 전술 지시서 JSON] 작성을 요청합니다...")

            prompt = f"""
            당신은 데이터를 기반으로 객관적인 피드백을 제공하는 데이터 분석가입니다.
            제 {latest_actual_draw}회차 예측 모델의 성과에 대한 '피드백 보고서'와 기계가 읽을 '가중치 JSON 데이터'를 작성하십시오.
            
            [비교 데이터]
            - 시스템 예측 15개 주요 번호: {sorted(pred_nums)}
            - 실제 당첨 번호: {actual_nums} (보너스: {bonus_num})
            - 적중한 번호: {sorted(list(hits))}
            
            [중요: 절대 준수 규칙]
            답변은 반드시 아래 (1)번과 (2)번 양식을 모두 포함해야 하며, 어떠한 경우에도 (2)번 JSON 마크다운 블록을 누락해서는 안 됩니다.

            (1) 사람을 위한 분석 리포트 (일반 텍스트)
            - 모델이 어떤 통계적 편향(구간 쏠림, 끝수 등)을 보였는지 원인을 진단하십시오.
            - 다음 회차({latest_actual_draw + 1}회차) 예측 시 보완할 전략을 서술하십시오.

            (2) 기계(M5 엔진)를 위한 전술 지시서 (반드시 마크다운 json 블록으로 작성)
            - 아래 JSON 키값을 절대 변경하지 말고, 분석 결과에 맞춰 값만 수정하십시오.
            ```json
            {{
                "strategy_mode": "trend_following",
                "zone_weights": {{
                    "zone_1": 1.0, 
                    "zone_2": 1.0, 
                    "zone_3": 1.0, 
                    "zone_4": 1.0, 
                    "zone_5": 1.0  
                }},
                "hot_last_digits": [], 
                "require_consecutive": false, 
                "carryover_weight": 1.0 
            }}
            ```
            """

            pipeline = self.armory.get_model_pipeline(target_tier="중급")
            # [NEW] 차순위 모델로 넘어가지 않고 대시보드 기준 최상위 단일 모델만 타격
            target_model = pipeline[0] if pipeline else "models/gemini-3.8-flash"
            max_retries = 3
            human_text = ""
            tactical_json = {}
            success = False

            print(f"   🔄 [최상위 모델 지정]: {target_model} (서버 부하 시 3회 재시도 후 작전 정지)")

            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(model=target_model, contents=prompt)
                    review_text = response.text
                    
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', review_text, re.DOTALL)
                    
                    if json_match:
                        tactical_json = json.loads(json_match.group(1))
                        human_text = re.sub(r'(?i)(?:---|)\n*###\s*\(?2\)?.*?```(?:json)?\s*\{.*?\}\s*```', '', review_text, flags=re.DOTALL).strip()
                        print(f"   🎯 [처리 완료] 모델({target_model}) 피드백 및 JSON 지시서 생성 성공.")
                        success = True
                        break
                    else:
                        raise ValueError("API 응답에 필수적인 JSON 마크다운 블록이 누락되었습니다.")
                        
                except Exception as e:
                    error_msg = str(e)
                    if any(err in error_msg for err in ["503", "UNAVAILABLE", "429", "quota", "누락"]):
                        if attempt < max_retries - 1:
                            wait_time = 5 * (attempt + 1)
                            print(f"      ⚠️ 서버 고부하/응답 불량 감지. {wait_time}초 대기 후 재시도... (시도 {attempt+1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            print(f"      🚨 {max_retries}회 재시도 실패. 최상위 모델({target_model}) 서버 부하 초과.")
                    else:
                        print(f"      🚨 알 수 없는 API 오류 ({target_model}): {error_msg}")
                        break

            # [NEW] 3회 재시도 후에도 실패한 경우, 차순위 전환 없이 즉시 프로세스 종료
            if not success:
                print("\n" + "="*65)
                print(f"🚨 [작전 중단] 최상위 모델({target_model}) 통신 3회 연속 실패로 인해 시스템을 안전하게 정지합니다.")
                print("   - 자동 차순위 전환을 차단하였습니다. 나중에 수동으로 다시 실행해 주십시오.")
                print("="*65)
                sys.exit(0)
            
            if human_text:
                print("\n" + "="*65)
                print(human_text)
                print("="*65)
                
                self._save_to_review_sheet(latest_actual_draw, human_text)
                self._save_to_local_json(latest_actual_draw, tactical_json)

        except Exception as e:
            print(f"   🚨 분석 프로세스 치명적 오류 발생: {e}")

    def _save_to_review_sheet(self, draw_no, review_text):
        try:
            try:
                ws_review = self.doc.worksheet("오답노트")
            except:
                ws_review = self.doc.add_worksheet(title="오답노트", rows="1000", cols="5")
                ws_review.append_row(["회차", "분석 일자", "성과 분석 피드백"])

            all_records = ws_review.get_all_values()
            for i, r in enumerate(all_records):
                if r and str(r[0]).strip() == str(draw_no):
                    ws_review.delete_rows(i + 1)
                    break

            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ws_review.insert_row([draw_no, now, review_text], 2)
            print("   ✅ [오답노트] 구글 시트에 성과 분석 리포트 기록(덮어쓰기) 완료.")
        except Exception as e:
            print(f"   ⚠️ 오답노트 시트 저장 실패: {e}")

    def _save_to_local_json(self, draw_no, tactical_json):
        try:
            memory_file = "m5_memory.json"
            target_draw = draw_no + 1
            data = {
                "target_draw": target_draw,
                "update_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tactical_directives": tactical_json
            }
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"   💾 [로컬 메모리] 제 {target_draw}회차 예측을 위한 M5 전술 지시서(JSON) 덮어쓰기 완료.")
        except Exception as e:
            print(f"   ⚠️ 로컬 JSON 메모리 저장 실패: {e}")