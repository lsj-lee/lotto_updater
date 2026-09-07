# -*- coding: utf-8 -*-
import sys
import subprocess
from core.sheets_handler import SheetsHandler

# 시스템의 4가지 핵심 모듈을 로드합니다.
from phase_01.sync_engine import SyncEngine 
from phase_02.m5_ultimate import M5UltimateEngine
from phase_03.gemini_tactician import GeminiTactician
from phase_04.tactical_review import TacticalReviewer

# [NEW] XAI 투시경 모듈을 메인 시스템에 통합
from xai_mind_reader import run_mind_reader

def sync_armory():
    print("\n📡 [무기고 동기화] 깃허브 원격 저장소에서 최신 무기고(Armory) 명단을 수신합니다...")
    try:
        result = subprocess.run(["git", "pull"], capture_output=True, text=True)
        if "Already up to date." in result.stdout or "최신 상태입니다." in result.stdout:
            print("   ✅ 무기고가 이미 최신 상태입니다.")
        else:
            print("   📥 갱신 완료:\n" + result.stdout.strip())
    except Exception as e:
        print(f"   ⚠️ Git 동기화 실패 (시스템에 git이 설치되어 있는지 확인하십시오): {e}")

def main():
    print("="*60)
    print(" 📊 [M5 System] 로또 AI 예측 통합 컨트롤러")
    print("="*60)
    
    print("\n[ 시스템 메뉴를 선택하십시오 ]")
    print("  1. Phase 01 : 최신 당첨 데이터 동기화 (Vision AI)")
    print("  2. Phase 02 : M5 예측 모델 단독 가동 (AI 분석)")
    print("  3. Phase 03 : AI 분석 리포트 생성")
    print("  4. Phase 04 : 과거 예측 결과 복기 및 피드백 생성")
    print("  5. ALL      : 전체 페이즈 자동 연속 실행")
    print("  6. XAI      : 기계 심리 분석 가동 (Alpha & Beta 뇌파 스캔)")
    print("  7. SYNC     : 무기고 강제 동기화 (Git Pull)")
    print("  0. 시스템 종료")
    print("-" * 60)
    
    choice = input("명령 입력 (0~7): ").strip()
    
    if choice == '0':
        print("\n시스템을 정상 종료합니다.")
        sys.exit(0)
        
    elif choice == '7':
        sync_armory()
        
    elif choice == '1':
        print("\n[진행] Phase 01: 시각 지능(Vision AI) 데이터 동기화를 시작합니다...")
        sheets = SheetsHandler()
        sync = SyncEngine(armory=None, sheets=sheets)
        sync.run()
        
    elif choice == '2':
        print("\n[진행] Phase 02: M5 예측 모델 구동을 시작합니다...")
        sheets = SheetsHandler()
        engine = M5UltimateEngine(sheets)
        final_sets, hot_nums, stats_log = engine.execute_strike()
        
        print("\n[결과] 모델 구동 요약")
        print(f"  - 추출된 15개 핫존 (Hot Zones): {hot_nums}")
        print("\n[최종 생성된 10대 예측 매트릭스]")
        for i, lotto_set in enumerate(final_sets, 1):
            print(f"  [{i:02d} 세트] : {lotto_set}")
        print("\n✅ Phase 02 예측 모델 실행이 완료되었습니다.")
        
    elif choice == '3':
        print("\n[진행] Phase 03: AI 분석 리포트 생성을 시작합니다...")
        sheets = SheetsHandler()
        tactician = GeminiTactician(sheets)
        final_sets, hot_nums, m5_data = tactician.call_m5_and_get_results()
        briefing_text = tactician.request_tactical_briefing(m5_data)
        tactician.save_to_spreadsheet(final_sets, hot_nums, briefing_text)
        print("\n✅ Phase 03 분석 리포트 생성 및 기록이 완료되었습니다.")
        
    elif choice == '4':
        print("\n[진행] Phase 04: 예측 결과 복기 및 피드백 분석을 시작합니다...")
        sheets = SheetsHandler()
        reviewer = TacticalReviewer(sheets)
        reviewer.execute_review()
        print("\n✅ Phase 04 결과 복기 및 분석이 완료되었습니다.")
        
    elif choice == '6':
        print("\n[진행] XAI 기계 심리 분석 스캐너를 단독 가동합니다...")
        run_mind_reader()
        print("\n✅ XAI 스캔이 완료되었습니다.")

    elif choice == '5':
        print("\n[진행] 전체 페이즈(ALL) 자동 연속 실행을 개시합니다.")
        
        # [NEW] 작전 개시 전 무기고 강제 동기화 수행
        sync_armory()
        
        sheets = SheetsHandler()
        
        # Phase 01 (시각 지능 동기화)
        print("\n>> Phase 01 실행 중...")
        sync = SyncEngine(armory=None, sheets=sheets)
        sync.run()
        
        # Phase 02 & 03 (M5 예측 및 분석 리포트 생성 동시 진행)
        print("\n>> Phase 02 & 03 실행 중 (M5 예측 및 분석 리포트 생성)...")
        tactician = GeminiTactician(sheets)
        final_sets, hot_nums, m5_data = tactician.call_m5_and_get_results()
        briefing_text = tactician.request_tactical_briefing(m5_data)
        tactician.save_to_spreadsheet(final_sets, hot_nums, briefing_text)
        
        # Phase 04 (전술 복기)
        print("\n>> Phase 04 실행 중...")
        reviewer = TacticalReviewer(sheets)
        reviewer.execute_review()
        
        # [NEW] 작전 최종 단계: XAI 스캐너 자동 개입
        print("\n>> [XAI 개입] 모든 프로세스 완료. 기계 심리 분석 스캐너를 가동합니다...")
        run_mind_reader()
        
        print("\n✅ 모든 프로세스 및 XAI 스캔이 성공적으로 종료되었습니다.")
        
    else:
        print("❌ 잘못된 입력입니다. 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()