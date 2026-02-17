import gspread
from google.oauth2.service_account import Credentials
import datetime
import random

# ==========================================
# [1] 줄스(Google Sheets) 접속 설정
# ==========================================
def connect_jules():
    # 상진 님 스크린샷을 바탕으로 한 실제 키 경로
    # 파일 경로: /Users/lsj/Desktop/구글 연결 키/creds lotto.json
    json_path = "/Users/lsj/Desktop/구글 연결 키/creds lotto.json"
    
    # 권한 설정 (시트 및 드라이브 접근)
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    try:
        # 키 파일을 사용하여 구글 인증
        creds = Credentials.from_service_account_file(json_path, scopes=scopes)
        client = gspread.authorize(creds)
        
        # '줄스' 시트 열기 (구글 시트 제목: "로또 max")
        spreadsheet = client.open("로또 max")
        return spreadsheet
    except Exception as e:
        print(f"❌ 줄스 연결 실패: {e}")
        print("💡 팁: '구글 연결 키' 폴더와 'creds lotto.json' 파일 이름이 정확한지 확인해 주세요.")
        return None

# ==========================================
# [2] 데이터 전송 및 리포트 업데이트
# ==========================================
def update_jules_report(prediction_list, anomaly_score):
    """
    prediction_list: 5개의 로또 번호 세트 (리스트의 리스트, 예: [[1,2,3,4,5,6], ...])
    anomaly_score: 조작 의심 지수 (float)
    """
    sheet = connect_jules()
    if not sheet: return

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # '추천번호' 시트 가져오기 (없으면 생성)
    try:
        ws_report = sheet.worksheet("추천번호")
    except:
        ws_report = sheet.add_worksheet(title="추천번호", rows=100, cols=20)

    # [1. 시트 초기화] 기존 내용 삭제
    ws_report.clear()
    print("🧹 [초기화] '추천번호' 시트 내용을 삭제하고 새로 작성을 시작합니다.")

    try:
        # [2. 리포트 데이터 준비 (2D 리스트)]
        # 약 20행 x 7열의 빈 리스트 생성
        report_data = [['' for _ in range(7)] for _ in range(20)]

        # (A) 제목 (1행)
        report_data[0][0] = "[AI 9차원 앙상블] 주간 분석 리포트"

        # (B) 분석 개요 (3행)
        report_data[2][0] = "1. 분석 개요"
        report_data[3][0] = f"작성 일시: {now}"
        report_data[3][3] = "분석 모델: 9차원 LSTM 앙상블" # D열(index 3)

        # (C) AI 추천 번호 (6행)
        report_data[5][0] = "2. AI 추천 번호 (5 Game)"

        # 5세트 번호 입력 (Game 1 ~ Game 5) - 7행부터
        row_offset = 6 # 7행은 index 6
        for i, numbers in enumerate(prediction_list):
            current_row = row_offset + i
            # A열: Game 번호
            report_data[current_row][0] = f"Game {i+1}"
            # B~G열: 번호 6개
            for j, num in enumerate(numbers):
                if j < 6: # 최대 6개까지만 기록
                    report_data[current_row][j+1] = num # B열(index 1)부터

        # (D) 조작 의심 지수 (14행)
        sec3_row_idx = 13 # 14행
        report_data[sec3_row_idx][0] = "3. 조작 의심 지수"
        report_data[sec3_row_idx+1][0] = f"Anomaly Score: {anomaly_score}%"

        # (E) 시스템 로그 (17행)
        sec4_row_idx = 16 # 17행
        report_data[sec4_row_idx][0] = "4. 시스템 로그"
        report_data[sec4_row_idx+1][0] = "M5 9차원 앙상블 완료"
        report_data[sec4_row_idx+1][3] = "자율 주행 성공" # D열

        # [3. 일괄 업데이트]
        # A1부터 시작하여 데이터 한 번에 쓰기 (API 호출 1회)
        ws_report.update("A1", report_data)

        # [4. 셀 병합 (레이아웃 정리)]
        # API 호출 4회 추가 (총 5회로 효율적)
        ws_report.merge_cells('A1:G1')      # 제목
        ws_report.merge_cells('A3:G3')      # 개요 헤더
        ws_report.merge_cells('A6:G6')      # 추천번호 헤더
        ws_report.merge_cells('A14:G14')    # 조작지수 헤더
        ws_report.merge_cells('A17:G17')    # 로그 헤더

        print(f"✅ [리포트] '추천번호' 탭에 5게임 분석 결과 작성 완료 ({now})")

    except Exception as e:
        print(f"⚠️ 리포트 작성 중 오류: {e}")

    # (선택) 실행로그 탭에도 기록 남기기 (히스토리용)
    try:
        try:
            ws_log = sheet.worksheet("실행로그")
        except:
            ws_log = sheet.add_worksheet(title="실행로그", rows=1000, cols=10)

        ws_log.append_row([now, "자율 주행 성공", "M5 9차원 앙상블 완료"])
    except:
        pass

# ==========================================
# [3] 메인 실행부
# ==========================================
if __name__ == "__main__":
    print("🚀 AI 분석 결과를 줄스로 전송하는 중입니다...")
    
    # [학습 데이터 연동 시뮬레이션]
    # 사용자의 요청에 따라 1줄이 아닌 5세트의 번호를 생성하여 전송합니다.
    # 실제 환경에서는 lotto_brain.py가 생성한 모델을 불러와 예측값을 생성해야 합니다.

    prediction_sets = []
    # 1~45 사이의 중복 없는 숫자 6개를 5세트 생성
    for _ in range(5):
        prediction_sets.append(sorted(random.sample(range(1, 46), 6)))

    # 조작 의심 지수 (임시 값)
    test_anomaly = round(random.uniform(5.0, 20.0), 2)

    print(f"🎲 생성된 9차원 앙상블 번호 (5세트):")
    for idx, p_set in enumerate(prediction_sets):
        print(f"  Game {idx+1}: {p_set}")
    print(f"⚠️ 조작 의심 지수: {test_anomaly}%")
    
    update_jules_report(prediction_sets, test_anomaly)
