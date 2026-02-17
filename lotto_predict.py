import gspread
from google.oauth2.service_account import Credentials
import datetime
import time

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
def update_jules_report(prediction_data, anomaly_score):
    sheet = connect_jules()
    if not sheet: return

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # ------------------------------------------
    # (A) [기능 추가] AI 주간 분석 리포트 (새로 작성)
    # ------------------------------------------
    try:
        # 'AI_분석_리포트' 시트 가져오기 (없으면 생성)
        try:
            ws_report = sheet.worksheet("AI_분석_리포트")
            ws_report.clear() # 기존 내용 삭제 (최신 리포트 갱신)
        except:
            ws_report = sheet.add_worksheet(title="AI_분석_리포트", rows=100, cols=20)

        # 리포트 작성 데이터 준비
        # 1. 제목 (A1:G1 병합)
        ws_report.update_cell(1, 1, "[AI 9차원 앙상블] 주간 분석 리포트")
        ws_report.merge_cells('A1:G1')

        # 2. 분석 개요 (A3:G3 병합)
        ws_report.update_cell(3, 1, "1. 분석 개요")
        ws_report.merge_cells('A3:G3')
        ws_report.update_cell(4, 1, f"작성 일시: {now}")
        ws_report.update_cell(4, 3, "분석 모델: 9차원 LSTM 앙상블") # C열쯤에 배치

        # 3. AI 추천 번호 (A6:G6 병합)
        ws_report.update_cell(6, 1, "2. AI 추천 번호")
        ws_report.merge_cells('A6:G6')

        # 번호 입력 (A7~G7: 7개 숫자)
        # prediction_data는 [번호1, 번호2, ..., 번호6, 보너스] 형태라고 가정
        for i, num in enumerate(prediction_data):
            # A7(1,1) -> G7(1,7)
            ws_report.update_cell(7, i+1, num)

        # 4. 조작 의심 지수 (A9:G9 병합)
        ws_report.update_cell(9, 1, "3. 조작 의심 지수")
        ws_report.merge_cells('A9:G9')
        ws_report.update_cell(10, 1, f"{anomaly_score}%")

        # 5. 시스템 로그 (A12:G12 병합)
        ws_report.update_cell(12, 1, "4. 시스템 로그")
        ws_report.merge_cells('A12:G12')
        ws_report.update_cell(13, 1, "M5 9차원 앙상블 완료")
        ws_report.update_cell(13, 3, "자율 주행 성공")

        print(f"✅ [리포트] 'AI_분석_리포트' 작성 및 병합 완료 ({now})")

    except Exception as e:
        print(f"⚠️ 리포트 작성 중 오류: {e}")

    # ------------------------------------------
    # (B) 히스토리 로그 저장 (기존 '추천번호' 시트)
    # ------------------------------------------
    try:
        try:
            ws_nums = sheet.worksheet("추천번호")
        except:
            ws_nums = sheet.add_worksheet(title="추천번호", rows=1000, cols=20)
            # 헤더 추가
            ws_nums.append_row(["시간", "번호1", "번호2", "번호3", "번호4", "번호5", "번호6", "보너스", "조작의심지수"])

        # 저장할 데이터 배열: [시간, 번호1, 번호2, 번호3, 번호4, 번호5, 번호6, 보너스, 조작의심지수]
        row_data = [now] + prediction_data + [f"{anomaly_score}%"]
        ws_nums.append_row(row_data)
        print(f"✅ [히스토리] '추천번호'에 기록 완료")
    except Exception as e:
        print(f"⚠️ 추천번호 시트 기록 중 오류: {e}")

    # ------------------------------------------
    # (C) 실행로그 (선택 사항)
    # ------------------------------------------
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
    
    # 임시 테스트 데이터 (실제 실행 시에는 뇌의 연산값이 들어갑니다)
    test_numbers = [5, 14, 21, 30, 35, 42, 11] # 샘플 추천 번호
    test_anomaly = 12.8 # 샘플 조작 의심 지수
    
    update_jules_report(test_numbers, test_anomaly)
