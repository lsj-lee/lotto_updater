import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import datetime
import os

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
        
        # '줄스' 시트 열기 (구글 시트 제목: "로또_AI_자율주행_리포트")
        # 실제 시트 제목이 다르다면 아래 이름을 시트 제목과 똑같이 맞춰주세요.
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
    
    # 1. '추천번호' 탭 업데이트
    try:
        ws_nums = sheet.worksheet("추천번호")
        # 저장할 데이터 배열: [시간, 번호1, 번호2, 번호3, 번호4, 번호5, 번호6, 보너스, 조작의심지수]
        row_data = [now] + prediction_data + [f"{anomaly_score}%"]
        ws_nums.append_row(row_data)
        print(f"✅ [추천번호] 줄스에 기록 완료 ({now})")
    except Exception as e:
        print(f"⚠️ 추천번호 시트 기록 중 오류: {e}")

    # 2. '실행로그' 탭 업데이트
    try:
        ws_log = sheet.worksheet("실행로그")
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