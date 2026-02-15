import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def update_latest_lotto():
    # 1. 구글 시트 연결
    key_path = "creds_lotto.json" # 깃허브 액션 환경에 맞춘 경로 (필요시 수정)
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
    client = gspread.authorize(creds)
    doc = client.open("로또 max")
    sheet1 = doc.worksheet("시트1")

    # 2. 마지막 회차 확인
    records = sheet1.get_all_values()
    last_draw_no = int(records[1][0]) # B열(회차) 확인
    next_draw_no = last_draw_no + 1

    # 3. 동행복권 API 호출 (당첨자 수 및 금액 포함)
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={next_draw_no}"
    response = requests.get(url).json()

    if response.get("returnValue") == "success":
        # 4. 데이터 파싱 및 양식 맞춤
        row_data = [
            next_draw_no,
            response["drwtNo1"], response["drwtNo2"], response["drwtNo3"],
            response["drwtNo4"], response["drwtNo5"], response["drwtNo6"],
            response["bnusNo"],
            f"{response['firstPrzwnerCo']}명",
            f"{response['firstWinamnt']:,}원"
        ]
        
        # 5. 시트 최상단(2번째 행)에 삽입
        sheet1.insert_row(row_data, 2)
        print(f"✅ {next_draw_no}회차 데이터 업데이트 완료: {row_data}")
    else:
        print(f"⚠️ {next_draw_no}회차 데이터가 아직 발표되지 않았습니다.")

if __name__ == "__main__":
    update_latest_lotto()
