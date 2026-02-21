import requests
import gspread
import traceback
import re
from bs4 import BeautifulSoup
from oauth2client.service_account import ServiceAccountCredentials

def get_lotto_from_naver(drwNo):
    """
    네이버 검색 결과를 통해 로또 당첨 번호를 크롤링합니다.
    동행복권 API가 차단되었을 때 사용합니다.
    """
    try:
        url = f'https://search.naver.com/search.naver?query=로또+{drwNo}회'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,ko;q=0.8'
        }

        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')

        # 번호 추출
        numbers = [span.text for span in soup.select('.winning_number .ball')]
        if not numbers or len(numbers) < 6:
            # 아직 결과가 없는 경우
            return None

        bonus = soup.select_one('.bonus_number .ball').text

        # 당첨금 및 당첨자 수 추출
        win_text_elem = soup.select_one('.win_text')
        if not win_text_elem:
            return None

        win_text = win_text_elem.text

        # 당첨자 수 추출 (예: "당첨 복권수 13개")
        winners_match = re.search(r'\(당첨 복권수 (\d+)개\)', win_text)
        winners = int(winners_match.group(1)) if winners_match else 0

        # 당첨금 추출 (예: "1등 당첨금 2,207,575,472원")
        prize_match = re.search(r'([0-9,]+)원', win_text)
        prize = int(prize_match.group(1).replace(',', '')) if prize_match else 0

        return {
            "drwNo": drwNo,
            "drwtNo1": int(numbers[0]),
            "drwtNo2": int(numbers[1]),
            "drwtNo3": int(numbers[2]),
            "drwtNo4": int(numbers[3]),
            "drwtNo5": int(numbers[4]),
            "drwtNo6": int(numbers[5]),
            "bnusNo": int(bonus),
            "firstPrzwnerCo": winners,
            "firstWinamnt": prize,
            "returnValue": "success"
        }
    except Exception as e:
        print(f"⚠️ 네이버 크롤링 실패: {e}")
        # traceback.print_exc()
        return None

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
    last_draw_no = int(str(records[1][0]).replace(',', '')) # B열(회차) 확인
    next_draw_no = last_draw_no + 1

    # 3. 동행복권 API 호출 (당첨자 수 및 금액 포함)
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={next_draw_no}"
    response = None

    try:
        # User-Agent 추가하여 차단 우회 시도
        api_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        api_resp = requests.get(url, headers=api_headers, timeout=10)

        # HTML 응답이 오거나 리다이렉트 되는 경우 JSON 파싱 에러 발생 가능
        response = api_resp.json()
        print(f"ℹ️ 동행복권 API 호출 성공")

    except (requests.RequestException, ValueError, Exception) as e:
        print(f"⚠️ 동행복권 API 호출 실패/차단됨 ({e}). 네이버 크롤링으로 전환합니다.")
        response = get_lotto_from_naver(next_draw_no)

    if response and response.get("returnValue") == "success":
        # 4. 데이터 파싱 및 양식 맞춤
        # firstWinamnt 등이 int일 수도, 문자열일 수도 있음 -> 안전하게 int 변환 후 포맷팅
        try:
            win_amnt = int(str(response['firstWinamnt']).replace(',', ''))
            row_data = [
                next_draw_no,
                response["drwtNo1"], response["drwtNo2"], response["drwtNo3"],
                response["drwtNo4"], response["drwtNo5"], response["drwtNo6"],
                response["bnusNo"],
                f"{response['firstPrzwnerCo']}명",
                f"{win_amnt:,}원"
            ]

            # 5. 시트 최상단(2번째 행)에 삽입
            sheet1.insert_row(row_data, 2)
            print(f"✅ {next_draw_no}회차 데이터 업데이트 완료: {row_data}")
        except Exception as parse_error:
             print(f"❌ 데이터 파싱 오류: {parse_error}")
    else:
        print(f"⚠️ {next_draw_no}회차 데이터가 아직 발표되지 않았거나 가져올 수 없습니다.")

if __name__ == "__main__":
    update_latest_lotto()
