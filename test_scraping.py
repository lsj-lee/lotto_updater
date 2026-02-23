import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.naver.com/"
}

def get_latest_round():
    url = "https://search.naver.com/search.naver?query=로또"
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Method 1: Find text like '1211회차'
        text = soup.get_text()
        match = re.search(r'(\d+)회차', text)
        if match:
            print(f"✅ Found Latest Round (Text Regex): {match.group(1)}")
            return int(match.group(1))

        # Method 2: Specific class (This changes often)
        # Often inside <a class="_lotto-btn-current"> or similar
        element = soup.select_one('a._lotto-btn-current')
        if element:
            txt = element.get_text()
            num = re.sub(r'[^0-9]', '', txt)
            print(f"✅ Found Latest Round (Selector): {num}")
            return int(num)

        print("❌ Could not find latest round number.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def get_round_numbers(round_no):
    url = f"https://search.naver.com/search.naver?query=로또+{round_no}회+당첨번호"
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()

        # We need 6 numbers + 1 bonus
        # Pattern usually: "당첨번호 ... 1 2 3 4 5 6 ... 보너스 7"
        # Or structured div

        print(f"Searching for Round {round_no}...")

        # Attempt to find specific numbers via regex on text content
        # Looking for sequence of 6 numbers
        # Naver usually displays them in a row

        # Let's dump some text to see structure if needed
        # print(text[:500])

        # Simplistic regex for "X회차 ... date ... numbers"
        # Not reliable for extraction, but good for existence check
        if f"{round_no}회차" in text:
            print("✅ Round Exists in Text")
        else:
            print("⚠️ Round Number not found in text (might be hidden or structure changed)")

        # In real code, I will use Gemini to parse this text,
        # but for verification I want to see if I can get the numbers via regex/soup

        # Try finding the number balls
        balls = soup.select('span.ball, div.ball, span.win_num, div.win_num') # Common classes
        # Naver search specific classes: .num_box .num

        # Let's try a very generic scraping approach for numbers
        # The structure is often: <div class="num_box"> <span class="num">1</span> ... </div>
        pass

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    latest = get_latest_round()
    if latest:
        get_round_numbers(latest)
