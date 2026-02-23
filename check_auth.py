import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import sys

CREDS_FILE = 'creds_lotto.json'

def diagnose_gspread():
    print("🔍 [Diagnosis] Checking Google Sheets Authentication...")

    # 1. Load Creds
    try:
        with open(CREDS_FILE, 'r') as f:
            creds_data = json.load(f)
            client_email = creds_data.get('client_email')
            print(f"✅ Loaded Creds File: {CREDS_FILE}")
            print(f"📧 Service Account Email: {client_email}")
            print("⚠️ ACTION REQUIRED: Share your '로또 max' sheet with this email!")
    except Exception as e:
        print(f"❌ Failed to read {CREDS_FILE}: {e}")
        return

    # 2. Authenticate
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
        gc = gspread.authorize(creds)
        print("✅ Authenticated with Google API.")
    except Exception as e:
        print(f"❌ Authentication Failed: {e}")
        return

    # 3. List Spreadsheets
    try:
        sheets = gc.openall()
        print(f"📚 Accessible Spreadsheets ({len(sheets)}):")
        for s in sheets:
            print(f"   - {s.title} (ID: {s.id})")

        if not sheets:
            print("❌ No sheets found! Did you share '로또 max' with the email above?")

    except Exception as e:
        print(f"❌ Failed to list sheets: {e}")

if __name__ == "__main__":
    diagnose_gspread()
