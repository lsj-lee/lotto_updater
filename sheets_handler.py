# -*- coding: utf-8 -*-
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

load_dotenv()

class SheetsHandler:
    """
    📦 [보급창] 구글 시트 통신 및 기록 전담 부대
    """
    def __init__(self):
        self.spreadsheet_id = os.getenv('SPREADSHEET_ID', '1l0ifE_xRUocAY_Av-P67uBMKOV1BAb4mMwg_wde_tyA')
        self.creds_file = 'creds_lotto.json'
        self.doc = self._connect()

    def _connect(self):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, scope)
        return gspread.authorize(creds).open_by_key(self.spreadsheet_id)

    def get_ws(self, name_or_idx):
        """인덱스나 이름으로 탭을 가져오고 없으면 새로 생성"""
        if isinstance(name_or_idx, int):
            return self.doc.get_worksheet(name_or_idx)
        else:
            try:
                return self.doc.worksheet(name_or_idx)
            except gspread.exceptions.WorksheetNotFound:
                return self.doc.add_worksheet(title=name_or_idx, rows="100", cols="20")