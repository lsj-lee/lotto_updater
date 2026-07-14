# -*- coding: utf-8 -*-
import os
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

load_dotenv()

class SheetsHandler:
    """
    📦 [보급창] 구글 시트 통신 및 데이터 로드 전담 부대
    """
    def __init__(self):
        self.spreadsheet_id = os.getenv('SPREADSHEET_ID', '1l0ifE_xRUocAY_Av-P67uBMKOV1BAb4mMwg_wde_tyA')
        self.creds_file = 'creds_lotto.json'
        self.doc = self._connect()

    def _connect(self):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        if not os.path.exists(self.creds_file):
            raise FileNotFoundError(f"❌ [보급 실패] 최상위 폴더에 '{self.creds_file}' 파일이 없습니다.")
        creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, scope)
        return gspread.authorize(creds).open_by_key(self.spreadsheet_id)

    def get_ws(self, name_or_idx):
        if isinstance(name_or_idx, int):
            return self.doc.get_worksheet(name_or_idx)
        else:
            try:
                return self.doc.worksheet(name_or_idx)
            except gspread.exceptions.WorksheetNotFound:
                return self.doc.add_worksheet(title=name_or_idx, rows="100", cols="20")

    def get_history_data(self):
        """
        📊 통합 컨트롤러의 요청에 따라 과거 당첨 데이터를 로드합니다.
        쉼표(,)가 포함된 숫자 데이터도 완벽하게 숫자로 변환합니다.
        """
        ws = self.get_ws(0) 
        all_values = ws.get_all_values()
        
        if not all_values:
            raise ValueError("❌ [데이터 오류] 구글 시트에 데이터가 비어 있습니다.")

        df = pd.DataFrame(all_values[1:], columns=all_values[0])
        
        # [핵심 수정] 모든 셀 데이터에서 쉼표(,)를 제거하여 "1,221"을 "1221"로 만듭니다.
        df = df.replace(',', '', regex=True)
        
        # 최신 규격에 맞춰 숫자로 변환합니다.
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if '회차' in df.columns:
            # 회차를 정수(int)로 변환하여 정확한 순서로 정렬합니다.
            df['회차'] = df['회차'].astype(int)
            df = df.set_index('회차').sort_index()
            
        return df