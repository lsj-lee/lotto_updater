# -*- coding: utf-8 -*-
import os
import sys
import difflib
import ast
import datetime
from dotenv import load_dotenv

# 필수 라이브러리: google-genai (v1.0+)
try:
    from google import genai
except ImportError:
    print("❌ 'google-genai' 라이브러리가 필요합니다. (pip install google-genai)")
    sys.exit(1)

load_dotenv()

class EvolutionManager:
    """
    🧬 [Phase 4] 자율 진화 관리자 (Self-Evolution Manager)
    - 기존 코드를 분석하여 개선점을 제안받고,
    - 사용자의 승인 하에 안전하게 코드를 업데이트합니다.
    - 진화 결과를 반환하여 시스템 상태(Feedback)에 반영합니다.
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("⚠️ GEMINI_API_KEY가 설정되지 않았습니다. 진화 기능이 제한됩니다.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("🧬 [Evolution] Gemini AI 연결 성공.")
            except Exception as e:
                print(f"❌ Gemini 클라이언트 초기화 실패: {e}")
                self.client = None

    def analyze_code(self, file_path='lotto_predict.py'):
        """소스 코드를 읽고 Gemini에게 개선 제안을 요청합니다."""
        if not self.client:
            print("❌ AI 모델이 연결되지 않아 분석할 수 없습니다.")
            return None

        print(f"🔍 [Evolution] {file_path} 분석 중...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_code = f.read()
        except FileNotFoundError:
            print(f"❌ 파일({file_path})을 찾을 수 없습니다.")
            return None

        prompt = f"""
        당신은 파이썬 전문가이자 로또 분석 시스템의 설계자입니다.
        아래 코드는 현재 작동 중인 시스템입니다.
        이 코드의 성능, 가독성, 또는 안정성을 개선할 수 있는 구체적인 수정안을 1가지만 제안하세요.

        [제약 사항]
        1. 전체 코드를 수정된 상태로 출력해야 합니다.
        2. 기존 로직을 크게 해치지 않아야 합니다.
        3. 주석으로 변경 이유를 설명해야 합니다.
        4. 오직 파이썬 코드만 출력하세요. (마크다운 코드 블록 제외)

        [현재 코드]
        {current_code}
        """

        try:
            # gemini-1.5-flash 모델 사용 (빠르고 저렴)
            response = self.client.models.generate_content(
                model='gemini-1.5-flash',
                contents=prompt
            )

            # 응답 정제 (Markdown 제거)
            new_code = response.text.strip()
            if new_code.startswith("```python"):
                new_code = new_code[9:]
            if new_code.startswith("```"):
                new_code = new_code[3:]
            if new_code.endswith("```"):
                new_code = new_code[:-3]

            return new_code.strip()

        except Exception as e:
            print(f"❌ AI 분석 요청 실패: {e}")
            return None

    def safe_apply_update(self, file_path, new_code):
        """제안된 코드를 검증하고 사용자 승인 후 적용합니다."""
        if not new_code: return {"success": False, "detail": "No code generated"}

        # 1. 문법 검사 (Syntax Check)
        try:
            ast.parse(new_code)
            print("✅ [Safety] 제안된 코드 문법 검사 통과.")
        except SyntaxError as e:
            print(f"❌ [Safety] 제안된 코드에 문법 오류가 있습니다: {e}")
            return {"success": False, "detail": f"Syntax Error: {e}"}

        # 2. 변경 사항 비교 (Diff)
        print("\n📝 [Diff Check] 변경 사항 미리보기:")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
        except:
            original_lines = []

        new_lines = new_code.splitlines(keepends=True)
        diff = difflib.unified_diff(original_lines, new_lines, fromfile='Original', tofile='Proposed')

        diff_text = "".join(diff)
        if not diff_text:
            print("ℹ️ 변경 사항이 없습니다.")
            return {"success": False, "detail": "No changes detected"}

        print(diff_text[:2000] + "\n...(생략)..." if len(diff_text) > 2000 else diff_text)

        # 3. 사용자 승인 (Human-in-the-loop)
        # 백그라운드 실행 중일 때는 터미널 입력 불가능하므로 로그만 남기고 종료
        if not sys.stdin.isatty():
            print("ℹ️ 백그라운드 모드: 변경 제안만 생성하고 적용은 보류합니다.")
            # 실제로는 변경 제안을 파일로 저장해두는 것이 좋음 (proposals/ 폴더 등)
            return {"success": False, "detail": "Background mode (Proposal skipped)"}

        print("\n⚠️ [Caution] 위 변경 사항을 적용하시겠습니까?")
        choice = input("👉 승인하려면 'Y'를 입력하세요 (그 외 취소): ").strip().upper()

        if choice == 'Y':
            # 백업 생성
            backup_path = file_path + ".bak"
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.writelines(original_lines)
                print(f"💾 백업 파일 생성됨: {backup_path}")

                # 파일 덮어쓰기
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_code)
                print(f"✅ {file_path} 업데이트 완료! (Phase 4 Evolution)")
                return {"success": True, "detail": "Applied updates"}
            except Exception as e:
                print(f"❌ 파일 쓰기 실패: {e}")
                return {"success": False, "detail": f"Write Error: {e}"}
        else:
            print("❌ 변경이 취소되었습니다.")
            return {"success": False, "detail": "User cancelled"}

    def execute_evolution_cycle(self, target_file='lotto_predict.py'):
        """전체 진화 사이클 실행 및 결과 반환"""
        new_code = self.analyze_code(target_file)
        if new_code:
            return self.safe_apply_update(target_file, new_code)
        return {"success": False, "detail": "Analysis failed"}

if __name__ == "__main__":
    # 단독 실행 시 테스트
    manager = EvolutionManager()
    manager.execute_evolution_cycle()
