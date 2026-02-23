# -*- coding: utf-8 -*-
import os
import sys
import difflib
import ast
import datetime
import json
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    print("❌ 'google-genai' 라이브러리가 필요합니다. (pip install google-genai)")
    sys.exit(1)

load_dotenv()

class EvolutionManager:
    """
    🧬 [Phase 4] 자율 진화 관리자 (Self-Evolution Manager)
    - 코드 자가 수정 (Code Evolution)
    - 프롬프트 자가 개선 (Meta-Prompting)
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("🧬 [Evolution] Gemini AI 연결 성공.")
            except Exception as e:
                print(f"❌ Gemini 클라이언트 초기화 실패: {e}")
                self.client = None

    # ------------------------------------------------------------------
    # 1. Code Evolution (기존 기능)
    # ------------------------------------------------------------------
    def analyze_code(self, file_path='lotto_predict.py'):
        if not self.client: return None
        print(f"🔍 [Code Evolution] {file_path} 분석 중...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_code = f.read()
        except FileNotFoundError:
            return None

        prompt = f"""
        당신은 파이썬 전문가입니다. 아래 코드를 분석하여 성능, 가독성, 안정성을 높일 수 있는 개선안을 1가지만 제안하세요.
        반드시 전체 코드를 수정된 상태로 출력해야 합니다.

        [현재 코드]
        {current_code}
        """
        try:
            response = self.client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            new_code = response.text.strip()
            if new_code.startswith("```python"): new_code = new_code[9:]
            if new_code.startswith("```"): new_code = new_code[3:]
            if new_code.endswith("```"): new_code = new_code[:-3]
            return new_code.strip()
        except: return None

    def safe_apply_update(self, file_path, new_code):
        if not new_code: return {"success": False, "detail": "No code generated"}
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            return {"success": False, "detail": f"Syntax Error: {e}"}

        print("\n📝 [Code Diff] 변경 사항 미리보기:")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
        except: original_lines = []

        new_lines = new_code.splitlines(keepends=True)
        diff = difflib.unified_diff(original_lines, new_lines, fromfile='Original', tofile='Proposed')
        diff_text = "".join(diff)

        if not diff_text: return {"success": False, "detail": "No changes"}
        print(diff_text[:1000] + "...")

        if not sys.stdin.isatty():
            return {"success": False, "detail": "Background mode (Skipped)"}

        choice = input("👉 코드 변경을 승인하시겠습니까? (Y/N): ").strip().upper()
        if choice == 'Y':
            backup_path = file_path + ".bak"
            try:
                with open(backup_path, 'w', encoding='utf-8') as f: f.writelines(original_lines)
                with open(file_path, 'w', encoding='utf-8') as f: f.write(new_code)
                return {"success": True, "detail": "Applied updates"}
            except Exception as e:
                return {"success": False, "detail": f"Write Error: {e}"}
        return {"success": False, "detail": "User cancelled"}

    # ------------------------------------------------------------------
    # 2. Meta-Prompting (신규 기능)
    # ------------------------------------------------------------------
    def evolve_prompt(self, current_prompt, recent_stats):
        """
        [Meta-Prompting] 성과 데이터를 바탕으로 분석 프롬프트를 스스로 개선합니다.
        """
        if not self.client:
            return {"success": False, "detail": "No AI Client"}

        avg_hit = np.mean(recent_stats) if recent_stats else 0.0
        print(f"🧬 [Prompt Evolution] 최근 평균 적중률: {avg_hit:.2f}개")

        meta_prompt = f"""
        당신은 '프롬프트 엔지니어링 AI'입니다.
        우리는 로또 번호 예측 시스템을 운영 중이며, 현재 사용 중인 '분석 프롬프트'의 성능을 개선하고 싶습니다.

        [현재 프롬프트]
        "{current_prompt}"

        [최근 성과]
        - 최근 5주 평균 적중 개수: {avg_hit:.2f}개 (목표: 3.0개 이상)

        [임무]
        위 성과를 분석하여, 더 높은 적중률을 낼 수 있도록 '현재 프롬프트'를 수정/보완해 주세요.
        예를 들어, "번호의 분포를 더 넓게 퍼뜨려라", "최근 10회차의 미출현 번호를 고려하라" 등의 구체적 지시를 추가할 수 있습니다.

        [출력 형식]
        오직 개선된 프롬프트 내용만 출력하세요. (설명이나 마크다운 없이)
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=meta_prompt
            )
            new_prompt = response.text.strip()
            return {"success": True, "new_prompt": new_prompt}
        except Exception as e:
            return {"success": False, "detail": f"Meta-Prompting Failed: {e}"}

    # ------------------------------------------------------------------
    # Main Entry
    # ------------------------------------------------------------------
    def execute_evolution_cycle(self, target_file='lotto_predict.py', state_manager=None):
        """
        전체 진화 사이클:
        1. 코드 최적화 제안 (Interactive)
        2. 프롬프트 진화 (Background/Automatic)
        """
        results = {}

        # 1. 코드 진화 (사용자 개입 필요)
        if sys.stdin.isatty():
            code_res = self.analyze_code(target_file)
            if code_res:
                results['code'] = self.safe_apply_update(target_file, code_res)

        # 2. 프롬프트 진화 (자동 수행)
        if state_manager:
            current_state = state_manager.state
            current_prompt = current_state.get("active_strategy_prompt", {}).get("content", "")
            recent_stats = current_state.get("recent_hit_rates", [])

            # 성과 데이터가 충분할 때만 진화 시도
            if len(recent_stats) > 0:
                print("🧬 [Meta-Prompting] 프롬프트 자가 개선 시도 중...")
                prompt_res = self.evolve_prompt(current_prompt, recent_stats)

                if prompt_res.get("success"):
                    new_version = f"v{datetime.datetime.now().strftime('%m%d-%H%M')}"
                    state_manager.update_strategy_prompt(prompt_res['new_prompt'], new_version)
                    results['prompt'] = {"success": True, "version": new_version}
                    print(f"✨ 전략 프롬프트가 '{new_version}'으로 진화했습니다.")
                else:
                    results['prompt'] = prompt_res
            else:
                print("ℹ️ 데이터 부족으로 프롬프트 진화를 건너뜁니다.")

        return results

if __name__ == "__main__":
    manager = EvolutionManager()
    # 테스트용 모의 객체
    class MockState:
        state = {"active_strategy_prompt": {"content": "Test Prompt"}, "recent_hit_rates": [2.5, 3.0]}
        def update_strategy_prompt(self, p, v): print(f"Updated: {v}")

    manager.execute_evolution_cycle(state_manager=MockState())
