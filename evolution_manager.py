# -*- coding: utf-8 -*-
import os
import sys
import difflib
import ast
import datetime
import numpy as np
from dotenv import load_dotenv

# 🎯 무기고(Armory) 시스템 연동
from model_selector import SniperArmory

load_dotenv()

class EvolutionManager:
    """
    🧬 [Phase 4] 자율 진화 관리자 (Self-Evolution Manager)
    """
    def __init__(self):
        self.armory = SniperArmory()
        print("🧬 [Evolution] 무기고(Armory) 연동 완료.")

    def analyze_code(self, file_path='lotto_predict.py'):
        print(f"🔍 [Code Evolution] {file_path} 분석 중 (고급 타격)...")
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
        
        # 🎯 코드 분석은 지능이 높아야 하므로 '고급' 무기 사용
        new_code = self.armory.fire_prompt(prompt, target_tier="고급")
        
        if new_code:
            new_code = new_code.strip()
            if new_code.startswith("```python"): new_code = new_code[9:]
            if new_code.startswith("```"): new_code = new_code[3:]
            if new_code.endswith("```"): new_code = new_code[:-3]
            return new_code.strip()
        return None

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

    def evolve_prompt(self, current_prompt, recent_stats):
        avg_hit = np.mean(recent_stats) if recent_stats else 0.0
        print(f"🧬 [Prompt Evolution] 최근 평균 적중률: {avg_hit:.2f}개 (하급 타격)")

        meta_prompt = f"""
        당신은 '프롬프트 엔지니어링 AI'입니다.
        현재 프롬프트를 수정하여 로또 적중률을 높여주세요. 출력은 오직 개선된 프롬프트 내용만 해주세요.

        [현재 프롬프트]
        "{current_prompt}"
        [최근 성과]
        최근 5주 평균 적중 개수: {avg_hit:.2f}개
        """

        # 🎯 프롬프트 반복 개선은 탄약이 많은 '하급' 무기 사용
        new_prompt = self.armory.fire_prompt(meta_prompt, target_tier="하급")
        
        if new_prompt:
            return {"success": True, "new_prompt": new_prompt.strip()}
        return {"success": False, "detail": "Prompt evolution failed."}

    def execute_evolution_cycle(self, target_file='lotto_predict.py', state_manager=None):
        results = {}
        if sys.stdin.isatty():
            code_res = self.analyze_code(target_file)
            if code_res:
                results['code'] = self.safe_apply_update(target_file, code_res)

        if state_manager:
            current_state = state_manager.state
            current_prompt = current_state.get("active_strategy_prompt", {}).get("content", "")
            recent_stats = current_state.get("recent_hit_rates", [])

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
        return results

if __name__ == "__main__":
    manager = EvolutionManager()