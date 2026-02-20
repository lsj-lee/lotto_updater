import os
import glob
import shutil
import sys
import re

def scan_proposals():
    proposal_dir = "proposals"
    if not os.path.exists(proposal_dir):
        return None

    files = glob.glob(os.path.join(proposal_dir, "*_proposal.py"))
    if not files:
        return None

    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def extract_header(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Try to find the docstring block
        # Usually it's at the very top, or just after shebang/imports
        # We look for """ [Evolution Proposal] ... """
        # Matches """ followed by anything, then [Evolution Proposal], then anything, then """
        match = re.search(r'"""(.*?)\[Evolution Proposal\](.*?)"""', content, re.DOTALL)
        if match:
            # Reconstruct the text
            full_text = f"[Evolution Proposal]{match.group(2)}"
            # Clean up indentation for display
            lines = [line.strip() for line in full_text.split('\n')]
            return "\n".join(lines)

        # Fallback: just look for the text directly if regex fails
        if "[Evolution Proposal]" in content:
            start = content.find("[Evolution Proposal]")
            # Look for the closing triple quotes
            end = content.find('"""', start + 20)
            if end != -1:
                raw_text = content[start:end]
                lines = [line.strip() for line in raw_text.split('\n')]
                return "\n".join(lines)

        return "⚠️ 제안서 헤더를 찾을 수 없습니다. 코드를 직접 확인하세요."
    except Exception as e:
        return f"⚠️ 파일 읽기 오류: {e}"

def main():
    print("\n" + "="*60)
    print("🤵 [Evolution Manager] 상진 CEO님, 환영합니다.")
    print("="*60)

    latest_proposal = scan_proposals()

    if not latest_proposal:
        print("\n📭 현재 검토 대기 중인 진화 제안서가 없습니다.")
        print("   (lotto_predict.py 실행 후 제안서가 생성됩니다.)")
        print("\n" + "="*60)
        return

    print(f"\n📄 최신 제안서 도착: {os.path.basename(latest_proposal)}")
    print("-" * 60)

    header_info = extract_header(latest_proposal)
    print(header_info)

    print("-" * 60)
    print("\n[의사결정 프로세스]")
    print("  [A] Apply  : 승인 및 시스템 적용 (자동 백업 수행)")
    print("  [D] Delete : 거절 및 제안서 삭제")
    print("  [C] Cancel : 보류 및 종료")

    while True:
        choice = input("\n👉 CEO님의 결정을 입력해주세요 (A/D/C): ").strip().upper()

        if choice == 'A':
            print("\n🔄 시스템 업그레이드를 시작합니다...")
            try:
                # Backup
                if os.path.exists("lotto_predict.py"):
                    shutil.copy("lotto_predict.py", "lotto_predict_bak.py")
                    print("  ✅ 현재 시스템 백업 완료 (lotto_predict_bak.py)")

                # Apply
                shutil.copy(latest_proposal, "lotto_predict.py")
                print("  ✅ 차세대 코드 적용 완료 (lotto_predict.py)")
                print("\n✨ 진화가 성공적으로 적용되었습니다. 행운을 빕니다!")
            except Exception as e:
                print(f"❌ 적용 중 오류 발생: {e}")
            break

        elif choice == 'D':
            print("\n🗑️ 제안서를 삭제합니다...")
            try:
                os.remove(latest_proposal)
                print("  ✅ 삭제 완료.")
            except Exception as e:
                print(f"❌ 삭제 중 오류 발생: {e}")
            break

        elif choice == 'C':
            print("\n⏳ 결정을 보류하고 종료합니다.")
            break

        else:
            print("⚠️ 올바른 옵션을 선택해주세요.")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
