import os
import sys
import time

# [Execution Guide] Phase 1: Pre-flight Check
print("\n" + "="*60)
print("🚀 [Sniper V5] Gemini API Diagnostic Tool")
print("   - Required Library: google-genai (v1.0+)")
print("   - Command: pip install google-genai python-dotenv")
print("="*60 + "\n")

# Try importing the new SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ Critical: 'google-genai' library not found.")
    print("💡 Please run: pip install google-genai")
    sys.exit(1)

from dotenv import load_dotenv

def validate_api_key(key):
    """
    [Security] Validates API Key format without exposing the full key.
    """
    if not key:
        return False, "Key is empty or None."

    if key.strip() != key:
        return False, "Key has leading/trailing whitespace. Check .env file."

    if len(key) < 30: # Heuristic length check
        return False, f"Key seems too short ({len(key)} chars). Expected > 30."

    # Check for non-ascii chars (encoding issues)
    if not key.isascii():
         return False, "Key contains non-ASCII characters. Check file encoding."

    return True, "Valid Format"

def main():
    print("🛰️ Initializing System Diagnostics...")

    # 1. Environment Variable Verification
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    print(f"\n🔑 API Key Verification:")
    is_valid, message = validate_api_key(api_key)

    if not is_valid:
        print(f"   ❌ {message}")
        print("   ⚠️ Stopping Execution. Please fix .env file.")
        sys.exit(1)
    else:
        masked_key = f"{api_key[:5]}...{api_key[-5:]}"
        print(f"   ✅ Key Format OK ({masked_key})")

    # Initialize Client
    try:
        client = genai.Client(api_key=api_key)
        print("   ✅ Client Initialized.")
    except Exception as e:
        print(f"   ❌ Client Initialization Failed: {e}")
        sys.exit(1)

    # 2. Model List Scan (Detailed)
    print("\n🔍 Scanning for available Gemini models...")
    print("-" * 60)
    print(f"{'Model Name':<40} | {'Status':<20}")
    print("-" * 60)

    available_models = []
    scan_failed = False

    try:
        # Paging through models
        # Note: In v1.0+, client.models.list() returns an iterator of Model objects
        for model in client.models.list():
            # Filter for generation capable models
            # Attributes might vary, check capability safely
            methods = getattr(model, 'supported_generation_methods', [])

            if 'generateContent' in methods:
                name = model.name.replace('models/', '')
                print(f"{name:<40} | {'Ready 🟢':<20}")
                available_models.append(name)
            else:
                # Debug: Show other models too? No, keep it clean.
                pass

    except Exception as e:
        print(f"⚠️ Model List Error: {e}")
        scan_failed = True
        # Often purely permission errors on 'List' but 'Generate' might work
        if "PERMISSION_DENIED" in str(e):
            print("   -> Tip: Your API Key might lack 'List Models' permission but allow generation.")
        elif "INVALID_ARGUMENT" in str(e):
             print("   -> Tip: API Key might be invalid or project restriction.")

    print("-" * 60)

    # 3. Force Fire Mechanism
    # If list is empty or failed, we MUST try a known model directly.
    target_models = available_models if available_models else ['gemini-1.5-flash', 'gemini-1.5-pro']

    if not available_models:
        print("\n⚠️ No models discovered via List API.")
        print("🚀 Initiating FORCE FIRE protocol on fallback models...")
    else:
        print(f"\n🎯 {len(available_models)} models found. Selecting best candidate...")
        # Priority sort
        def model_priority(name):
            if 'gemini-2.0' in name: return 4
            if 'gemini-1.5-pro' in name: return 3
            if 'gemini-1.5-flash' in name: return 2
            return 1
        target_models.sort(key=model_priority, reverse=True)

    # 4. Firing Test
    best_model = target_models[0]
    print(f"\n💥 Executing Firing Test on target: [{best_model}]")

    try:
        response = client.models.generate_content(
            model=best_model,
            contents="Hello, Commander! Status Report."
        )

        print("\n📝 Mission Response:")
        print(f"> {response.text.strip()}")

        print("\n" + "="*60)
        print(f"✅ SYSTEM OPERATIONAL. Model [{best_model}] is active.")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Firing Test Failed on {best_model}:")
        print(f"   Error: {e}")
        print("\n💡 Troubleshooting:")
        if "404" in str(e) or "NOT_FOUND" in str(e):
             print("   - Model name might be incorrect or you don't have access.")
        elif "400" in str(e) or "INVALID_ARGUMENT" in str(e):
             print("   - API Key is likely invalid or project billing is disabled.")
        elif "429" in str(e):
             print("   - Quota exceeded. Slow down or check billing.")

if __name__ == "__main__":
    main()
