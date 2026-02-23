import os
import sys
from dotenv import load_dotenv

# Try importing the new SDK
try:
    from google import genai
except ImportError:
    print("❌ Critical: 'google-genai' library not found.")
    print("💡 Please run: pip install google-genai")
    sys.exit(1)

def main():
    print("\n🛰️ Gemini Model Intelligence Explorer Initializing...")

    # 1. API Key Check
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file.")
        sys.exit(1)

    print("✅ API Key loaded successfully.")

    # Initialize Client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"❌ Failed to initialize Gemini Client: {e}")
        sys.exit(1)

    # 2. Model Discovery
    print("\n🔍 Scanning for available Gemini models...")
    print("-" * 60)
    print(f"{'Model Name':<40} | {'Status':<15}")
    print("-" * 60)

    available_models = []

    try:
        # List models (paginated iterator)
        for model in client.models.list():
            # Check if it supports content generation
            # Note: SDK attributes might vary, checking safely
            methods = getattr(model, 'supported_generation_methods', [])

            if 'generateContent' in methods:
                name = model.name.replace('models/', '')
                print(f"{name:<40} | {'Ready 🟢':<15}")
                available_models.append(name)
    except Exception as e:
        print(f"⚠️ Model scanning encountered an issue: {e}")
        # If list fails, we can't proceed with selection effectively

    print("-" * 60)

    if not available_models:
        print("❌ No models found supporting 'generateContent'. Check API permissions.")
        sys.exit(1)

    # 3. Select Best Model (Priority Logic)
    # Priority: 2.0 > 1.5 Pro > 1.5 Flash > Pro
    def model_priority(name):
        if 'gemini-2.0' in name: return 4
        if 'gemini-1.5-pro' in name: return 3
        if 'gemini-1.5-flash' in name: return 2
        if 'gemini-pro' in name: return 1
        return 0

    available_models.sort(key=model_priority, reverse=True)
    best_model = available_models[0]

    print(f"\n🎯 Target Locked: [{best_model}] selected for Firing Test.")

    # 4. Firing Test
    print("\n💥 Initiating Firing Test...")
    try:
        response = client.models.generate_content(
            model=best_model,
            contents="Hello, Commander! Report your status."
        )

        print("\n📝 Mission Response:")
        print(f"> {response.text.strip()}")

        print("\n" + "="*60)
        print(f"✅ 사령관님, 현재 [{best_model}]이 최상의 컨디션으로 대기 중입니다.")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Firing Test Failed: {e}")
        print("💡 Suggestion: Check API Quota or Model Access.")

if __name__ == "__main__":
    main()
