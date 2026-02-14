"""
Check Available Groq Models
---------------------------
- Lists all available models
- Tests a selected model
"""

import os
from groq import Groq
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def main():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file")
        return

    print("✅ API Key found")
    print("\n🔍 Connecting to Groq...\n")

    try:
        client = Groq(api_key=api_key)

        # ------------------------------
        # 1️⃣ List Available Models
        # ------------------------------
        print("📌 AVAILABLE MODELS:\n")
        models = client.models.list()

        model_ids = [model.id for model in models.data]

        for i, model_id in enumerate(model_ids, 1):
            print(f"{i}. {model_id}")

        # ------------------------------
        # 2️⃣ Test a Model
        # ------------------------------
        print("\n🧪 Testing first available model...\n")

        test_model = model_ids[0]
        print(f"Using model: {test_model}")

        response = client.chat.completions.create(
            model=test_model,
            messages=[
                {"role": "system", "content": "You are a helpful AI."},
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            temperature=0.5,
            max_tokens=50
        )

        print("\n✅ Model Test Successful!")
        print("Response:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")


if __name__ == "__main__":
    main()
