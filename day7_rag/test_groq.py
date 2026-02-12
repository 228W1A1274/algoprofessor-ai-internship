import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def test_groq():
    """Test if Groq API key is working with current models"""
    
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file!")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        client = Groq(api_key=api_key)
        
        print("\n🚀 Testing Groq API with current model...")
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Current supported model
            messages=[
                {"role": "user", "content": "Say 'Groq API is working!' in one sentence."}
            ],
            max_tokens=50,
            temperature=0
        )
        
        print(f"✅ Response: {response.choices[0].message.content}")
        print(f"✅ Model used: {response.model}")
        print(f"✅ Tokens used: {response.usage.total_tokens}")
        print("\n🎉 Groq API is working perfectly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 If you see 'model_decommissioned' error:")
        print("   Check available models at: https://console.groq.com/docs/models")

if __name__ == "__main__":
    test_groq()