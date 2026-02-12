import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found")

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # ✅ ONLY THIS
        google_api_key=api_key,
        temperature=0.3
    )
