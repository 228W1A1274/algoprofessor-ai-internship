"""
agent.py — The AI Agent (Brain of TalentTwin)
----------------------------------------------
Uses Groq API (FREE, fast, no rate limits for normal use)
Model: llama3-8b-8192 (fast, smart, completely free)
"""

import time
from groq import Groq
from config import CALENDLY_URL
from memory import load_profile, build_context_text, format_conversation_history
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Create Groq client ──
client = Groq(api_key=GROQ_API_KEY)

# ── Model — free, fast, reliable ──
MODEL_NAME = "llama-3.3-70b-versatile"


def build_system_prompt(context_text: str) -> str:
    return f"""You are an AI agent acting as a personal job representative for a candidate.
Your job is to answer recruiter questions on behalf of this candidate in a professional,
confident, and friendly manner.

You have access to the following candidate profile data:

{context_text}

INSTRUCTIONS:
1. Answer ONLY based on the profile data above. Do NOT make up experience or skills.
2. If a question cannot be answered from the profile, say: "I don't have that information,
   but you can reach out directly to discuss further."
3. Keep answers concise and recruiter-friendly (3-5 sentences max unless asked for more).
4. Highlight the candidate's strongest points when relevant.
5. For scheduling/interview questions, mention the Calendly link: {CALENDLY_URL}
6. Be enthusiastic but professional — like a good career advisor speaking on their behalf.
7. Use "The candidate has..." or "They have..." when referring to the candidate.
8. If asked about salary/compensation, say it depends on the role and to discuss directly.

Remember: You are the candidate's digital representative. Make them look great!"""


def ask_agent(
    question: str,
    conversation_history: list = None,
    profile_path: str = "data/profile.json"
) -> str:

    # ── Step 1: Load profile ──
    profile = load_profile(profile_path)
    if not profile:
        return (
            "Profile not loaded yet. Please set your GitHub username in the .env file "
            "and restart the backend."
        )

    # ── Step 2: Build context ──
    context_text = build_context_text(profile)
    system_prompt = build_system_prompt(context_text)

    # ── Step 3: Build messages list for Groq ──
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last 10 messages only)
    if conversation_history:
        for msg in conversation_history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Add current question
    messages.append({"role": "user", "content": question})

    # ── Step 4: Call Groq API with retry ──
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)

            if "429" in error_msg or "rate" in error_msg.lower():
                if attempt < 2:
                    time.sleep((attempt + 1) * 10)
                    continue
                return "Rate limit reached. Please wait 10 seconds and try again."

            elif "401" in error_msg or "api key" in error_msg.lower() or "auth" in error_msg.lower():
                return "Groq API key is invalid. Check GROQ_API_KEY in your .env file."

            elif "model" in error_msg.lower() or "404" in error_msg:
                return "Model not found. Check MODEL_NAME in agent.py."

            else:
                return f"AI Error: {error_msg}"
