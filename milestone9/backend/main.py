"""
main.py — FastAPI Backend Server
----------------------------------
This is the entry point of our backend. It:
1. Creates a FastAPI web server
2. Exposes API endpoints (URLs that the frontend calls)
3. Connects the frontend chat UI to the AI agent

Key API Endpoints:
  GET  /              → Health check (is the server running?)
  GET  /profile       → Returns the loaded candidate profile
  POST /chat          → Main endpoint: takes question, returns AI answer
  POST /refresh       → Re-fetches GitHub data (if you update your profile)
  GET  /calendly      → Returns your Calendly URL

FastAPI automatically creates documentation at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json, os

from config import GITHUB_USERNAME, CORS_ORIGINS, CALENDLY_URL, APP_HOST, APP_PORT
from agent import ask_agent
from github_loader import save_github_data
from memory import load_profile

# ── Create the FastAPI app instance ──
app = FastAPI(
    title="TalentTwin AI Agent",
    description="AI-powered job representation agent — answers recruiter questions on your behalf",
    version="1.0.0"
)

# ── Add CORS Middleware ──
# CORS = Cross-Origin Resource Sharing
# Without this, the browser blocks the frontend from calling the backend
# (because they run on different URLs/ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,   # Which frontends can talk to us
    allow_credentials=True,
    allow_methods=["*"],          # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],          # Allow all headers
)

# ── In-memory conversation history ──
# Dict mapping session_id → list of messages
# NOTE: This resets when the server restarts (no database needed for now)
conversation_sessions: dict[str, list[dict]] = {}


# ──────────────────────────────────────────────
# 📦 Request/Response Models (Pydantic)
# These define the shape of data we accept/return
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """What the frontend sends when the recruiter asks a question"""
    question: str               # The recruiter's question
    session_id: str = "default" # Identify the chat session (for memory)

class ChatResponse(BaseModel):
    """What we send back to the frontend"""
    answer: str         # The AI's answer
    session_id: str     # Echo back the session ID


# ──────────────────────────────────────────────
# 🌐 API Endpoints
# ──────────────────────────────────────────────

@app.get("/")
def health_check():
    """
    Health check endpoint.
    When Render deploys our app, it pings this URL to verify the server is alive.
    
    Expected response: {"status": "ok", "message": "TalentTwin is running!"}
    """
    return {"status": "ok", "message": "TalentTwin AI Agent is running! 🚀"}


@app.get("/profile")
def get_profile():
    """
    Returns the candidate's loaded profile data.
    The frontend can use this to display profile info alongside the chat.
    """
    profile = load_profile()
    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Profile not found. Make sure GITHUB_USERNAME is set and /refresh was called."
        )
    return profile


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    MAIN ENDPOINT — This is called every time a recruiter sends a message.
    
    Flow:
    1. Receive question + session_id from frontend
    2. Get the conversation history for this session
    3. Ask the AI agent
    4. Save the Q&A to conversation history
    5. Return the answer to the frontend
    """
    question = request.question.strip()
    session_id = request.session_id
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Get or create conversation history for this session
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []
    
    history = conversation_sessions[session_id]
    
    # Ask the AI agent (this calls Gemini API internally)
    answer = ask_agent(
        question=question,
        conversation_history=history,
        profile_path="data/profile.json"
    )
    
    # Save this turn to history (so the AI remembers previous messages)
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})
    
    # Keep history to last 20 messages (10 turns) to avoid memory bloat
    if len(history) > 20:
        conversation_sessions[session_id] = history[-20:]
    
    return ChatResponse(answer=answer, session_id=session_id)


@app.post("/refresh")
def refresh_profile():
    """
    Re-fetches your GitHub data and updates profile.json.
    Call this when you push new projects to GitHub and want the AI to know about them.
    """
    if not GITHUB_USERNAME:
        raise HTTPException(
            status_code=400,
            detail="GITHUB_USERNAME not set in .env file"
        )
    
    try:
        data = save_github_data(GITHUB_USERNAME, output_path="data/profile.json")
        return {
            "status": "success",
            "message": f"Profile refreshed for {GITHUB_USERNAME}",
            "repos_count": len(data.get("repositories", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GitHub fetch failed: {str(e)}")


@app.get("/calendly")
def get_calendly():
    """Returns your Calendly scheduling link"""
    return {"url": CALENDLY_URL}


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    """Clear conversation history for a session (start fresh)"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


# ── Run the server directly ──
# This block runs when you type: python main.py
# In production (Render), Render uses: uvicorn main:app instead
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🚀 TalentTwin AI Agent — Starting Up")
    print("=" * 50)
    
    # Auto-load GitHub data on startup if profile doesn't exist yet
    if GITHUB_USERNAME and not os.path.exists("data/profile.json"):
        print(f"\n📡 Auto-fetching GitHub data for: {GITHUB_USERNAME}")
        try:
            save_github_data(GITHUB_USERNAME, output_path="data/profile.json")
            print("✅ Profile loaded successfully!")
        except Exception as e:
            print(f"⚠️  Could not fetch GitHub data: {e}")
            print("   You can manually call POST /refresh after the server starts.")
    
    print(f"\n🌐 Server running at: http://localhost:{APP_PORT}")
    print(f"📖 API Docs at:       http://localhost:{APP_PORT}/docs")
    print(f"🔍 Health check:      http://localhost:{APP_PORT}/")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
