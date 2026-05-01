# 🤖 AI Browser Sidekick

> OpenAI Operator-style browser agent — plain English → automated web tasks

**AlgoProfessor AI R&D Internship | Days 73–75**

---

## What This Is

An AI-powered Chrome Extension that lets you give plain English instructions and have an intelligent agent execute them in your browser. Inspired by OpenAI Operator.

```
"Fill the job application at careers.example.com with my details"
        ↓
  Chrome Extension (popup.html)
        ↓  POST /task/stream
  FastAPI Backend (main.py)
        ↓
  LangGraph Agent (agent.py)
    ↓         ↑
 Think      Observe
    ↓
  Tool Node (tools.py)
        ↓
  Playwright Browser
        ↓
  🌐 Actual webpage interactions
```

---

## Quick Start

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env → add your GROQ_API_KEY
# Get free key at: https://console.groq.com/

# Start the backend
uvicorn main:app --reload
```

Backend runs at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 2. Chrome Extension Setup

1. Open Chrome → go to `chrome://extensions`
2. Enable **Developer Mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `extension/` folder
5. The 🤖 icon appears in your toolbar

### 3. Test It

1. Click the 🤖 icon
2. Check the green dot (backend connected)
3. Type: `Go to https://httpbin.org/forms/post and fill the customer name field with "Test User"`
4. Click **▶ Run Task**
5. Watch the steps appear in real-time!

---

## AI Model Selection

| Provider | Speed | Tool Support | Free Tier | Winner |
|----------|-------|--------------|-----------|--------|
| **Groq** | ~300 tok/s | ✅ Full | 6k req/day | ✅ **Best** |
| OpenRouter | ~80 tok/s | Model-dependent | $5 credit | Runner-up |
| Gemini Flash | ~150 tok/s | Limited | 1M tok/mo | Fallback |

**Using: Groq + `llama-3.3-70b-versatile`**

---

## Project Structure

```
ai-sidekick/
├── backend/
│   ├── main.py          ← FastAPI app + SSE streaming endpoint
│   ├── agent.py         ← LangGraph ReAct agent
│   ├── tools.py         ← Playwright browser tools
│   ├── config.py        ← Environment settings
│   ├── requirements.txt
│   └── .env.example
│
├── extension/
│   ├── manifest.json    ← Chrome Extension V3 config
│   ├── popup.html       ← Extension UI
│   ├── popup.js         ← Task submission + SSE consumer
│   ├── content.js       ← Injected into web pages
│   └── background.js    ← Service worker
│
└── README.md
```

---

## End-to-End Flow Example

**User input:** "Fill job application form"

1. **popup.js** collects instruction + current URL
2. **POST /task/stream** sent to FastAPI backend
3. **FastAPI** creates SSE StreamingResponse
4. **LangGraph** initialises AgentState with SystemPrompt + HumanMessage
5. **Agent node** calls LLM → decides to call `get_page_info`
6. **Tool node** runs `get_page_info` → returns inputs/buttons on page
7. **Agent node** calls LLM again → decides to call `fill_input(#name, "John Doe")`
8. **Tool node** runs Playwright fill → returns "Filled #name"
9. **Repeat** until form submitted or task complete
10. **SSE events** stream back to popup.js at each step
11. **popup.js** renders each step in real-time UI
12. **Final result** shown when agent returns END

---

## Debugging

### Backend won't start
```bash
# Check Python version (need 3.11+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check .env has GROQ_API_KEY set
cat .env
```

### Playwright browser doesn't open
```bash
playwright install chromium
playwright install-deps  # Linux only
```

### Extension shows red dot (offline)
- Ensure backend is running: `uvicorn main:app --reload`
- Check backend is at http://localhost:8000
- Check Chrome console (F12 in extension popup)

### CORS error in console
- The FastAPI backend has CORS enabled for all origins
- If still failing, check the backend terminal for errors

### Extension permissions error
- Go to `chrome://extensions`
- Click "Details" on AI Browser Sidekick
- Enable "Allow on all sites" under Site Access

---

## Deployment

### Backend (Free)
**Render.com:**
1. Push backend/ to GitHub
2. Create Web Service on render.com
3. Set environment variables in Render dashboard
4. Change `PLAYWRIGHT_HEADLESS=true` in production

### Chrome Extension
1. Go to `chrome://extensions`
2. Click **Pack Extension** → creates `.crx` + `.pem`
3. For Chrome Web Store: zip the extension/ folder
4. Upload at [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole/)

---

## Viva Questions

1. **What is the difference between Manifest V2 and V3?**
   V3 replaced persistent background pages with Service Workers that are event-driven and terminate when idle.

2. **Why use SSE instead of WebSockets for streaming?**
   SSE is simpler (HTTP/1.1), auto-reconnects, and is perfect for server→client push. WebSockets are bidirectional, which is overkill here.

3. **How does LangGraph prevent infinite loops?**
   The `should_continue` conditional edge checks if the last message has `tool_calls`. No tool calls → END. Also enforced by `MAX_AGENT_STEPS`.

4. **Why Groq over other free providers?**
   Groq uses LPU (Language Processing Unit) hardware for ~300 tokens/sec — 3–10x faster than GPU-based providers. Native tool-calling support is critical for agent workflows.

5. **How is the agent state preserved across tool calls?**
   LangGraph's `AgentState` TypedDict with `add_messages` annotation automatically appends messages, maintaining full conversation history across the ReAct loop.

---

*AlgoProfessor AI R&D Solutions | Batch 2026*
