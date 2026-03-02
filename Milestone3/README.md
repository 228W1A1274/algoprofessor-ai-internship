# ✈️ SkyAssist — Multi-Modal Airline Support Agent
### Milestone 3 | Day 20 | AlgoProfessor AI R&D Internship 2026

---

## 🎯 What This Project Does

SkyAssist is a complete AI-powered airline customer support agent that handles:

| Input Type | Technology | Cost |
|---|---|---|
| Text messages | GPT-4o (reasoning + tools) | ~$0.003/session |
| Voice messages | OpenAI Whisper (local) | **FREE** |
| Boarding pass images | GPT-4o Vision (OCR) | ~$0.008/image |

**Capabilities:**
- 🔍 Flight search by route, date, and cabin class
- 📄 Booking lookup by reference code
- ⬆️ Seat changes and cabin class upgrades
- 🧳 Baggage policy Q&A
- 📋 Boarding pass scanning and interpretation

---

## 🏗️ Architecture

```
INPUT (Text / Voice / Image)
        │
        ├── Voice → Whisper STT → Text
        └── Image → GPT-4o Vision → Text
        │
        ▼
Entity Extractor (gpt-4o-mini)
        │
        ▼
ReAct Agent (gpt-4o)
  ┌─────┴──────┐
  │ Tool Loop  │ ← search_flights / lookup_booking / upgrade_seat / baggage_policy
  └─────┬──────┘
        │
        ▼
Gradio Web UI (localhost:7860 or public URL)
```

**Memory System:** Entity Store (permanent facts) + Buffer Window (last 5 turns)
**Cost Strategy:** gpt-4o-mini for extraction, gpt-4o for reasoning

---

## 📁 Repository Structure

```
skyassist/
├── notebook/
│   └── SkyAssist_Milestone3.ipynb   ← Main deliverable (run this)
├── src/
│   └── skyassist_app.py             ← Standalone Python version
├── assets/
│   ├── skyassist_architecture.png   ← System architecture diagram
│   ├── react_agent_flow.png         ← ReAct loop visualization
│   ├── memory_architecture.png      ← Memory system diagram
│   └── token_economics.png          ← Cost analysis chart
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open `notebook/SkyAssist_Milestone3.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run cells top to bottom
4. Enter your OpenAI API key when prompted
5. Click the public URL to open the app

### Option 2: Local
```bash
git clone <your-repo-url>
cd skyassist
pip install -r requirements.txt
python src/skyassist_app.py
# Open http://localhost:7860
```

---

## 🔑 API Key
Get a free OpenAI API key at: https://platform.openai.com/api-keys
New accounts get $5 free credit. This full demo costs less than $0.05.

---

## 📚 Sprint Learning Summary

| Day | Topic | Applied In |
|---|---|---|
| Day 16 | OpenAI API + Token Management | CostTracker, dual-model routing |
| Day 17 | Prompt Engineering + ReAct | Agent system prompt, tool loop |
| Day 18 | Function Calling + Pydantic | 4 airline tools with validation |
| Day 19 | Memory (Buffer + Entity) | EntityStore + conversation buffer |
| Day 20 | Whisper + Vision + Gradio | Full multimodal pipeline |

---

## 🧪 Demo Bookings (for testing)

| Booking Ref | Passenger | Flight | Route | Class |
|---|---|---|---|---|
| SKY123 | James Wilson | SK101 | JFK → LHR | Economy |
| SKY456 | Priya Sharma | SK202 | LHR → DXB | Business |
| SKY789 | Marcus Chen | SK303 | DXB → SIN | First |

---

*AlgoProfessor AI R&D Internship | 3-Month Course | AI to Agentic AI*
