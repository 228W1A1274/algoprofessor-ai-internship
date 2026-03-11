# 🎙️ MeetScribe — AI Meeting Intelligence Tool

> **AlgoProfessor AI Internship 2026 | Phase 1 | Milestone 4 | Day 25**

---

## 📌 What is MeetScribe?

MeetScribe converts raw meeting audio into structured, actionable intelligence — automatically.

**Pipeline:**
```
Upload Audio
    ↓
Whisper Transcription (faster-whisper)
    ↓
Speaker Diarisation (pyannote.audio)
    ↓
Merge → "SPEAKER_00: Hello everyone..."
    ↓
3-Way LLM Summarisation
  ├── Groq  →  Llama 3.3 70B  (FREE)
  ├── Anthropic → Claude 3.5 Sonnet (optional)
  └── OpenAI  →  GPT-4o        (optional)
    ↓
PDF Report Export (ReportLab)
    ↓
Slack / Email Notification
    ↓
Gradio Web UI
```

---

## 🛠️ Tech Stack

| Tool | Purpose | Cost |
|------|---------|------|
| `faster-whisper` | Speech-to-text transcription | FREE |
| `pyannote.audio` | Speaker diarisation (who spoke when) | FREE |
| `Groq + Llama 3.3 70B` | Primary LLM summarisation | **100% FREE** |
| `Anthropic Claude 3.5` | LLM comparison | Optional |
| `OpenAI GPT-4o` | LLM comparison | Optional |
| `ReportLab` | PDF meeting report generation | FREE |
| `Gradio` | Web interface | FREE |
| `Matplotlib` | Analytics charts | FREE |

---

## 📁 Project Structure

```
meetscribe/
├── MeetScribe_Colab.ipynb          ← Main runnable notebook (Google Colab)
├── requirements.txt                ← All Python dependencies
├── README.md                       ← This file
│
├── data/
│   └── sample_meeting.wav          ← Sample test audio file
│
├── outputs/                        ← Auto-generated after running
│   ├── transcript_raw.json         ← Raw Whisper transcription output
│   ├── diarised_transcript.json    ← Merged speaker + text segments
│   ├── llm_comparison.json         ← All 3 LLM summaries + metrics
│   └── meeting_report.pdf          ← Final professional PDF report
│
└── utils/
    ├── audio_processor.py          ← Whisper transcription + pyannote diarisation
    ├── llm_comparator.py           ← 3-way LLM caller (Groq, Claude, GPT-4o)
    └── pdf_generator.py            ← ReportLab PDF + chart generation
```

---

## 🚀 Quick Start (Google Colab)

### Step 1 — Get API Keys (All Free Options Available)

| Key Name | Where to Get | Required? |
|----------|-------------|-----------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys | ✅ YES (FREE) |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | ✅ For diarisation |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | ⚙️ Optional |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | ⚙️ Optional |
| `SLACK_WEBHOOK_URL` | Slack App → Incoming Webhooks | ⚙️ Optional |

### Step 2 — Accept pyannote License
Go to → [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
Click **"Agree and access repository"**

### Step 3 — Open in Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload Notebook → select `MeetScribe_Colab.ipynb`
3. Runtime → Change Runtime Type → **T4 GPU** (free)

### Step 4 — Add Secrets
Click the 🔑 **key icon** in the Colab left sidebar → Add each API key

### Step 5 — Run Cells
Run cells **one by one**, top to bottom.
> ⚠️ Do NOT click "Run All" — install cell must complete before others run.

---

## 📊 Sample Output

### Diarised Transcript
```
[SPEAKER_00] (0s–12s): Good morning everyone, let's get started with today's standup.
[SPEAKER_01] (13s–28s): Yesterday I finished the authentication module and pushed to GitHub.
[SPEAKER_00] (29s–41s): Great, any blockers on your end?
[SPEAKER_01] (42s–55s): Yes, I need access to the staging database to run integration tests.
```

### LLM Summary (Llama 3.3 70B via Groq — FREE)
```
EXECUTIVE SUMMARY
Brief standup meeting covering yesterday's progress and current blockers.
SPEAKER_01 completed the authentication module. Access to staging database
is the current blocker requiring action from SPEAKER_00.

ACTION ITEMS
- [SPEAKER_00] must grant staging database access to SPEAKER_01 — ASAP

MEETING SENTIMENT: Positive — collaborative and focused discussion.
```

---

## ⏱️ Performance Benchmarks

| Audio Length | Whisper Time | Diarisation | Total Pipeline |
|-------------|-------------|-------------|----------------|
| 5 minutes | ~45 sec | ~60 sec | ~2.5 min |
| 15 minutes | ~2 min | ~2.5 min | ~6 min |
| 30 minutes | ~4 min | ~5 min | ~11 min |

*Tested on Google Colab free tier (CPU), Whisper base model*

---

## 🔧 Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `401 Unauthorized` from pyannote | Accept model license at huggingface.co/pyannote/speaker-diarization-3.1 |
| `CUDA out of memory` | Change Whisper model from `large` to `base` |
| Gradio URL not appearing | Check Cell 14 ran without errors; look for `https://xxxx.gradio.live` |
| `No module named faster_whisper` | Run Cell 1 again (install cell) |
| OpenAI `insufficient_quota` | Use Groq (free) — GPT-4o integration is coded, just needs credits |

---

## 📈 Evaluation Rubric

| Component | Weight | What's Assessed |
|-----------|--------|----------------|
| Functionality | 40% | Does the full pipeline run end-to-end? |
| UI/UX | 20% | Is the Gradio interface clean and usable? |
| Code Quality | 20% | Is the code structured, commented, PEP8? |
| Documentation | 20% | README, comments, output clarity |

---

## 🎓 Learning Outcomes (Days 21–25)

- ✅ HuggingFace ecosystem: pipeline API, model hub, Inference API
- ✅ Open-source LLMs: Ollama, vLLM, Groq — running LLMs locally and via free APIs
- ✅ Audio AI: Whisper speech-to-text, pyannote speaker diarisation
- ✅ Multi-LLM orchestration: parallel API calls, cost/latency comparison
- ✅ PDF generation with ReportLab
- ✅ Production Gradio UI with full pipeline integration

---

## 👤 Author

**[Your Name]**
AlgoProfessor AI Internship 2026
📧 [your-email@domain.com]
🔗 [github.com/your-username/algoprofessor-rd-internship-2026](https://github.com)

---

## 📜 Commit Reference

```
[Day-21 to Day-25] Milestone 4: MeetScribe — AI Meeting Intelligence Tool
with Whisper transcription, pyannote diarisation, 3-way LLM comparison
(Groq Llama3/Claude 3.5/GPT-4o), PDF export and Gradio UI
```

---

*© 2026 AlgoProfessor AI R&D Solutions — Internship Batch 2026*
