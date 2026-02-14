# 🚀 Quick Start Guide - Day 9 Multi-Agent System

Get up and running in 5 minutes!

## ⚡ Fast Setup

### 1. Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

### 2. Setup API Key (1 minute)

```bash
# Copy template
cp .env.example .env

# Edit .env file and add your key:
# GROQ_API_KEY=gsk_your_key_here
```

**Get a FREE API key:** https://console.groq.com

### 3. Test System (1 minute)

```bash
python test_system.py
```

You should see:
```
🎉 ALL TESTS PASSED! System is ready to use!
```

### 4. Create Your First Content (2 minutes)

```bash
python multi_agent_system.py
```

Choose option 1 (Blog Post), enter a topic like "Future of AI", and wait 2-3 minutes!

---

## 📁 Where Are My Files?

All generated content goes to:
```
outputs/
├── YYYYMMDDHHMMSS_content.md       ← Your article!
├── YYYYMMDDHHMMSS_metadata.json    ← Stats
├── YYYYMMDDHHMMSS_workflow.json    ← Process log
└── YYYYMMDDHHMMSS_messages.json    ← Agent chat log
```

---

## 🎯 What Each File Does

| File | Purpose |
|------|---------|
| `multi_agent_system.py` | **Main program** - Run this! |
| `test_system.py` | Test everything works |
| `agent_definitions.py` | Agent roles & configs |
| `workflow_orchestrator.py` | Manages the workflow |
| `research_agent.py` | Google search agent |
| `writer_agent.py` | Content creator |
| `reviewer_agent.py` | Quality checker |
| `communication_protocol.py` | Agent messaging |
| `custom_tools.py` | Search & scraping tools |

---

## 🔧 Quick Examples

### Example 1: Blog Post

```python
from workflow_orchestrator import WorkflowOrchestrator
import os

api_key = os.getenv("GROQ_API_KEY")
orchestrator = WorkflowOrchestrator(api_key)

result = orchestrator.create_content_workflow(
    topic="10 Python Tips for Beginners",
    content_type="blog_post",
    word_count=1000,
    tone="casual"
)
```

### Example 2: Research Report

```python
result = orchestrator.create_content_workflow(
    topic="Climate Change Impact on Agriculture",
    content_type="research_report",
    word_count=2000,
    tone="academic",
    research_depth="deep"  # More thorough research
)
```

---

## ⚠️ Troubleshooting

### Problem: "GROQ_API_KEY not found"

**Solution:**
1. Copy `.env.example` to `.env`
2. Add your API key: `GROQ_API_KEY=your_key`
3. Get key from: https://console.groq.com

### Problem: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: Search fails

**Solution:**
- Check internet connection
- Try `research_depth="quick"` for faster/simpler search

---

## 📊 Workflow Process

```
1. User enters topic
        ↓
2. Research Agent searches Google (30-60s)
        ↓
3. Writer Agent creates content (60-120s)
        ↓
4. Reviewer Agent checks quality (20-40s)
        ↓
5. If score < 75%, Writer revises (40-80s)
        ↓
6. Export final content ✅
```

**Total Time:** 2-5 minutes

---

## 🎓 What You'll Get

✅ **Professional Content**: Blog posts, articles, reports  
✅ **Real Research**: Live Google search results  
✅ **Quality Control**: Automated review and scoring  
✅ **Proper Citations**: APA-formatted sources  
✅ **Complete Logs**: Full workflow history  

---

## 💡 Pro Tips

1. **Faster results**: Use `research_depth="quick"`
2. **Better quality**: Use `research_depth="deep"`
3. **Technical content**: Set `tone="technical"`
4. **Casual blog**: Set `tone="casual"`
5. **Check logs**: Review `*_workflow.json` for details

---

## 📖 Full Documentation

For complete details, see [README.md](README.md)

---

## 🎯 Next Steps

1. ✅ Run `test_system.py` to verify setup
2. ✅ Run `multi_agent_system.py` to create content
3. ✅ Check `outputs/` folder for your files
4. ✅ Read [README.md](README.md) for advanced usage
5. ✅ Customize agents in `agent_definitions.py`

---

**Ready? Let's go!**

```bash
python multi_agent_system.py
```

🎉 **Happy creating!**
