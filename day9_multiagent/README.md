# 🤖 Day 9: Multi-Agent Content Creation System

A production-ready multi-agent system that creates high-quality content through collaboration between specialized AI agents. Features real-time Google search, quality control, and automated workflows.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Agent Descriptions](#agent-descriptions)
- [Workflow Process](#workflow-process)
- [File Structure](#file-structure)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This multi-agent system demonstrates advanced AI orchestration where specialized agents work together to:

1. **Research** → Gather real-time information from Google
2. **Write** → Create engaging, well-structured content
3. **Review** → Ensure quality and accuracy
4. **Revise** → Improve based on feedback
5. **Deliver** → Export publication-ready content

**Key Technologies:**
- LLM: Groq (Llama 3.3 70B)
- Search: DuckDuckGo (no API key needed)
- Web Scraping: BeautifulSoup4
- Framework: Custom + CrewAI-ready

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────┐
│         WORKFLOW ORCHESTRATOR               │
│   (Plans, Coordinates, Monitors)            │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌─────▼──────┐
│  RESEARCHER │  │   WRITER   │
│  - Google   │  │  - Creates │
│  - Search   │  │  - Drafts  │
│  - Scrape   │  │  - Formats │
└──────┬──────┘  └─────┬──────┘
       │                │
       └────────┬───────┘
                │
         ┌──────▼──────┐
         │  REVIEWER   │
         │  - Checks   │
         │  - Scores   │
         │  - Feedback │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │   OUTPUT    │
         │ Final Files │
         └─────────────┘
```

### Communication Flow

```
User Request
    ↓
Orchestrator (plans workflow)
    ↓
Researcher (gathers data)
    ↓ [research.json]
Writer (creates content)
    ↓ [draft.md]
Reviewer (quality check)
    ↓ [feedback]
Writer (revises) ←─ [loop if needed]
    ↓
Reviewer (approves)
    ↓
Orchestrator (exports)
    ↓
Final Content ✅
```

---

## ✨ Features

### Core Capabilities

- ✅ **Real-time Research**: Google search + web scraping
- ✅ **Multi-Agent Collaboration**: 4 specialized agents
- ✅ **Quality Control**: Automated review and scoring
- ✅ **Feedback Loops**: Iterative improvement (up to 3 cycles)
- ✅ **Citation Management**: Automatic source formatting
- ✅ **Multiple Formats**: Blog posts, articles, reports
- ✅ **Tone Control**: Professional, casual, technical, academic
- ✅ **Message Bus**: Complete communication logging
- ✅ **Error Handling**: Graceful failure recovery
- ✅ **Export System**: JSON + Markdown outputs

### Advanced Features

- 📊 **Workflow Statistics**: Track performance metrics
- 🔄 **Revision Cycles**: Automatic content improvement
- 📝 **Quality Scoring**: 5-metric evaluation system
- 💬 **Agent Communication**: Tracked message passing
- 🎯 **Research Depth Control**: Quick/Standard/Deep modes
- 📁 **Complete Logging**: Full workflow history

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Groq API key ([Get one free](https://console.groq.com))

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Setup Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API key
# GROQ_API_KEY=your_key_here
```

### Step 3: Verify Installation

```bash
python agent_definitions.py
```

You should see agent configurations printed successfully.

---

## 🚀 Quick Start

### Run the Interactive System

```bash
python multi_agent_system.py
```

### Quick Example

```bash
# When prompted, choose:
# 1. Create Blog Post
# Enter topic: "AI Trends in 2025"
# Accept defaults or customize
# Wait 2-5 minutes
# Check outputs/ folder for results
```

### Output Location

```
outputs/
├── 20250212143020_content.md       # Final content
├── 20250212143020_metadata.json    # Workflow stats
├── 20250212143020_workflow.json    # Detailed logs
└── 20250212143020_messages.json    # Agent communications
```

---

## 📝 Usage Examples

### Example 1: Create a Blog Post

```python
from workflow_orchestrator import WorkflowOrchestrator
import os

api_key = os.getenv("GROQ_API_KEY")
orchestrator = WorkflowOrchestrator(api_key)

result = orchestrator.create_content_workflow(
    topic="The Future of Electric Vehicles",
    content_type="blog_post",
    word_count=1000,
    tone="professional",
    research_depth="standard"
)

# Export to files
orchestrator.export_workflow(result)
```

### Example 2: Research-Heavy Article

```python
result = orchestrator.create_content_workflow(
    topic="Quantum Computing Applications",
    content_type="article",
    word_count=2000,
    tone="technical",
    research_depth="deep"  # More sources, deeper scraping
)
```

### Example 3: Quick Casual Content

```python
result = orchestrator.create_content_workflow(
    topic="10 Productivity Tips for Developers",
    content_type="blog_post",
    word_count=800,
    tone="casual",
    research_depth="quick"  # Faster, fewer sources
)
```

### Example 4: Use Individual Agents

```python
from research_agent import ResearchAgent

# Research only
researcher = ResearchAgent(api_key)
report = researcher.research(
    topic="Machine Learning in Healthcare",
    depth="deep"
)

researcher.export_research("/path/to/research.json")
```

---

## 🤖 Agent Descriptions

### 1. Research Agent 🔬

**Role:** Senior Research Analyst

**Capabilities:**
- Google search (via DuckDuckGo)
- Web page scraping
- Fact extraction
- Source verification
- Citation formatting

**Configuration:**
- Temperature: 0.3 (factual accuracy)
- Max iterations: 5
- Sources: Up to 15 (deep mode)

**Example Output:**
```json
{
  "topic": "AI in Healthcare",
  "summary": "Comprehensive research summary...",
  "facts": [
    "AI detects cancer with 95% accuracy",
    "ML reduces diagnostic time by 40%"
  ],
  "sources": [
    "Smith, J. (2025). AI in Healthcare. Retrieved from..."
  ]
}
```

### 2. Writer Agent ✍️

**Role:** Expert Content Writer

**Capabilities:**
- Content outlining
- Multi-format writing (blog, article, report)
- Tone adaptation
- Citation integration
- Content revision

**Configuration:**
- Temperature: 0.7 (creativity)
- Max iterations: 3
- Word count: Customizable

**Output Formats:**
- Blog posts
- Articles
- Research reports
- Technical guides
- Academic papers

### 3. Reviewer Agent 👁️

**Role:** Quality Assurance Specialist

**Capabilities:**
- 5-metric quality scoring
- Factual verification
- Structure analysis
- Citation checking
- Actionable feedback

**Scoring System:**
1. **Quality** (word count, headings, paragraphs)
2. **Factual** (research fact usage)
3. **Structure** (intro, conclusion, sections)
4. **Citations** (references, sources)
5. **Style** (sentence length, variety, passive voice)

**Approval Threshold:** 75% overall score

### 4. Workflow Orchestrator 🎯

**Role:** Project Manager

**Capabilities:**
- Workflow planning
- Agent coordination
- Message routing
- Error handling
- Result compilation

**Features:**
- Max 3 revision cycles
- Timeout protection (10 min)
- Complete logging
- Statistics tracking

---

## 🔄 Workflow Process

### Standard Workflow (Sequential)

```
1. RESEARCH (30-60s)
   ├─ Google search
   ├─ Web scraping
   ├─ Fact extraction
   └─ Citation formatting

2. WRITING (60-120s)
   ├─ Outline creation
   ├─ Content generation
   ├─ Citation integration
   └─ Document formatting

3. REVIEW (20-40s)
   ├─ Quality checks
   ├─ Fact verification
   ├─ Structure analysis
   ├─ Citation verification
   └─ Feedback generation

4. REVISION (if needed, 40-80s)
   ├─ Apply feedback
   ├─ Improve content
   └─ Re-review

5. EXPORT (5-10s)
   ├─ Save content
   ├─ Save metadata
   ├─ Save workflow log
   └─ Save messages
```

### Revision Loop

```
Write → Review → Approved? ─Yes→ Export
           ↓                         
          No (Score < 75%)            
           ↓                         
        Revise ──┘                   
(Max 3 cycles)
```

---

## 📁 File Structure

```
day9_multi_agent/
├── agent_definitions.py          # Agent roles & configs
├── communication_protocol.py     # Message bus & routing
├── custom_tools.py               # Google search & scraping
├── research_agent.py             # Research specialist
├── writer_agent.py               # Content creator
├── reviewer_agent.py             # Quality controller
├── workflow_orchestrator.py      # Workflow manager
├── multi_agent_system.py         # Main executable
├── requirements.txt              # Dependencies
├── .env.example                  # Environment template
├── README.md                     # This file
└── outputs/                      # Generated content
    ├── *_content.md
    ├── *_metadata.json
    ├── *_workflow.json
    └── *_messages.json
```

---

## 🔧 Advanced Usage

### Custom Agent Configuration

```python
from agent_definitions import AgentConfig, AgentRole

custom_config = AgentConfig(
    role=AgentRole.WRITER,
    name="Technical Writer",
    goal="Create detailed technical documentation",
    backstory="Expert in software documentation",
    temperature=0.5,
    max_iterations=5
)
```

### Message Bus Inspection

```python
from communication_protocol import MessageBus

bus = MessageBus()
# ... after workflow ...

# Get statistics
stats = bus.get_statistics()
print(f"Total messages: {stats['total_messages_sent']}")

# Get conversation
messages = bus.get_conversation("Researcher", "Writer")
for msg in messages:
    print(f"{msg.from_agent} → {msg.to_agent}: {msg.message_type}")
```

### Custom Workflow

```python
# Create your own workflow
orchestrator = WorkflowOrchestrator(api_key)

# Step 1: Research
research = orchestrator.researcher.research("Topic", "deep")

# Step 2: Write
content = orchestrator.writer.write_content(
    "Topic", research, "blog_post", 1500, "casual"
)

# Step 3: Review
review = orchestrator.reviewer.review_content(
    content["content"], research, "blog_post"
)

# Step 4: Export
orchestrator.writer.export_content("output.md", content["content"])
```

---

## 🐛 Troubleshooting

### Common Issues

**1. API Key Error**
```
❌ Error: GROQ_API_KEY not found
```
**Solution:** Create `.env` file with your API key

**2. Import Errors**
```
ModuleNotFoundError: No module named 'duckduckgo_search'
```
**Solution:** `pip install -r requirements.txt`

**3. Search Fails**
```
❌ Search error: timeout
```
**Solution:** Check internet connection, retry with `research_depth="quick"`

**4. Low Quality Scores**
```
⚠️ Content needs revision (Score: 65%)
```
**Solution:** System will auto-revise (up to 3 times). If still low, review suggestions in output JSON.

### Debug Mode

```python
# Enable verbose logging
orchestrator = WorkflowOrchestrator(api_key)
orchestrator.researcher.config.verbose = True
orchestrator.writer.config.verbose = True
orchestrator.reviewer.config.verbose = True
```

### Performance Tips

1. **Use "quick" research** for faster results (fewer sources)
2. **Lower word count** for faster generation
3. **Increase timeout** for deep research: `orchestrator.timeout_seconds = 1200`
4. **Disable revision** by setting `max_revision_cycles = 0`

---

## 📊 System Metrics

Typical workflow times (standard depth, 1500 words):

- Research: 30-60 seconds
- Writing: 60-120 seconds  
- Review: 20-40 seconds
- Revision: 40-80 seconds (if needed)
- **Total: 2-5 minutes**

Quality scores achieved:
- 80%+ : Excellent (usually approved first review)
- 75-79%: Good (may need minor revision)
- 70-74%: Acceptable (likely needs revision)
- <70% : Needs improvement (will auto-revise)

---

## 🎓 Learning Outcomes

By using this system, you'll understand:

- ✅ Multi-agent architecture design
- ✅ Inter-agent communication protocols
- ✅ Workflow orchestration patterns
- ✅ Quality control automation
- ✅ Feedback loop implementation
- ✅ Real-world tool integration
- ✅ Error handling in distributed systems
- ✅ Message bus patterns
- ✅ Agent specialization benefits

---

## 📜 License

MIT License - feel free to use and modify!

---

## 🙏 Acknowledgments

- Groq for fast LLM inference
- DuckDuckGo for free search API
- BeautifulSoup for web scraping
- CrewAI for inspiration

---

## 💡 Next Steps

1. **Try different topics**: Test various content types
2. **Customize agents**: Modify roles and capabilities
3. **Add more agents**: Create specialist agents (SEO, Fact-checker)
4. **Integrate CrewAI**: Port to CrewAI framework
5. **Add human feedback**: Interactive approval loops
6. **Build UI**: Web interface for easier use

---

**Built with ❤️ for Day 9 of the AI Agent Learning Journey**

For questions or issues, check the troubleshooting section or review the code comments!
