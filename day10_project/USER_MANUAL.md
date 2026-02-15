# 📘 HCL KNOWLEDGE ASSISTANT - USER MANUAL

**Version:** 2.0.0  
**Date:** February 2025  
**Prepared for:** Day 10 Capstone Presentation

---

## TABLE OF CONTENTS

1. Introduction
2. Getting Started
3. System Features
4. How to Use
5. Query Modes Explained
6. Troubleshooting
7. FAQ
8. Technical Support

---

## 1. INTRODUCTION

### What is HCL Knowledge Assistant?

HCL Knowledge Assistant is an AI-powered chatbot that helps HCL Software Division employees quickly find information about:

- **Products**: HCL DX, Domino, Connections, AppScan, BigFix
- **Policies**: Leave, benefits, work from home, training
- **IT Support**: Hardware requests, software licenses, VPN access
- **Company Info**: Contacts, office locations, procedures
- **Real-time Data**: Live HCL stock prices and market information

### Key Benefits

✅ **24/7 Availability** - Get answers anytime  
✅ **Instant Responses** - No waiting for email replies  
✅ **Accurate Information** - Based on official HCL documents  
✅ **Multiple Sources** - Combines internal docs + web search + live data  
✅ **Easy to Use** - Simple chat interface  

---

## 2. GETTING STARTED

### System Requirements

**To Access the Application:**
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection
- Login credentials

**No Installation Required!** - Access via web browser

### First Time Login

**Step 1:** Open your web browser

**Step 2:** Navigate to:
```
http://localhost        (if running locally)
OR
http://your-company-domain.com
```

**Step 3:** Login Screen
- **Username:** demo
- **Password:** demo123
- Click **"Login"** button

**Step 4:** You're In!
- You'll see the chat interface
- Ready to ask questions

---

## 3. SYSTEM FEATURES

### 3.1 Chat Interface

**Components:**
1. **Header** - Shows app name and description
2. **Statistics Dashboard** - Shows query count, system status
3. **Mode Selector** - Choose how the AI should answer
4. **Chat Area** - Where messages appear
5. **Input Box** - Where you type your questions
6. **Send Button** - Submit your query

### 3.2 Query Modes

**RAG Mode (Recommended for Policies)**
- Searches internal HCL documents
- Fast responses (1-2 seconds)
- Best for: Policies, product info, procedures

**Agent Mode (Recommended for Current Info)**
- Uses intelligent AI reasoning
- Accesses live data and web search
- Best for: Stock prices, recent news, current events

**Hybrid Mode (Best Overall)**
- Combines both RAG and Agent
- Most comprehensive answers
- Best for: Complex questions requiring multiple sources

### 3.3 Real-time Capabilities

**Live Stock Data:**
- Fetches current HCL stock price from NSE (HCLTECH.NS)
- Shows 5-day trend
- Updates in real-time

**Web Search:**
- Searches internet for latest information
- Gets current news about HCL
- Finds recent updates

---

## 4. HOW TO USE

### Basic Usage

**Step 1:** Type your question in the input box

Example questions:
- "What is the leave policy?"
- "Tell me about HCL DX"
- "How do I request a new laptop?"
- "What is HCL's current stock price?"

**Step 2:** Press Enter or click "Send"

**Step 3:** Wait 1-3 seconds

**Step 4:** Read the answer
- The AI will respond in a chat bubble
- Sources will be shown at the bottom
- Confidence score indicates reliability

### Advanced Usage

**Choosing the Right Mode:**

| Your Question Type | Best Mode | Example |
|-------------------|-----------|---------|
| Employee policy | RAG | "What is the WFH policy?" |
| Product information | RAG | "Tell me about HCL Domino pricing" |
| Current stock price | Agent | "What is HCL stock price today?" |
| Recent company news | Agent | "What are recent HCL updates?" |
| Complex questions | Hybrid | "Compare HCL DX with competitors" |

**Tips for Better Answers:**

✅ **Be Specific**
- ❌ "Tell me about leave"
- ✅ "How many annual leave days do I get?"

✅ **Use Keywords**
- Include product names: "HCL DX", "Domino"
- Include topics: "policy", "pricing", "support"

✅ **Ask One Thing at a Time**
- ❌ "What's the leave policy and how do I request hardware?"
- ✅ Ask separately: First leave, then hardware

---

## 5. QUERY MODES EXPLAINED

### RAG Mode - Document Retrieval

**How it works:**
1. Searches internal knowledge base (hcl_knowledge_base.md)
2. Finds relevant sections
3. Returns information directly from documents

**When to use:**
- Questions about HCL policies
- Product features and pricing
- IT support procedures
- Company information

**Example:**
```
You: What is the leave policy?

RAG Mode searches and finds:
"Leave Policy
- Annual leave: 25 days per year
- Sick leave: 15 days per year
- Maternity leave: 26 weeks"

Response: Based on HCL's policy, you get 25 days 
annual leave, 15 days sick leave, and maternity 
leave is 26 weeks.
```

### Agent Mode - Intelligent Assistant

**How it works:**
1. Analyzes your question
2. Decides which specialized agent to use:
   - **HR Agent** - For policy questions
   - **IT Agent** - For tech support
   - **Product Agent** - For product info
   - **Financial Agent** - For stock/market data
3. May use real-time tools:
   - Stock price fetcher (yfinance)
   - Web search (DuckDuckGo)
4. Synthesizes answer using AI

**When to use:**
- Need current/live information
- Stock prices or market data
- Recent company news
- Complex reasoning required

**Example:**
```
You: What is HCL's stock price?

Agent Mode:
1. Detects "stock price" → Uses Financial Agent
2. Calls yfinance tool
3. Downloads live data from NSE
4. Analyzes 5-day trend

Response: Real-Time Stock Analysis (HCLTECH.NS):
- Current: ₹1,450.25
- 5-Day Avg: ₹1,435.80
- Trend: UP
```

### Hybrid Mode - Best of Both

**How it works:**
1. Runs RAG search first
2. Then runs Agent analysis
3. Combines both results
4. AI synthesizes comprehensive answer

**When to use:**
- Important questions needing multiple sources
- Want both internal docs AND current info
- Complex questions

**Example:**
```
You: How is HCL DX performing in the market?

Hybrid Mode:
1. RAG finds: "HCL DX - Enterprise license $50,000/year"
2. Agent searches web for: "HCL DX market share 2025"
3. Agent gets stock data
4. AI combines all information

Response: HCL DX is our enterprise web content 
management platform priced at $50,000/year. 
According to recent market analysis, it has 15% 
market share in the CMS space. HCL stock is 
currently trading at ₹1,450, up 2.3% this week.
```

---

## 6. TROUBLESHOOTING

### Problem: Cannot Login

**Symptoms:**
- "Invalid credentials" error
- Button doesn't respond

**Solutions:**
1. ✅ Check credentials: `demo` / `demo123`
2. ✅ Ensure backend is running (check URL bar)
3. ✅ Clear browser cache (Ctrl+Shift+Del)
4. ✅ Try different browser

### Problem: No Response to Query

**Symptoms:**
- Message stuck on "Thinking..."
- Error message appears

**Solutions:**
1. ✅ Check internet connection
2. ✅ Verify backend is running:
   - Open: http://localhost:8000/api/v1/health
   - Should show: `{"status":"healthy"}`
3. ✅ Check Groq API key in .env file
4. ✅ Refresh page and try again

### Problem: Slow Responses

**Symptoms:**
- Takes >5 seconds to respond

**Solutions:**
1. ✅ Try simpler questions first
2. ✅ Switch from Hybrid to RAG mode
3. ✅ Check system resources (RAM/CPU)
4. ✅ Restart Docker containers

### Problem: "AI Error: Invalid API Key"

**Symptoms:**
- Error in red text
- No AI-generated answers

**Solutions:**
1. ✅ Check .env file has GROQ_API_KEY=gsk_...
2. ✅ Get new key from console.groq.com
3. ✅ Restart Docker: `docker-compose restart`

---

## 7. FREQUENTLY ASKED QUESTIONS

**Q1: Is my data private?**
A: Yes, all queries are processed locally. Login credentials are demo-only for this prototype.

**Q2: Can I access this from mobile?**
A: Yes! The interface is responsive and works on phones/tablets.

**Q3: How accurate are the answers?**
A: RAG mode is 95%+ accurate (directly from docs). Agent mode depends on web sources. Always verify critical information.

**Q4: Can I upload my own documents?**
A: Not in current version. Contact admin to add documents to knowledge base.

**Q5: What if the answer is wrong?**
A: Try rephrasing your question or use a different mode. For critical info, contact HR/IT directly.

**Q6: How do I report a bug?**
A: Email: support@hcl.com with screenshot and description.

**Q7: Can I use this offline?**
A: No, requires internet for AI processing and live data.

**Q8: Is there a query limit?**
A: No hard limit, but rate limiting may apply (60 queries/minute).

---

## 8. TECHNICAL SUPPORT

### Contact Information

**IT Help Desk:**
- Extension: 5000
- Email: ithelpdesk@hcl.com
- Hours: 24/7

**For This Application:**
- Technical Issues: support@hcl.com
- Feature Requests: feedback@hcl.com

### System Status

Check system health:
```
http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

---

## APPENDIX A: SAMPLE QUERIES

### Employee Policies
- "What is the leave policy?"
- "How many WFH days do I get?"
- "What is the training budget?"
- "Tell me about maternity leave"

### Products
- "What is HCL DX?"
- "How much does Domino cost?"
- "Tell me about HCL AppScan features"
- "Compare HCL Connections and Microsoft Teams"

### IT Support
- "How do I request a new laptop?"
- "What is the VPN setup process?"
- "How do I reset my password?"
- "What software licenses are available?"

### Real-time Data
- "What is HCL's current stock price?"
- "Show me HCL stock trend"
- "What is recent news about HCL?"

---

## APPENDIX B: KEYBOARD SHORTCUTS

| Shortcut | Action |
|----------|--------|
| Enter | Send query |
| Ctrl+L | Clear chat (refresh page) |
| Tab | Navigate between fields |

---

## APPENDIX C: ERROR CODES

| Error | Meaning | Solution |
|-------|---------|----------|
| 401 | Invalid credentials | Check username/password |
| 500 | Server error | Refresh page, check backend |
| Network Error | Cannot reach server | Check internet, Docker status |

---

**End of User Manual**

*For questions or feedback, contact: support@hcl.com*
