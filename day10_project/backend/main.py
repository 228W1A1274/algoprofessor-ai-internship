"""
HCL Knowledge Assistant - Main Application
Day 10 Capstone Project - PRODUCTION VERSION
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import sys
from dotenv import load_dotenv
from loguru import logger

# --- LIBRARIES ---
import yfinance as yf
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO")
load_dotenv()

app = FastAPI(title="HCL Knowledge Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

# =============================================================================
# DATA MODELS
# =============================================================================

class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    mode_used: str

# =============================================================================
# TOOLS
# =============================================================================

class FinancialDataTool:
    def get_hcl_stock_data(self):
        try:
            logger.info("📉 Downloading Live HCL Dataset...")
            stock = yf.Ticker("HCLTECH.NS")
            df = stock.history(period="5d")
            if df.empty: return "No data found."
            
            current = df['Close'].iloc[-1]
            avg = df['Close'].mean()
            trend = "UP" if df['Close'].iloc[-1] > df['Close'].iloc[0] else "DOWN"
            
            return (f"Real-Time Stock Analysis (HCLTECH.NS):\n"
                    f"- Current: ₹{current:.2f}\n"
                    f"- 5-Day Avg: ₹{avg:.2f}\n"
                    f"- Trend: {trend}")
        except Exception as e:
            logger.error(f"Stock tool error: {e}")
            return "Financial data unavailable."

# =============================================================================
# AI ENGINE
# =============================================================================

class Brain:
    def __init__(self):
        key = os.getenv("GROQ_API_KEY", "")
        if not key.startswith("gsk_"):
            logger.warning("⚠️ Invalid GROQ_API_KEY. AI will use fallback mode.")
            self.llm = None
        else:
            self.llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=key)

    def synthesize(self, query: str, context: str):
        if not self.llm:
            return "AI Error: Invalid API Key. Please check .env file."
        try:
            prompt = ChatPromptTemplate.from_template("Answer based on context: {context}. Question: {query}")
            chain = prompt | self.llm
            return chain.invoke({"query": query, "context": context}).content
        except Exception as e:
            return f"AI Generation Error: {str(e)}"

class RAGSystem:
    def __init__(self):
        try:
            with open("data/hcl_knowledge_base.md", "r") as f:
                self.data = f.read().split("\n\n")
        except: self.data = []
    
    def search(self, query):
        return [c for c in self.data if any(w in c.lower() for w in query.lower().split())][:3]

class AgentSystem:
    def __init__(self):
        # We wrap this in try/except so the server starts even if Search fails
        try:
            self.web_tool = DuckDuckGoSearchRun()
        except:
            self.web_tool = None
            logger.error("⚠️ Web Search Tool failed to load. Update requirements.")
            
        self.finance_tool = FinancialDataTool()

    def run_agent(self, query):
        results = {}
        if any(w in query.lower() for w in ["stock", "price", "market"]):
            results["Financial Dataset"] = self.finance_tool.get_hcl_stock_data()
        elif "current" in query.lower() or "news" in query.lower():
            if self.web_tool:
                try:
                    results["Live Web Search"] = self.web_tool.run(f"{query} HCL Technologies")
                except:
                    results["Live Web Search"] = "Search failed."
        return results

# =============================================================================
# API ROUTES
# =============================================================================

rag = RAGSystem()
agent = AgentSystem()
brain = Brain()

@app.post("/api/v1/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    return {"access_token": "valid_token", "token_type": "bearer"}

@app.post("/api/v1/query")
async def query(request: QueryRequest):
    context = ""
    sources = []
    
    # 1. RAG
    for r in rag.search(request.query):
        context += f"\n[INTERNAL]: {r}\n"
        sources.append({"source": "Internal KB", "preview": r[:50]})

    # 2. AGENT
    agent_results = agent.run_agent(request.query)
    for k, v in agent_results.items():
        context += f"\n[{k}]: {v}\n"
        sources.append({"source": k, "preview": str(v)[:50]})

    # 3. SYNTHESIZE
    answer = brain.synthesize(request.query, context)
    return {"answer": answer, "sources": sources, "mode_used": "hybrid"}

# =============================================================================
# MISSING UTILITY ROUTES (Paste this at the bottom of backend/main.py)
# =============================================================================

@app.get("/api/v1/health")
async def health_check():
    """Docker Health Check"""
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/api/v1/stats")
async def get_stats():
    """Frontend Stats"""
    return {
        "total_queries": 12,  # Mock data for demo
        "knowledge_base_size": 5,
        "status": "online"
    }