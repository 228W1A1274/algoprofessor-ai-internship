"""
Research Agent - Day 9 Multi-Agent System
Specialized agent for gathering information from the internet
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq
import json
from datetime import datetime

from agent_definitions import BaseAgent, RESEARCHER_CONFIG, AgentStatus
from custom_tools import GoogleSearchTool, WebScraperTool, ContentAnalyzerTool, CitationFormatter

load_dotenv()


class ResearchAgent(BaseAgent):
    """
    Specialized Research Agent
    
    Capabilities:
    - Google search for current information
    - Web scraping for detailed content
    - Source verification
    - Citation formatting
    - Fact extraction
    """
    
    def __init__(self, api_key: str):
        super().__init__(RESEARCHER_CONFIG)
        
        # Initialize LLM
        self.client = Groq(api_key=api_key)
        self.model = self.config.model
        
        # Initialize tools
        self.search_tool = GoogleSearchTool(max_results=10)
        self.scraper_tool = WebScraperTool()
        self.analyzer_tool = ContentAnalyzerTool()
        
        # Research storage
        self.research_data = {
            "search_results": [],
            "scraped_content": [],
            "facts": [],
            "sources": []
        }
    
    def research(self, topic: str, depth: str = "standard") -> Dict[str, Any]:
        """
        Conduct research on a topic
        
        Args:
            topic: Research topic
            depth: Research depth ("quick", "standard", "deep")
        
        Returns:
            Research findings dictionary
        """
        try:
            self.set_status(AgentStatus.WORKING)
            self.current_task = f"Research: {topic}"
            
            print(f"\n{'='*60}")
            print(f"🔬 RESEARCH AGENT: Starting research on '{topic}'")
            print(f"   Depth: {depth}")
            print(f"{'='*60}\n")
            
            # Step 1: Google Search
            print("📝 Step 1: Conducting Google search...")
            search_results = self._google_search(topic, depth)
            
            # Step 2: Analyze search results and scrape top sources
            print("\n📝 Step 2: Analyzing and scraping top sources...")
            scraped_data = self._scrape_top_sources(search_results, depth)
            
            # Step 3: Extract facts and key information
            print("\n📝 Step 3: Extracting facts and key information...")
            facts = self._extract_facts(scraped_data)
            
            # Step 4: Generate research summary using LLM
            print("\n📝 Step 4: Generating research summary...")
            summary = self._generate_summary(topic, search_results, facts)
            
            # Step 5: Format citations
            print("\n📝 Step 5: Formatting citations...")
            citations = self._format_citations(search_results)
            
            # Compile research report
            report = {
                "topic": topic,
                "depth": depth,
                "summary": summary,
                "facts": facts,
                "sources": citations,
                "search_results_count": len(search_results),
                "sources_scraped": len(scraped_data),
                "researched_at": datetime.now().isoformat(),
                "raw_data": {
                    "search_results": search_results,
                    "scraped_content": scraped_data
                }
            }
            
            self.research_data = report
            self.set_status(AgentStatus.COMPLETED)
            
            print(f"\n✅ Research complete!")
            print(f"   - Found {len(search_results)} sources")
            print(f"   - Scraped {len(scraped_data)} websites")
            print(f"   - Extracted {len(facts)} facts")
            
            return report
            
        except Exception as e:
            self.handle_error(e, "Research task")
            return {"error": str(e)}
    
    def _google_search(self, topic: str, depth: str) -> List[Dict]:
        """Perform Google search based on depth"""
        max_results = {
            "quick": 5,
            "standard": 10,
            "deep": 15
        }.get(depth, 10)
        
        results = self.search_tool.search(topic, max_results=max_results)
        self.log_action("google_search", {
            "topic": topic,
            "results_count": len(results)
        })
        
        return results
    
    def _scrape_top_sources(self, search_results: List[Dict], depth: str) -> List[Dict]:
        """Scrape content from top search results"""
        scrape_count = {
            "quick": 2,
            "standard": 3,
            "deep": 5
        }.get(depth, 3)
        
        scraped_data = []
        for result in search_results[:scrape_count]:
            url = result.get('url', '')
            if url:
                scraped = self.scraper_tool.scrape_url(url)
                if 'error' not in scraped:
                    scraped_data.append(scraped)
        
        self.log_action("web_scraping", {
            "urls_scraped": len(scraped_data)
        })
        
        return scraped_data
    
    def _extract_facts(self, scraped_data: List[Dict]) -> List[str]:
        """Extract facts from scraped content"""
        all_facts = []
        
        for data in scraped_data:
            text = data.get('text', '')
            if text:
                facts = self.analyzer_tool.extract_facts(text)
                all_facts.extend(facts)
        
        # Remove duplicates and limit
        unique_facts = list(dict.fromkeys(all_facts))
        
        return unique_facts[:15]  # Top 15 facts
    
    def _generate_summary(self, topic: str, search_results: List[Dict], facts: List[str]) -> str:
        """Generate research summary using LLM"""
        
        # Prepare context
        sources_text = "\n\n".join([
            f"Source {i+1}: {r.get('title', '')}\n{r.get('snippet', '')}"
            for i, r in enumerate(search_results[:5])
        ])
        
        facts_text = "\n".join([f"- {fact}" for fact in facts[:10]])
        
        prompt = f"""Based on the following research on "{topic}", create a comprehensive summary.

Search Results:
{sources_text}

Key Facts Extracted:
{facts_text}

Please provide:
1. A 2-3 paragraph summary of the main findings
2. Key insights and trends
3. Important statistics or data points

Keep it factual and well-structured."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research analyst. Provide clear, factual summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=1000
            )
            
            summary = response.choices[0].message.content
            return summary
            
        except Exception as e:
            print(f"⚠️ LLM summary generation failed: {e}")
            return "Summary generation failed. See raw research data."
    
    def _format_citations(self, search_results: List[Dict]) -> List[str]:
        """Format citations for sources"""
        citations = []
        
        for result in search_results[:10]:
            source = {
                'title': result.get('title', 'Untitled'),
                'url': result.get('url', ''),
                'date': datetime.now().strftime('%Y, %B %d')
            }
            citation = CitationFormatter.format_apa(source)
            citations.append(citation)
        
        return citations
    
    def quick_search(self, query: str) -> str:
        """
        Quick search without full research
        
        Args:
            query: Search query
        
        Returns:
            Quick summary of search results
        """
        results = self.search_tool.search(query, max_results=5)
        return self.analyzer_tool.summarize_sources(results)
    
    def verify_fact(self, fact: str) -> Dict[str, Any]:
        """
        Verify a fact by searching for supporting evidence
        
        Args:
            fact: Fact to verify
        
        Returns:
            Verification results
        """
        print(f"🔍 Verifying fact: {fact}")
        
        # Search for the fact
        results = self.search_tool.search(fact, max_results=5)
        
        # Count mentions
        mentions = 0
        supporting_sources = []
        
        for result in results:
            snippet = result.get('snippet', '').lower()
            if any(word in snippet for word in fact.lower().split()):
                mentions += 1
                supporting_sources.append(result.get('url', ''))
        
        verification = {
            "fact": fact,
            "verified": mentions >= 2,
            "confidence": min(mentions / 5.0, 1.0),
            "supporting_sources": supporting_sources,
            "total_results": len(results)
        }
        
        print(f"✅ Verification: {'VERIFIED' if verification['verified'] else 'UNVERIFIED'}")
        print(f"   Confidence: {verification['confidence']*100:.0f}%")
        
        return verification
    
    def export_research(self, filepath: str) -> None:
        """
        Export research data to file
        
        Args:
            filepath: Path to save research
        """
        with open(filepath, 'w') as f:
            json.dump(self.research_data, f, indent=2)
        
        print(f"📄 Research exported to {filepath}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n🔬 RESEARCH AGENT - Testing\n")
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in .env file")
        print("\n💡 Create a .env file with:")
        print("GROQ_API_KEY=your_api_key_here")
        exit(1)
    
    # Create research agent
    researcher = ResearchAgent(api_key)
    
    # Test 1: Quick search
    print("="*60)
    print("TEST 1: Quick Search")
    print("="*60)
    
    quick_result = researcher.quick_search("latest AI developments 2025")
    print(f"\nQuick Search Result:\n{quick_result}")
    
    # Test 2: Full research
    print("\n\n" + "="*60)
    print("TEST 2: Full Research")
    print("="*60)
    
    topic = "Artificial Intelligence trends in 2025"
    report = researcher.research(topic, depth="standard")
    
    print(f"\n{'='*60}")
    print("RESEARCH REPORT")
    print(f"{'='*60}\n")
    print(f"Topic: {report.get('topic', 'N/A')}")
    print(f"\nSummary:\n{report.get('summary', 'N/A')}")
    print(f"\nKey Facts ({len(report.get('facts', []))}):")
    for i, fact in enumerate(report.get('facts', [])[:5], 1):
        print(f"{i}. {fact}")
    
    print(f"\nSources ({len(report.get('sources', []))}):")
    for i, source in enumerate(report.get('sources', [])[:3], 1):
        print(f"{i}. {source}")
    
    # Test 3: Export research
    print("\n\n" + "="*60)
    print("TEST 3: Export Research")
    print("="*60 + "\n")
    
    os.makedirs("./outputs", exist_ok=True)
    researcher.export_research("./outputs/research_report.json")
    
    print("\n✅ All tests complete!")