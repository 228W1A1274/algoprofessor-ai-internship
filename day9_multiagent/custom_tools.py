"""
Custom Tools - Day 9 Multi-Agent System
Real-world tools for research, analysis, and content creation
"""

from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time


class GoogleSearchTool:
    """
    Real-time search with multiple fallback options
    Includes rate limit handling and retry logic
    """
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.search_history: List[Dict] = []
        self.last_search_time = 0
        self.min_search_delay = 2  # seconds between searches
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Search for information using multiple fallback methods
        
        Args:
            query: Search query
            max_results: Maximum number of results (default: self.max_results)
        
        Returns:
            List of search results with title, url, snippet
        """
        max_results = max_results or self.max_results
        
        # Try DuckDuckGo first
        results = self._search_duckduckgo(query, max_results)
        
        # If DuckDuckGo fails, try Wikipedia API
        if not results:
            print("⚠️ DuckDuckGo failed, trying Wikipedia...")
            results = self._search_wikipedia(query, max_results)
        
        # If Wikipedia fails, use mock data for development
        if not results:
            print("⚠️ Wikipedia failed, using simulated results...")
            results = self._generate_mock_results(query, max_results)
        
        # Log search
        self.search_history.append({
            "query": query,
            "results_count": len(results),
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"✅ Found {len(results)} results")
        return results
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Try searching with DuckDuckGo"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_search_time
            if time_since_last < self.min_search_delay:
                time.sleep(self.min_search_delay - time_since_last)
            
            print(f"🔍 Searching with DuckDuckGo for: '{query}'")
            
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=max_results
                ))
            
            self.last_search_time = time.time()
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_result = {
                    "position": i,
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            print(f"⚠️ DuckDuckGo error: {e}")
            return []
    
    def _search_wikipedia(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Fallback: Search Wikipedia API"""
        try:
            print(f"🔍 Searching Wikipedia for: '{query}'")
            
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "opensearch",
                "search": query,
                "limit": max_results,
                "format": "json"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse Wikipedia response
            titles = data[1] if len(data) > 1 else []
            descriptions = data[2] if len(data) > 2 else []
            urls = data[3] if len(data) > 3 else []
            
            formatted_results = []
            for i, (title, desc, url) in enumerate(zip(titles, descriptions, urls), 1):
                formatted_result = {
                    "position": i,
                    "title": title,
                    "url": url,
                    "snippet": desc,
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            print(f"⚠️ Wikipedia error: {e}")
            return []
    
    def _generate_mock_results(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Generate mock search results for development/testing"""
        
        # Create realistic mock data based on query
        mock_data = {
            "ai": [
                {
                    "title": "Artificial Intelligence - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                    "snippet": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals."
                },
                {
                    "title": "What Is Artificial Intelligence? | IBM",
                    "url": "https://www.ibm.com/topics/artificial-intelligence",
                    "snippet": "Artificial intelligence leverages computers and machines to mimic the problem-solving and decision-making capabilities of the human mind."
                },
                {
                    "title": "AI News & Insights | MIT Technology Review",
                    "url": "https://www.technologyreview.com/topic/artificial-intelligence/",
                    "snippet": "The latest news, analysis and insights on artificial intelligence and machine learning from MIT Technology Review."
                }
            ],
            "automation": [
                {
                    "title": "Automation - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Automation",
                    "snippet": "Automation describes a wide range of technologies that reduce human intervention in processes, namely by predetermining decision criteria."
                },
                {
                    "title": "What is Automation? | IBM",
                    "url": "https://www.ibm.com/topics/automation",
                    "snippet": "Automation is the use of technology to perform tasks with reduced human assistance."
                }
            ],
            "default": [
                {
                    "title": f"Understanding {query.title()}",
                    "url": f"https://example.com/{query.replace(' ', '-')}",
                    "snippet": f"A comprehensive guide to {query}, covering key concepts, trends, and applications in modern technology."
                },
                {
                    "title": f"{query.title()} - Latest Developments",
                    "url": f"https://example.com/{query.replace(' ', '-')}-news",
                    "snippet": f"Explore the latest developments and innovations in {query}. Stay updated with current trends and insights."
                },
                {
                    "title": f"{query.title()} Best Practices",
                    "url": f"https://example.com/{query.replace(' ', '-')}-guide",
                    "snippet": f"Learn best practices and strategies for implementing {query} effectively in your organization."
                },
                {
                    "title": f"The Future of {query.title()}",
                    "url": f"https://example.com/future-{query.replace(' ', '-')}",
                    "snippet": f"Discover how {query} is shaping the future of technology and business. Expert analysis and predictions."
                },
                {
                    "title": f"{query.title()} Case Studies",
                    "url": f"https://example.com/{query.replace(' ', '-')}-cases",
                    "snippet": f"Real-world examples and case studies demonstrating successful {query} implementations across industries."
                }
            ]
        }
        
        # Select appropriate mock data
        query_lower = query.lower()
        if "ai" in query_lower or "artificial intelligence" in query_lower:
            base_results = mock_data["ai"]
        elif "automation" in query_lower or "automate" in query_lower:
            base_results = mock_data["automation"]
        else:
            base_results = mock_data["default"]
        
        # Format results
        formatted_results = []
        for i, result in enumerate(base_results[:max_results], 1):
            formatted_result = {
                "position": i,
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"]
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def search_news(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for news articles
        
        Args:
            query: Search query
            max_results: Maximum number of results
        
        Returns:
            List of news articles
        """
        try:
            print(f"📰 Searching news for: '{query}'")
            
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_search_time
            if time_since_last < self.min_search_delay:
                time.sleep(self.min_search_delay - time_since_last)
            
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.news(
                    query,
                    max_results=max_results
                ))
            
            self.last_search_time = time.time()
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_result = {
                    "position": i,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "source": result.get("source", ""),
                    "published": result.get("date", ""),
                    "snippet": result.get("body", ""),
                }
                formatted_results.append(formatted_result)
            
            print(f"✅ Found {len(formatted_results)} news articles")
            return formatted_results
            
        except Exception as e:
            print(f"⚠️ News search error: {e}, using mock news data")
            return self._generate_mock_news(query, max_results)
    
    def _generate_mock_news(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Generate mock news results"""
        formatted_results = []
        for i in range(min(max_results, 3)):
            formatted_result = {
                "position": i + 1,
                "title": f"Latest {query.title()} Developments: Report #{i+1}",
                "url": f"https://example.com/news/{query.replace(' ', '-')}-{i+1}",
                "source": f"Tech News {i+1}",
                "published": datetime.now().strftime("%Y-%m-%d"),
                "snippet": f"Recent developments in {query} show promising results. Industry experts weigh in on the latest trends and innovations."
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def get_search_summary(self) -> str:
        """Get summary of search history"""
        if not self.search_history:
            return "No searches performed yet"
        
        total_searches = len(self.search_history)
        total_results = sum(s["results_count"] for s in self.search_history)
        
        return f"Performed {total_searches} searches, found {total_results} total results"


class WebScraperTool:
    """
    Extract content from web pages
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.scraped_urls: List[str] = []
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a URL
        
        Args:
            url: URL to scrape
        
        Returns:
            Dictionary with title, text, links, metadata
        """
        try:
            print(f"🌐 Scraping: {url[:50]}...")
            
            # Set headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title = title.get_text().strip() if title else "No title"
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                links.append({
                    "text": link.get_text().strip(),
                    "url": link['href']
                })
            
            # Extract metadata
            metadata = {}
            for meta in soup.find_all('meta'):
                name = meta.get('name', meta.get('property', ''))
                content = meta.get('content', '')
                if name and content:
                    metadata[name] = content
            
            self.scraped_urls.append(url)
            
            result = {
                "url": url,
                "title": title,
                "text": text[:5000],  # Limit to first 5000 chars
                "text_length": len(text),
                "links_count": len(links),
                "links": links[:10],  # First 10 links
                "metadata": metadata,
                "scraped_at": datetime.now().isoformat()
            }
            
            print(f"✅ Scraped {len(text)} characters")
            return result
            
        except Exception as e:
            print(f"❌ Scraping error: {e}")
            return {
                "url": url,
                "error": str(e),
                "scraped_at": datetime.now().isoformat()
            }
    
    def scrape_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs
        
        Args:
            urls: List of URLs to scrape
        
        Returns:
            List of scraping results
        """
        results = []
        for url in urls:
            result = self.scrape_url(url)
            results.append(result)
        
        return results


class ContentAnalyzerTool:
    """
    Analyze and extract insights from content
    """
    
    @staticmethod
    def extract_facts(text: str) -> List[str]:
        """
        Extract factual statements from text
        
        Args:
            text: Text to analyze
        
        Returns:
            List of facts
        """
        # Simple fact extraction based on sentences with numbers or specific patterns
        sentences = text.split('.')
        facts = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for sentences with numbers or percentages
            if re.search(r'\d+', sentence) or '%' in sentence:
                if len(sentence) > 20:  # Meaningful length
                    facts.append(sentence)
        
        return facts[:10]  # Return top 10 facts
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Text to analyze
            top_n: Number of keywords to return
        
        Returns:
            List of keywords
        """
        # Simple keyword extraction
        # Remove common words
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Tokenize and count
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:top_n]]
    
    @staticmethod
    def summarize_sources(search_results: List[Dict]) -> str:
        """
        Create a summary of search result sources
        
        Args:
            search_results: List of search results
        
        Returns:
            Summary string
        """
        if not search_results:
            return "No sources found"
        
        summary_parts = []
        summary_parts.append(f"Found {len(search_results)} sources:\n")
        
        for i, result in enumerate(search_results[:5], 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            snippet = result.get('snippet', '')[:100]
            
            summary_parts.append(f"{i}. {title}")
            summary_parts.append(f"   URL: {url}")
            summary_parts.append(f"   Preview: {snippet}...\n")
        
        return '\n'.join(summary_parts)


class CitationFormatter:
    """
    Format citations in various styles
    """
    
    @staticmethod
    def format_apa(source: Dict[str, str]) -> str:
        """
        Format citation in APA style
        
        Args:
            source: Dictionary with title, url, date
        
        Returns:
            APA formatted citation
        """
        title = source.get('title', 'Untitled')
        url = source.get('url', '')
        date = source.get('date', datetime.now().strftime('%Y, %B %d'))
        
        return f"{title}. ({date}). Retrieved from {url}"
    
    @staticmethod
    def format_mla(source: Dict[str, str]) -> str:
        """
        Format citation in MLA style
        
        Args:
            source: Dictionary with title, url, date
        
        Returns:
            MLA formatted citation
        """
        title = source.get('title', 'Untitled')
        url = source.get('url', '')
        date = source.get('date', datetime.now().strftime('%d %b. %Y'))
        
        return f'"{title}." Web. {date}. <{url}>.'
    
    @staticmethod
    def create_bibliography(sources: List[Dict[str, str]], style: str = 'apa') -> str:
        """
        Create a bibliography from sources
        
        Args:
            sources: List of source dictionaries
            style: Citation style ('apa' or 'mla')
        
        Returns:
            Formatted bibliography
        """
        formatter = CitationFormatter.format_apa if style == 'apa' else CitationFormatter.format_mla
        
        bibliography = ["## References\n"]
        for i, source in enumerate(sources, 1):
            citation = formatter(source)
            bibliography.append(f"{i}. {citation}")
        
        return '\n'.join(bibliography)


# =============================================================================
# TOOL WRAPPER FUNCTIONS (for agent use)
# =============================================================================

# Global tool instances
_search_tool = GoogleSearchTool()
_scraper_tool = WebScraperTool()
_analyzer_tool = ContentAnalyzerTool()


def google_search(query: str) -> str:
    """
    Search Google and return formatted results
    
    Args:
        query: Search query
    
    Returns:
        Formatted search results
    """
    results = _search_tool.search(query)
    return _analyzer_tool.summarize_sources(results)


def search_news(query: str) -> str:
    """
    Search for news articles
    
    Args:
        query: Search query
    
    Returns:
        Formatted news results
    """
    results = _search_tool.search_news(query)
    return _analyzer_tool.summarize_sources(results)


def scrape_website(url: str) -> str:
    """
    Scrape content from a website
    
    Args:
        url: Website URL
    
    Returns:
        Scraped content
    """
    result = _scraper_tool.scrape_url(url)
    if 'error' in result:
        return f"Error scraping {url}: {result['error']}"
    
    return f"Title: {result['title']}\n\nContent Preview:\n{result['text'][:1000]}..."


def extract_facts(text: str) -> str:
    """
    Extract facts from text
    
    Args:
        text: Text to analyze
    
    Returns:
        Extracted facts
    """
    facts = _analyzer_tool.extract_facts(text)
    if not facts:
        return "No facts extracted"
    
    return "Extracted Facts:\n" + "\n".join(f"- {fact}" for fact in facts)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n🔧 CUSTOM TOOLS - Testing Search with Fallbacks\n")
    print("="*60)
    
    # Test 1: Search with automatic fallback
    print("\n1️⃣ Testing Search (with automatic fallback)...\n")
    search_tool = GoogleSearchTool(max_results=5)
    results = search_tool.search("artificial intelligence trends 2025")
    
    print(f"\nSearch Results:")
    for result in results[:3]:
        print(f"\n{result['position']}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
    
    # Test 2: Wikipedia fallback
    print("\n\n2️⃣ Testing Wikipedia Search...\n")
    wiki_results = search_tool._search_wikipedia("machine learning", 3)
    
    for result in wiki_results:
        print(f"\n{result['position']}. {result['title']}")
        print(f"   URL: {result['url']}")
    
    # Test 3: Mock results
    print("\n\n3️⃣ Testing Mock Results...\n")
    mock_results = search_tool._generate_mock_results("AI automation", 3)
    
    for result in mock_results:
        print(f"\n{result['position']}. {result['title']}")
        print(f"   URL: {result['url']}")
    
    print("\n" + "="*60)
    print("✅ Custom Tools Tests Complete!")
    print("="*60 + "\n")