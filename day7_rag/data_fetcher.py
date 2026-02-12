"""
WHAT: Fetches real-time stock data and downloads SEC filings
WHY: RAG needs up-to-date information about companies
HOW: Uses yfinance for stock data, sec-edgar-downloader for filings
"""

import yfinance as yf
import os
import requests
from datetime import datetime, timedelta
from sec_edgar_downloader import Downloader
from dotenv import load_dotenv

load_dotenv()

class StockDataFetcher:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.sec_dir = os.path.join(data_dir, "sec_filings")
        self.news_dir = os.path.join(data_dir, "news_articles")
        
        # Create directories
        os.makedirs(self.sec_dir, exist_ok=True)
        os.makedirs(self.news_dir, exist_ok=True)
    
    def fetch_realtime_stock_data(self, ticker):
        """
        WHAT: Get current stock price and key metrics
        WHY: Provides real-time context for user queries
        RETURNS: Dict with current price, volume, market cap, etc.
        """
        print(f"📊 Fetching real-time data for {ticker}...")
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current data
        current_data = {
            'ticker': ticker,
            'current_price': info.get('currentPrice', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'volume': info.get('volume', 'N/A'),
            'company_name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"✅ Current price: ${current_data['current_price']}")
        return current_data
    
    def download_sec_filings(self, ticker, filing_type="10-K", num_filings=3):
        """
        WHAT: Downloads SEC filings (10-K annual reports, 10-Q quarterly)
        WHY: These contain detailed financial information
        HOW: Uses sec-edgar-downloader library
        """
        print(f"📥 Downloading {num_filings} {filing_type} filings for {ticker}...")
        
        dl = Downloader("MyCompany", "my_email@example.com", self.sec_dir)
        
        try:
            # Download filings
            dl.get(filing_type, ticker, limit=num_filings)
            print(f"✅ Downloaded {filing_type} filings to {self.sec_dir}")
            
            # Return paths to downloaded files
            company_dir = os.path.join(self.sec_dir, "sec-edgar-filings", ticker, filing_type)
            filing_paths = []
            
            if os.path.exists(company_dir):
                for root, dirs, files in os.walk(company_dir):
                    for file in files:
                        if file.endswith('.txt'):
                            filing_paths.append(os.path.join(root, file))
            
            return filing_paths
            
        except Exception as e:
            print(f"❌ Error downloading filings: {e}")
            return []
    
    def fetch_financial_news(self, ticker, days=7):
        """
        WHAT: Fetch recent news articles about the company
        WHY: Provides current sentiment and events
        HOW: Uses News API (free tier: 100 requests/day)
        """
        print(f"📰 Fetching news for {ticker} from last {days} days...")
        
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            print("⚠️ No NEWS_API_KEY found. Skipping news fetch.")
            return []
        
        # Get company name from ticker
        stock = yf.Ticker(ticker)
        company_name = stock.info.get('longName', ticker)
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # News API endpoint
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': company_name,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': api_key,
            'pageSize': 10
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            
            print(f"✅ Found {len(articles)} news articles")
            
            # Save articles as text files
            news_files = []
            for i, article in enumerate(articles):
                filename = os.path.join(
                    self.news_dir, 
                    f"{ticker}_news_{i}_{datetime.now().strftime('%Y%m%d')}.txt"
                )
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {article['title']}\n")
                    f.write(f"Source: {article['source']['name']}\n")
                    f.write(f"Published: {article['publishedAt']}\n")
                    f.write(f"URL: {article['url']}\n\n")
                    f.write(f"Content: {article.get('content', article.get('description', ''))}\n")
                
                news_files.append(filename)
            
            return news_files
            
        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            return []


# USAGE EXAMPLE
if __name__ == "__main__":
    fetcher = StockDataFetcher()
    
    # Example: Apple Inc.
    ticker = "AAPL"
    
    # 1. Get real-time data
    realtime_data = fetcher.fetch_realtime_stock_data(ticker)
    print("\n" + "="*50)
    print("REAL-TIME DATA:")
    for key, value in realtime_data.items():
        print(f"{key}: {value}")
    
    # 2. Download SEC filings (10-K and 10-Q)
    print("\n" + "="*50)
    filing_paths = fetcher.download_sec_filings(ticker, "10-K", num_filings=2)
    print(f"Downloaded {len(filing_paths)} SEC filings")
    
    # 3. Fetch recent news
    print("\n" + "="*50)
    news_paths = fetcher.fetch_financial_news(ticker, days=7)
    print(f"Saved {len(news_paths)} news articles")