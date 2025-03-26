#!/usr/bin/env python3
"""
CryptoTrendAgent - A lightweight AI research assistant for crypto traders.
"""

import os
import sys
import sqlite3
import json
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import click
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Database initialization
def init_db():
    """Initialize the SQLite database with required tables."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'crypto_trends.db')
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create articles table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        coin TEXT NOT NULL,
        url TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        publication_date TEXT,
        retrieval_date TEXT NOT NULL,
        content TEXT,
        summary TEXT,
        sentiment TEXT,
        score REAL
    )
    ''')
    
    conn.commit()
    conn.close()
    
    return db_path

# API Clients
class BraveSearchClient:
    """Client for Brave Search API."""
    
    def __init__(self):
        self.api_key = os.getenv('BRAVE_API_KEY')
        if not self.api_key:
            raise ValueError("Brave API key not found in environment variables")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
    
    def search_crypto_news(self, coin, count=5):
        """Search for latest news about a cryptocurrency."""
        params = {
            "q": f"{coin} cryptocurrency news",
            "count": count,
            "search_lang": "en",
            "freshness": "pd"  # Past day
        }
        
        response = requests.get(self.base_url, headers=self.headers, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Brave Search API error: {response.status_code} - {response.text}")
        
        results = response.json().get('web', {}).get('results', [])
        return results

class ArticleFetcher:
    """Handles fetching and extracting content from articles."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def fetch_article(self, url):
        """Fetch and extract content from an article URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.text.strip() if soup.title else "No title found"
            
            # Extract article content (this is a simple heuristic and may need refinement)
            article_content = ""
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                if len(p.text.strip()) > 50:  # Skip short paragraphs
                    article_content += p.text.strip() + "\n\n"
            
            if not article_content:
                return None, "Failed to extract content"
            
            return {
                "title": title,
                "content": article_content[:10000]  # Limit content size
            }, None
            
        except Exception as e:
            return None, f"Error fetching article: {str(e)}"

class ClaudeClient:
    """Client for Anthropic's Claude API."""
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not found in environment variables")
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def summarize_article(self, article_content, coin):
        """Summarize an article using Claude."""
        prompt = f"""
        The following is an article about {coin} cryptocurrency. Please summarize the key points in 3-4 sentences, 
        focusing on market sentiment, price predictions, and notable events. Then, classify the overall sentiment 
        as 'bullish', 'bearish', or 'neutral', and provide a confidence score from 0.0 to 1.0.
        
        Format your response as JSON:
        {{
          "summary": "your summary here",
          "sentiment": "bullish/bearish/neutral",
          "score": 0.75
        }}
        
        Here's the article:
        
        {article_content}
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                system="You are a cryptocurrency analyst assistant that provides concise, factual summaries and sentiment analysis.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract JSON from response
            result_text = response.content[0].text
            
            # Find JSON in the response
            import re
            json_match = re.search(r'({.*})', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group(1))
                return result_json, None
            else:
                # Fallback: try to parse the entire response as JSON
                try:
                    result_json = json.loads(result_text)
                    return result_json, None
                except json.JSONDecodeError:
                    return None, "Failed to parse Claude's response as JSON"
            
        except Exception as e:
            return None, f"Error calling Claude API: {str(e)}"

class CryptoTrendAgent:
    """Main application class that orchestrates the crypto trend analysis."""
    
    def __init__(self):
        self.db_path = init_db()
        self.brave_client = BraveSearchClient()
        self.article_fetcher = ArticleFetcher()
        self.claude_client = ClaudeClient()
    
    def run(self, coin, count=3):
        """Run a complete analysis cycle for a cryptocurrency."""
        click.echo(f"🔍 Searching for latest {coin} news...")
        
        # Search for articles
        search_results = self.brave_client.search_crypto_news(coin, count)
        
        if not search_results:
            click.echo("No results found.")
            return
        
        click.echo(f"Found {len(search_results)} articles.")
        
        # Process each article
        for result in search_results:
            url = result.get('url')
            title = result.get('title')
            
            # Check if article already exists in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM articles WHERE url = ?", (url,))
            existing = cursor.fetchone()
            conn.close()
            
            if existing:
                click.echo(f"⏩ Skipping already processed article: {title}")
                continue
            
            click.echo(f"📄 Processing: {title}")
            
            # Fetch and extract article content
            article_data, error = self.article_fetcher.fetch_article(url)
            if error:
                click.echo(f"❌ {error}")
                continue
            
            # Summarize and analyze sentiment
            click.echo("🧠 Analyzing with Claude...")
            analysis, error = self.claude_client.summarize_article(article_data['content'], coin)
            if error:
                click.echo(f"❌ {error}")
                continue
            
            # Store results in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            try:
                cursor.execute("""
                INSERT INTO articles (coin, url, title, retrieval_date, content, summary, sentiment, score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    coin.lower(),
                    url,
                    article_data['title'],
                    datetime.now().isoformat(),
                    article_data['content'],
                    analysis['summary'],
                    analysis['sentiment'],
                    analysis['score']
                ))
                conn.commit()
                click.echo(f"✅ Stored analysis: {analysis['sentiment']} ({analysis['score']:.2f})")
            except sqlite3.IntegrityError:
                click.echo("⚠️ Article already exists in database")
            finally:
                conn.close()
            
            # Slight delay to avoid rate limiting
            time.sleep(1)
    
    def show(self, coin, limit=5):
        """Show the latest sentiment analysis for a cryptocurrency."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT title, url, summary, sentiment, score, retrieval_date
        FROM articles
        WHERE coin = ?
        ORDER BY retrieval_date DESC
        LIMIT ?
        """, (coin.lower(), limit))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            click.echo(f"No data found for {coin}.")
            return
        
        click.echo(f"\n📊 Latest {coin} Sentiment Analysis:\n")
        
        for title, url, summary, sentiment, score, date in results:
            # Get sentiment emoji
            sentiment_emoji = "🟢" if sentiment == "bullish" else "🔴" if sentiment == "bearish" else "⚪"
            
            click.echo(f"{sentiment_emoji} {title}")
            click.echo(f"   URL: {url}")
            click.echo(f"   Date: {date[:10]}")
            click.echo(f"   Sentiment: {sentiment.capitalize()} (confidence: {score:.2f})")
            click.echo(f"   Summary: {summary}")
            click.echo("\n" + "-" * 80 + "\n")
        
        # Calculate overall sentiment
        cursor = conn.cursor()
        cursor.execute("""
        SELECT sentiment, COUNT(*) as count
        FROM articles
        WHERE coin = ?
        GROUP BY sentiment
        ORDER BY count DESC
        """, (coin.lower(),))
        
        sentiment_counts = cursor.fetchall()
        
        if sentiment_counts:
            click.echo(f"Overall sentiment distribution for {coin}:")
            for sentiment, count in sentiment_counts:
                sentiment_emoji = "🟢" if sentiment == "bullish" else "🔴" if sentiment == "bearish" else "⚪"
                click.echo(f"{sentiment_emoji} {sentiment.capitalize()}: {count} articles")

# CLI Interface
@click.group()
def cli():
    """CryptoTrendAgent - AI research assistant for crypto traders."""
    pass

@cli.command()
@click.argument('coin')
@click.option('--count', '-c', default=3, help='Number of articles to process (default: 3)')
def run(coin, count):
    """Search and analyze latest news for a cryptocurrency."""
    agent = CryptoTrendAgent()
    agent.run(coin, count)

@cli.command()
@click.argument('coin')
@click.option('--limit', '-l', default=5, help='Number of results to show (default: 5)')
def show(coin, limit):
    """Show stored sentiment analysis for a cryptocurrency."""
    agent = CryptoTrendAgent()
    agent.show(coin, limit)

if __name__ == '__main__':
    cli()