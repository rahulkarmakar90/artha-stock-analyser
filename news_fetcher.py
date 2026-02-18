import feedparser
import requests
from datetime import datetime
from textblob import TextBlob  # pip install textblob

class NewsAnalyzer:
    def __init__(self, newsapi_key=None):
        self.newsapi_key = newsapi_key
    
    def fetch_google_news(self, stock_name: str, limit: int = 10):
        """Fetch news from Google News RSS"""
        query = f"{stock_name} stock India"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN"
        
        feed = feedparser.parse(url)
        articles = []
        
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", ""),
                "source": entry.get("source", {}).get("title", "Unknown")
            })
        
        return articles
    
    def analyze_sentiment(self, text: str):
        """Basic sentiment analysis using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "polarity": polarity,
            "sentiment": sentiment
        }
    
    def get_news_with_sentiment(self, stock_name: str):
        """Fetch news and analyze sentiment"""
        articles = self.fetch_google_news(stock_name)
        
        total_polarity = 0
        analyzed_articles = []
        
        for article in articles:
            sentiment = self.analyze_sentiment(article["title"])
            article["sentiment"] = sentiment
            analyzed_articles.append(article)
            total_polarity += sentiment["polarity"]
        
        avg_sentiment = total_polarity / len(articles) if articles else 0
        
        return {
            "articles": analyzed_articles,
            "overall_sentiment": avg_sentiment,
            "sentiment_label": "bullish" if avg_sentiment > 0.1 else "bearish" if avg_sentiment < -0.1 else "neutral"
        }