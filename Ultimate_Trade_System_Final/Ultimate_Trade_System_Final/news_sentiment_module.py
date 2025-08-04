
import requests
import datetime
import pandas as pd
from textblob import TextBlob

# CONFIG
NEWS_API_KEY = "your_newsapi_key"  # or use scraping fallback if needed

def fetch_nifty_news():
    url = f"https://newsapi.org/v2/everything?q=Nifty OR Sensex OR Indian Stock Market&language=en&sortBy=publishedAt&pageSize=20&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [(art["title"], art["description"]) for art in articles]

def analyze_sentiment(news_items):
    sentiment_scores = []
    for title, desc in news_items:
        text = title + " " + (desc or "")
        blob = TextBlob(text)
        sentiment_scores.append(blob.sentiment.polarity)  # range -1 to 1

    if not sentiment_scores:
        return 0  # Neutral if no news

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    return avg_sentiment

def get_daily_sentiment_score():
    try:
        news_items = fetch_nifty_news()
        score = analyze_sentiment(news_items)
        return round(score, 3)
    except Exception as e:
        print("Sentiment fetch error:", e)
        return 0

# Example
if __name__ == "__main__":
    score = get_daily_sentiment_score()
    print("📊 Market Sentiment Score:", score)
