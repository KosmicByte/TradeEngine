
import requests
from bs4 import BeautifulSoup

def fetch_market_sentiment():
    url = "https://finviz.com/news.ashx"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = [a.text for a in soup.select("table.fullview-news-outer td > a")]

        positive_keywords = ["beats", "soars", "surges", "gains", "rises", "expands"]
        negative_keywords = ["misses", "falls", "drops", "cuts", "loss", "downgrade"]

        score = 0
        for line in headlines[:20]:
            if any(word in line.lower() for word in positive_keywords):
                score += 1
            elif any(word in line.lower() for word in negative_keywords):
                score -= 1

        if score > 3:
            return "Positive"
        elif score < -3:
            return "Negative"
        else:
            return "Neutral"

    except Exception as e:
        print("Sentiment fetch failed:", e)
        return "Neutral"
