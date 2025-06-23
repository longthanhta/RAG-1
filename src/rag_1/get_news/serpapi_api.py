import requests
from dotenv import load_dotenv
import os

load_dotenv()
# Replace with your actual SerpAPI API key
api_key = os.getenv("GOOGLE_NEWS_API")
print('api_key',api_key)
# Full URL with base path
url = "https://serpapi.com/search.json"

# Query parameters directly from the URL you provided
params = {
    "engine": "google_news",
    "gl": "us",
    "hl": "en",
    "topic_token": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB",
    "api_key": api_key
}

# Make the request
response = requests.get(url, params=params)

# Handle the response
if response.status_code == 200:
    data = response.json()
    print("Top News Results:")
    for article in data.get("news_results", []):
        print(f"- {article.get('title')}")
        print(f"  Source: {article.get('source')}")
        print(f"  Link: {article.get('link')}\n")
else:
    print("Failed to fetch news:", response.status_code)
    print(response.text)
