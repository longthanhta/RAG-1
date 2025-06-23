import requests
from dotenv import load_dotenv
from pathlib import Path
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
    "topic_token": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB", # This is likely "Politics"
    "api_key": api_key
}

# Make the request
response = requests.get(url, params=params)

output_file = Path(__file__).parent /'serpapi_news_results.txt' # Changed to .txt for clear text output

# Handle the response
with open(output_file, 'w', encoding='utf-8') as f: # Added encoding for broader character support
    if response.status_code == 200:
        data = response.json()

        # Add a header to the file
        f.write(f"News Results for Topic: Politics ({params['topic_token']})\n\n")

        news_results = data.get("news_results", [])

        if not news_results:
            f.write("No news results found for this topic.\n")
            print("No news results found for this topic.")
        else:
            article_count = 0
            for article in news_results:
                title = article.get('title')
                source_info = article.get('source')
                link = article.get('link')

                # --- FIX: Check if title and link exist before processing ---
                if title and link: # Only process if both title and link are present
                    article_count += 1
                    source_name = source_info.get('name') if isinstance(source_info, dict) else source_info

                    # Write to file
                    f.write(f"--- Article {article_count} ---\n")
                    f.write(f"Title: {title}\n")
                    f.write(f"Source: {source_name if source_name else 'N/A'}\n") # Handle source being None if not found
                    f.write(f"Link: {link}\n")
                    f.write("-" * 50 + "\n\n") # Separator for readability
                else:
                    # Optional: Print a message to console if an entry is skipped
                    # print(f"Skipping entry due to missing title or link: {article}")
                    pass # Or handle it differently if you want to log these skipped entries

        print(f"News successfully stored in '{output_file}'. Processed {article_count} valid articles.")
    else:
        print("Failed to fetch news:", response.status_code)
        print(response.text)
