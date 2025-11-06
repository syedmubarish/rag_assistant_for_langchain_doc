import requests
import re


def extract_topics_urls(url: str):
    """
    Extract topics and urls from the given url
    """
    try:
        response = requests.get(url)
        text = response.text

        print(
            f"[INFO] Fetched text from the given url. Number of topics {len(text.splitlines())}"
        )
        print("[INFO] Extracting topics adn urls from text....")

        topics = re.findall(r"\[([^\[\]]+)\]", text)
        urls = re.findall(r"https?://[^\s\]]+md", text)

        return topics, urls

    except Exception as e:
        print(f"[ERROR] Error request failed: {e}")
        return [], []
