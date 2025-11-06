import requests
import re


def extract_topics_urls(url: str):
    """
    Extract topics and urls from the given url
    """
    try:
        response = requests.get(url)
        text = response.text

        

        topics = re.findall(r"\[([^\[\]]+)\]", text)
        urls = re.findall(r"https?://[^\s\]]+md", text)

        return topics, urls

    except Exception as e:
        print(f"[ERROR] Error request failed: {e}")
        return [], []
