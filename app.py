from src import data_extractor

if __name__ == "__main__":
    url = "https://docs.langchain.com/llms.txt"
    topics,urls = data_extractor.extract_topics_urls(url)
    print(f"TOPICS: {topics}")
    print(f"URLS: {urls}")