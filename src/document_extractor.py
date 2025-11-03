import requests
from typing import List,Any


def extract_contents(retrieved_docs:List[Any]):
    """
    Extract text from the given retrieved docs urls
    """

    content_list = []

    try:
        for doc in retrieved_docs:
            print(f"[INFO] Extracting texts from {doc['topic']}")
            texts = requests.get(doc['url']).text
            content_list.append(texts)
        
        return content_list
    
    except Exception as e:
        print(f"[ERROR] Error request failed: {e}")
        return ""