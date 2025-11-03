import requests


def extract_contents(url:str):
    """
    Extract text from the given url
    """
    try:
        print("[INFO] Extracting texts from url....")
        texts = requests.get(url).text
        return texts
    
    except Exception as e:
        print(f"[ERROR] Error request failed: {e}")
        return ""