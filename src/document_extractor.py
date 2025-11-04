import requests
from typing import List, Any
from src.embedding import EmbeddingPipeline
from src.vector_store import ChromaVectorStore


def extract_contents(retrieved_docs: List[Any]):
    """
    Extract text from the given retrieved docs urls
    """

    content_list = []

    try:
        for doc in retrieved_docs:
            print(f"[INFO] Extracting texts from {doc['topic']}")
            texts = requests.get(doc["url"]).text
            content_list.append(texts)

        return content_list

    except Exception as e:
        print(f"[ERROR] Error request failed: {e}")
        return ""


def chunking(
    content_list, embedding_pipeline: EmbeddingPipeline, vector_store: ChromaVectorStore
):
    """
    Convert the text from list into chunks and embedding it after that storing it into vector db
    """
    chunked_content_list=[]
    try:
        print(f"[INFO]Start chunking...chunk..chunk")
        for content in content_list:
            chunked_content = embedding_pipeline.chunk_text(content)
            chunked_content_list.append(chunked_content)
        return chunked_content_list
    except Exception as e:
        print(f"[ERROR] Error occured while chunking: {e}")

def flatten_chunks_with_metadata(chunks_list, docs_metadata):
    """
    Flatten a list of chunk lists into a list of dictionaries with text, source_url, and chunk_index.
    
    Args:
        chunks_list (list of lists): Nested list of chunks per source.
        docs_metadata (list of dict): List of document metadata dicts, each containing 'url'.
        
    Returns:
        list of dict: Flattened list with text, source_url, and chunk_index.
    """
    flattened = []
    
    for doc, chunks in zip(docs_metadata, chunks_list):
        url = doc['url']
        for idx, chunk in enumerate(chunks, start=1):
            flattened.append({
                "text": chunk,
                "source_url": url,
                "chunk_index": idx
            })
    
    return flattened