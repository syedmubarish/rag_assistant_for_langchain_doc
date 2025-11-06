import uuid
from langchain_core.documents import Document
from typing import List, Any


def create_document_structures(topics: List[Any], urls: List[Any]) -> List[Any]:

    document_structures = []

    for i, (topic, url) in enumerate(zip(topics, urls)):
        doc = Document(
            id=f"doc_{uuid.uuid4().hex[:8]}_{i}",
            page_content=topic,
            metadata={"url": url},
        )
        document_structures.append(doc)

    
    return document_structures
