from typing import Any, List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)

        

    def embed_topic(self, topics: List[Any]) -> np.ndarray:
        
        topic_embeddings = self.model.encode(topics, show_progress_bar=True)
        return topic_embeddings

    def chunk_text(self, text: str) -> List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_text(text)
        
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:

        texts = [chunk["text"] for chunk in chunks]
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        
        query_embedding = self.model.encode(query)
        
        return query_embedding
