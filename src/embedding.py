from typing import Any,List
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

        print(f"[INFO] Loaded embedding model: {model_name}")

    def embed_topic(self,topics: List[Any]) -> np.ndarray:
        print(f"[INFO] Generate embeddings for topics.. {len(topics)} topics")
        topic_embeddings = self.model.encode(topics,show_progress_bar=True)
        return topic_embeddings

    def chunk_text(self, text: str) -> List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_text(text)
        print(f"[INFO] Split {len(text)} texts into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generate embeddings for {len(chunks)} chunks....")
        embeddings = self.model.encode(texts,show_progress_bar=True)
        print(f"Embeddings shape:{embeddings.shape}")
        return embeddings

    def embed_query(self,query:str) -> np.ndarray:
        print(f"[INFO]Generate embedding for query")
        query_embedding = self.model.encode(query)
        print(f"Embeddings shape:{query_embedding.shape}")
        return query_embedding
