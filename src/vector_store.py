import chromadb
import os
import numpy as np
from typing import Any, List
from src.embedding import EmbeddingPipeline
from sentence_transformers import SentenceTransformer


class ChromaVectorStore:
    def __init__(
        self,
        topic_collection_name="langchain_topics",
        content_collection_name="langchain_content",
        persist_directory="db/vectorstore",
    ):
        self.topic_collection_name = topic_collection_name
        self.content_collection_name = content_collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.topic_collection = self.client.get_or_create_collection(
                name=self.topic_collection_name,
                metadata={"description": "Langchain topic embeddings for RAG"},
            )

            self.content_collection = self.client.get_or_create_collection(
                name=self.content_collection_name,
                metadata={"description": "Langchain content embeddings for RAG"},
            )
            print(
                f"[INFO] Vector store initialised. Topic Collection:{self.topic_collection_name}, Content Collection:{self.content_collection_name}"
            )
            print(
                f"[INFO] Existing topics in collection:{self.topic_collection.count()}"
            )
            print(
                f"[INFO] Existing content in collection:{self.content_collection.count()}"
            )

        except Exception as e:
            print(f"[ERROR]Error initializing vector store: {e}")
