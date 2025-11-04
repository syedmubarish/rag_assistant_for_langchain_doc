import chromadb
import os
import numpy as np
from typing import Any, List
from src.embedding import EmbeddingPipeline
from sentence_transformers import SentenceTransformer
import uuid


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


    def add_topics(self,topics: List[Any], topic_embeddings: np.ndarray, document_structures: List[Any]):
        if len(topics) != len(topic_embeddings):
            raise ValueError("Number of topics and embeddings must match")
        
        print(f"[INFO] Adding {len(topic_embeddings)} to vector store....")

        ids = []
        metadatas = []
        topic_list = []
        embeddings_list = []

        for document_structure in document_structures:
            ids.append(document_structure.id)
            metadatas.append(document_structure.metadata)
            topic_list = topics
            embeddings_list = topic_embeddings.tolist()

        try:
            self.topic_collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=topic_list,
                embeddings=embeddings_list
            )

            print(f"[INFO]Successfully added {len(topics)} documents to vector store")
            print(f"[INFO]Total documents in collection: {self.topic_collection.count()}")
        except Exception as e:
            print(f"[ERROR] Adding documents to chromadb : {e}")
            raise
    

    def add_documents(self, documents: List[Any], document_embeddings: np.ndarray):
        if len(documents) != len(document_embeddings):
            raise ValueError("Number of documents and embeddings must match")
    
        print(f"[INFO] Attempting to add {len(document_embeddings)} documents to vector store...")

    
        try:
            existing = self.content_collection.get(include=["metadatas"])
            existing_meta = existing.get("metadatas", [])
        except Exception as e:
            print(f"[WARN] Could not retrieve existing documents from DB: {e}")
            existing_meta = []

    
        existing_pairs = {
            (meta.get("source_url"), meta.get("doc_index"))
            for meta in existing_meta if meta
        }

        print(f"[INFO] Found {len(existing_pairs)} existing entries in DB")

    
        new_docs = []
        new_embeddings = []
        for doc, embedding in zip(documents, document_embeddings):
            pair = (doc.get("source_url"), doc.get("chunk_index"))
            if pair in existing_pairs:
                print(f"[SKIP] Duplicate found for {pair}, skipping...")
                continue
            new_docs.append(doc)
            new_embeddings.append(embedding)

        
        if not new_docs:
            print("[INFO] No new documents to add. All are already in DB.")
            return

    
        ids = [f"{uuid.uuid4().hex[:7]}_{i}" for i in range(len(new_docs))]
        metadatas = [
            {
                "source_url": doc["source_url"],
                "doc_index": doc["chunk_index"],
                "content_length": len(doc["text"]),
            }
            for doc in new_docs
        ]
        documents_list = [doc["text"] for doc in new_docs]
        embeddings_list = [embedding.tolist() for embedding in new_embeddings]

        
        try:
            self.content_collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_list,
                embeddings=embeddings_list
            )
            print(f"[INFO] Successfully added {len(new_docs)} new documents to vector store")
            print(f"[INFO] Total documents in collection: {self.content_collection.count()}")
        except Exception as e:
            print(f"[ERROR] Adding documents to ChromaDB: {e}")
            raise


    def peeking(self):
        print(self.content_collection.peek())

    def empty_collection(self):
    
        all_ids = self.content_collection.get()["ids"]
    
        if all_ids:  
            self.content_collection.delete(ids=all_ids)
