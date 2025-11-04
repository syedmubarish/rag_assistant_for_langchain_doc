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
    

    def add_documents(self,documents:List[Any],document_embeddings:np.ndarray):
        if len(documents) != len(document_embeddings):
            raise ValueError("Number of topics and embeddings must match")
        
        print(f"[INFO] Adding {len(document_embeddings)} to vector store....")
        ids = []
        metadatas = []
        documents_list = []
        embeddings_list = []

        for i, (doc,embedding) in enumerate(zip(documents, document_embeddings)):
            doc_id = f"{uuid.uuid4().hex[:7]}_{i}"
            ids.append(doc_id)

            metadata = {}
            metadata['source_url']=doc['source_url']
            metadata['doc_index']=doc['chunk_index']
            metadata['content_length']=len(doc['text'])
            metadatas.append(metadata)

            documents_list.append(doc['text'])

            embeddings_list.append(embedding.tolist())

        try:
            self.content_collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_list,
                embeddings=embeddings_list
            )

            print(f"[INFO]Successfully added {len(documents_list)} documents to vector store")
            print(f"[INFO]Total documents in collection: {self.content_collection.count()}")
        except Exception as e:
            print(f"[ERROR] Adding documents to chromadb : {e}")
            raise

    def peeking(self):
        print(self.content_collection.peek())