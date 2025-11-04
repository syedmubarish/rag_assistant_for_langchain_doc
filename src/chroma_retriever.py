import chromadb
from src.embedding import EmbeddingPipeline
from src.vector_store import ChromaVectorStore
from typing import Any, List, Dict


class ChromaRetriever:
    def __init__(
        self, vector_store: ChromaVectorStore, embedding_pipeline: EmbeddingPipeline
    ):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline

    def topic_retrieve(
        self, query: str, top_k: int = 5, score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        print(f"[INFO]Retrieving topics for query: '{query}'")
        print(f"[INFO]Top K: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_pipeline.embed_query(query)

        try:
            results = self.vector_store.topic_collection.query(
                query_embeddings=query_embedding.tolist(), n_results=top_k
            )

            retrieved_docs = []
            if results["documents"]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "doc_id": doc_id,
                                "topic": document,
                                "url": metadata.get("url"),
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )
                print(
                    f"[INFO]Retrieved {len(retrieved_docs)} documents after filtering"
                )
            else:
                print(f"[INFO] No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"[ERROR]Error during retrievel: {e}")
            return []
    

    def context_retrieve(
        self, query: str, top_k: int = 10, score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        print(f"[INFO]Retrieving context for query: '{query}'")
        print(f"[INFO]Top K: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_pipeline.embed_query(query)

        try:
            results = self.vector_store.content_collection.query(
                query_embeddings=query_embedding.tolist(), n_results=top_k
            )

            retrieved_docs = []
            if results["documents"]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "doc_id": doc_id,
                                "topic": document,
                                "url": metadata.get("source_url"),
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )
                print(
                    f"[INFO]Retrieved {len(retrieved_docs)} documents after filtering"
                )
            else:
                print(f"[INFO] No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"[ERROR]Error during retrievel: {e}")
            return []
