from src import data_extractor
from src import document_structure
from src import embedding
from src import vector_store

if __name__ == "__main__":
    url = "https://docs.langchain.com/llms.txt"
    topics, urls = data_extractor.extract_topics_urls(url)

    document_structures = document_structure.create_document_structures(topics, urls)

    emb_pipe = embedding.EmbeddingPipeline()

    topic_embeddings = emb_pipe.embed_topic(topics)
    

    chroma_store = vector_store.ChromaVectorStore()