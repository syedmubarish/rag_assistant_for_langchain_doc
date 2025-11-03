from src import data_extractor
from src import document_structure
from src import embedding
from src import vector_store
from src import chroma_retriever
from src import document_extractor

if __name__ == "__main__":
    url = "https://docs.langchain.com/llms.txt"
    topics, urls = data_extractor.extract_topics_urls(url)

    document_structures = document_structure.create_document_structures(topics, urls)

    emb_pipe = embedding.EmbeddingPipeline()

    topic_embeddings = emb_pipe.embed_topic(topics)
    

    chroma_store = vector_store.ChromaVectorStore()

    # chroma_store.add_topics(topics,topic_embeddings,document_structures) --> Uncomment this if topics are not added to collection

    chroma_retriever = chroma_retriever.ChromaRetriever(chroma_store,emb_pipe)

    retrived_docs = chroma_retriever.topic_retrieve("App development in LangSmith Deployment")

    doc_content = document_extractor.extract_contents('https://docs.langchain.com/langsmith/collector-proxy.md')
    print(doc_content)