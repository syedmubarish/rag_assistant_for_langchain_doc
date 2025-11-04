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

    doc_content = document_extractor.extract_contents(retrived_docs)

    chunked_content_list = document_extractor.chunking(doc_content,emb_pipe,chroma_store)
    
    # for chunk in chunked_content_list:
    #     print(chunk)
    #     print(len(chunk))
    #     print("==========================SYED MUBARISH THANGAL=========================")

    flattened_chunks = document_extractor.flatten_chunks_with_metadata(chunked_content_list,retrived_docs)

    chunks_embedded = emb_pipe.embed_chunks(flattened_chunks)

    
    
    
    