from src import data_extractor
from src import document_structure

if __name__ == "__main__":
    url = "https://docs.langchain.com/llms.txt"
    topics,urls = data_extractor.extract_topics_urls(url)
   
    
    document_structures = document_structure.create_document_structures(topics,urls)
    
    print(document_structures[1].__dict__)