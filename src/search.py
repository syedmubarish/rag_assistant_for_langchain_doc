from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os


llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key= os.getenv('GOOGLE_API_KEY')
)


def do_rag(query,retriever,llm,top_k=25,min_score=0.0):
    
    results = retriever.context_retrieve(query,top_k=top_k,score_threshold=min_score)
    
    if not results:
        return {'answer':'No relevant context found','sources':[],'confidence':0.0}
    

    context = "\n\n".join([doc['content'] for doc in results])

    

    sources = [{
        'source': doc.get('url','unknown')
    } for doc in results]

    

    prompt = f""" 
        Use the following context to answer the question


        You are a helpful assistant that answers questions based only on the provided LangChain documentation.

        If you are not sure or the answer is not in the context, say "I don't know."

        Context:{context}


        Question:{query}
        
        Sources:{sources}

        Now answer the question in detail, including relevant URLs in your answer.
        Refer the url if needed
    """

    response = llm.invoke([prompt])
    
    output = {
        'answer' : response.content,
        # 'sources' : sources
    }

    return output