# 🧠 LangChain Docs Assistant-Terminal Based (LangChain + Google Gemini + RAG)
A **terminal-based AI assistant** designed to help you explore and understand LangChain documentation faster.
Built with LangChain, Google Gemini, and Retrieval-Augmented Generation (RAG) — this tool allows you to chat with the LangChain docs directly from your terminal.

## 🚀 Overview
This assistant uses:
Google Gemini as the LLM to generate accurate and conversational answers.
LangChain’s RAG pipeline to retrieve and summarize relevant parts of the official documentation.
Vector search (Chroma) to store and fetch context efficiently.
A clean terminal interface for interactive chatting.
Think of it as your personal LangChain tutor in the terminal.

## 🚀 Features
 
- 🔍 **Context Retrieval:** Uses vector embeddings to find the most relevant sections  
- 🧠 **LLM Integration:** Combines retrieved context with a large language model for generation  
- 💾 **Persistent Vector Store:** Reuse embeddings between runs using Chroma  
- 🧩 **LangChain Framework:** Modular design with chain-based workflows  

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Python 3.13+** | Core language |
| **LangChain** | Framework for building RAG pipelines |
| **Gemini API** | LLM  |
| **Chroma** | Vector database for retrieval |

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/syedmubarish/rag_assistant_for_langchain_doc
   cd rag_assistant_for_langchain_doc

2. **Create virtual environment**
    ```
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows
    ```
3. **Install dependencies**
    pip install -r requirements.txt

4. **Set environment variables**
    Create a .env file with your keys:

    GOOGLE_API_KEY=your_api_key

5. **Initial Setup**
    Inside app.py, you’ll find a commented section:

    💡 You only need to run this part once — it loads the LangChain documentation, splits it into chunks, generates embeddings using Sentence Transformer, and stores them in the vector database (e.g., Chroma).
    Once the embeddings are created and persisted to disk, you don’t need to run this section again unless:
    You’ve added new documents
    You want to rebuild the vector store
    After the initial ingestion, comment it out again to avoid duplicates.

6. **Run the assistant**
    ```python app.py```


## ⚖️ Disclaimer
This tool is intended for educational and informational purposes only.


## 🌟 Future Enhancements

🌐 Web UI for interactive 
🔎 Assistant responds with awareness of query history

## 🧑‍💻 Author
Sayed Mubarish