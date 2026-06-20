# Context-Aware Chatbot Using LangChain & RAG

## Objective

Build a conversational AI chatbot capable of:

- Remembering conversation history
- Retrieving relevant information from documents
- Answering context-aware questions

---

## Technologies

- LangChain
- FAISS
- OpenAI GPT
- Sentence Transformers
- Streamlit

---

## Features

### Retrieval-Augmented Generation (RAG)

Retrieves relevant document chunks before generating responses.

### Conversational Memory

Stores previous chat history using ConversationBufferMemory.

### Vector Search

Uses FAISS for semantic similarity search.

### Streamlit Deployment

Provides an interactive web interface.

---

## Workflow

Documents
    ↓
Text Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
LLM
    ↓
Answer

---

## Skills Gained

- Conversational AI Development
- LangChain Framework
- Document Embeddings
- Vector Databases
- Retrieval-Augmented Generation
- Context Memory
- Streamlit Deployment

---

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
