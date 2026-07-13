# YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about a YouTube video's content. The application retrieves the video's transcript, processes it into searchable chunks, stores semantic embeddings in a vector database, and uses a Large Language Model (LLM) to generate context-aware answers.

This project was built to gain a practical understanding of the core concepts behind modern AI applications, including RAG pipelines, embeddings, semantic search, vector databases, and LangChain.

---

## Features

* Retrieve transcripts from YouTube videos
* Split transcripts into optimized text chunks
* Generate embeddings using Hugging Face models
* Store embeddings in a vector database
* Perform semantic similarity search
* Generate context-aware answers using an LLM
* End-to-end Retrieval-Augmented Generation workflow

---

## Project Workflow

1. Enter a YouTube video ID.
2. Retrieve the video's transcript.
3. Split the transcript into smaller chunks.
4. Generate embeddings for each chunk.
5. Store the embeddings in a vector database.
6. Retrieve the most relevant chunks based on the user's question.
7. Send the retrieved context and question to the language model.
8. Return an answer grounded in the video's transcript.

---

## Project Structure

```text
youtube-rag-chatbot/
│
├── chains/
│   └── ragChains.py
|__prompts/
|   |___prompt.py
|
├── utils/
│   ├── splitter.py
│   ├── transcript.py
│   └── vectorStore.py
│
├── main.py
└── README.md
```

---

## Tech Stack

### Backend

* Python
* LangChain

### AI & Machine Learning

* Hugging Face Inference API
* Hugging Face Embeddings
* Sentence Transformers
* Retrieval-Augmented Generation (RAG)

### Vector Database

* FAISS

### Data Source

* YouTube Transcript API

### Environment Management

* python-dotenv

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd youtube-rag-chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
```

---

## Usage

Run the application:

```bash
python main.py
```

Enter a YouTube video ID and ask questions about the video's content.

Example:

```text
Video ID:
LPZh9BOjkQs

Question:
What is self-attention?

Answer:
The video explains self-attention as...
```

---

## Learning Outcomes

This project helped me gain hands-on experience with:

* Retrieval-Augmented Generation (RAG)
* LangChain
* Prompt orchestration
* Semantic search
* Embedding models
* Vector databases
* Document chunking
* YouTube transcript processing
* LLM integration

---

## Future Improvements

* FastAPI backend
* Web-based user interface
* Chrome extension for YouTube
* Chat history
* Timestamp-aware responses
* PDF and Markdown export
* User authentication
* Persistent vector storage
* Multi-user support
* Conversation memory

---
