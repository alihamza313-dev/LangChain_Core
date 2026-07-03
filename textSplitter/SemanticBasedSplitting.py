from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

text = """
Python is a programming language.It is used for web development and AI.Machine learning is a field of AI.
It focuses on learning from data.
Deep learning uses neural networks.
It is a subset of machine learning.
"""

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = SemanticChunker(embeddings)

chunks = splitter.split_text(text)

print(len(chunks))
for i,chunk in enumerate(chunks):
    print(f"\nChunk {i+1}")
    print(chunk)