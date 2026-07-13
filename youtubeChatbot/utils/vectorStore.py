from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def retrieve_docs(text):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    vector_store = FAISS.from_documents(text,embeddings)

    #now make a retreiver to retreive the relevant documents according to the questions or query

    retreiver = vector_store.as_retriever(search_type = "mmr" , search_kwargs = {'k' : 4,'fetch_k':6})
    return retreiver
