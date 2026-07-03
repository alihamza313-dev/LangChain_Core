from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader('textSplitter/BCSF24M035.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0,
    separator='$'
)
#chunk overlap show the no of characters overlap between two consecutive chunks.

result = splitter.split_documents(docs)
print (result[0].page_content)
print (result[1].page_content)