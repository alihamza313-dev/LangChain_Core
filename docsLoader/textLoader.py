from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task= "text-generation",
)

model = ChatHuggingFace(llm = llm)
loader = TextLoader('data.txt',autodetect_encoding=True)

#"A text file is stored as bytes using a particular encoding. To read the file correctly, the program must use the same encoding that was used when the file was saved. If it doesn't know the encoding, it can either fail or produce incorrect text. autodetect_encoding=True helps by trying to identify the correct encoding automatically."

docs = loader.load()

# Loads the text file and returns a list of Document objects.
# Each Document has:
# - page_content: the actual text from the file
# - metadata: information about the file (e.g., source path)
#=======================
# print(type(docs))
# print(len(docs))
# print(docs[0])

#=======================
template = PromptTemplate(
    template="Write a summary in five lines for this document data {docs}",
    input_variables=['docs']
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'docs' : docs})

print(result)