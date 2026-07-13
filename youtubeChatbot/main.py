from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

from utils.splitter import split_text
from utils.transcript import get_transcript
from utils.vectorStore import retrieve_docs
from chains.ragChains import buildChains

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task= "text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm = llm)


video_id = "LPZh9BOjkQs"

transcript = get_transcript(video_id)

if transcript is None:
    print("unable to get transcript!")
    exit()

chunks = split_text(transcript)

retreiver = retrieve_docs(chunks)

finalChain = buildChains(model,retreiver)


question = input("Ask a question : ")
print("-------------------------------------------------")
print("Answer : ")
answer = finalChain.invoke(question)
print(answer)


