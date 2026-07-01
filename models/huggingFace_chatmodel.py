from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task= "text-generation",
    temperature= 0.9
)

model  =  ChatHuggingFace(llm = llm)

result = model.invoke("what is the capital of Pakistan?")

print(result.content)



