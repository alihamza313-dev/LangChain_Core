from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "zai-org/GLM-5.2",
    task= "text-generation",
)

model1 = ChatHuggingFace(llm = llm)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task = "text-generation",
)

model2 = ChatHuggingFace(llm = llm)

class Feedback(BaseModel):
    sentiment : Literal['Positive' , 'Negative'] = Field(description= "provide the sentiment about the feedback positive or negative")


parser1 = PydanticOutputParser(pydantic_object=Feedback)

template1 = PromptTemplate(
    template="give the sentiment positive or negative about the following feedback \n {feedback} \n {format_instruction}",
    input_variables=['format_instruction'],
    partial_variables={"format_instruction" : parser1.get_format_instructions()}
)

chain = template1 | model1 | parser1

# result = chain.invoke({'feedback' : 'This is wonderful mobile phone'})

# print (result)

parser2 = StrOutputParser()

template2 = PromptTemplate(
    template="write the proper response for the positive feedback {feedback}",
    input_variables=['feedback']
)

template3 = PromptTemplate(
    template="write the proper response for the negative feedback {feedback} by yourself",
    input_variables=['feedback']
)

conditionalTasks = RunnableBranch(
    (lambda x : x.sentiment == 'Positive' , template2 | model2 | parser2),
    (lambda x : x.sentiment == 'Negative' , template3 | model2 | parser2),
    RunnableLambda(lambda x : "Cannot find appropriate sentiment")
)

final = chain | conditionalTasks

result = final.invoke({'feedback' : 'This is terrible mobile phone'})

print (result)

final.get_graph().print_ascii() #This is for the visual representation