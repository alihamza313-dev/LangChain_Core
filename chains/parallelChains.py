from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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

template1 = PromptTemplate(
    template="give five important points from this text {text}",
    input_variables=['text']
)

template2 = PromptTemplate(
    template="Make three questions for quiz from this text {text}",
    input_variables=['text']
)

template3 = PromptTemplate(
    template="Merge the provided points {points} and quiz {quiz} in to a single clean document",
    input_variables=['points','quiz']
)

parser = StrOutputParser()

parallelTasks = RunnableParallel(
    {
        'points' : template1 | model1 | parser,
        'quiz' : template2 | model2 | parser
    }
)
finalTask = template3 | model2 | parser

mergeTasks = parallelTasks | finalTask

text = """Accept (some) Repetition In Documentation.

If you want to write good code, Don’t Repeat Yourself. But if you adhere strictly to this DRY principle when writing documentation, you won’t get far. Some amount of business logic described by your code must be described again in your documentation.

In an ideal world, an automated system would generate documentation from the software’s source code, and the system would be smart enough to generate good documentation without any additional input. Unfortunately, today, the best documentation is hand-written, which means that just by writing any documentation, you are repeating yourself. Sure, documentation generators exist and are useful, but it’s important to acknowledge that they still require input from humans to function.

The pursuit of minimizing repetition remains valiant! ARID does not mean WET, hence the word choice. It means: try to keep things as DRY as possible but also recognize that you’ll inevitably need some amount of “moisture” to produce documentation.

Cultivating an awareness of this inconvenient truth will hopefully be a helpful step toward reminding developers that a need often exists to update documentation along with code."""

result = mergeTasks.invoke({'text' : text})
print(result)

