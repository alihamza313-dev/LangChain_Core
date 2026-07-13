from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
from utils.vectorStore import retrieve_docs
from langchain_core.output_parsers import StrOutputParser
from prompt.prompt import prompt

def getContext(retrieve_docs):
    return "\n".join(doc.page_content for doc in retrieve_docs)

parser = StrOutputParser()

def buildChains(model,retreiver):

    chain1  = RunnableParallel(
        {
            'context' : retreiver | RunnableLambda(getContext),
            'question' : RunnablePassthrough()
        }
    )

    chain2 = prompt | model  | parser

    finalChain = chain1 | chain2

    return finalChain