from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""You are a helpful assistant.
Answer the question only using the provided transcript context.
Context:
{context}
Question:
{question}
If the context does not contain enough information to answer, say:
"I do not understand from the given context because it is insufficient."
""",
    input_variables=["question", "context"]
)