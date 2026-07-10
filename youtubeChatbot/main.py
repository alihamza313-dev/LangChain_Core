from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled,RequestBlocked

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task= "text-generation",
)

model = ChatHuggingFace(llm = llm)

#First get a transcript of a youtube video

transcript = None
video_id = "LPZh9BOjkQs"
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id , languages=['en'])

    transcript = ' '.join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("No captions available for this video")
    exit()
except RequestBlocked:
    print("YouTube blocked the request. Try again later or use another network.\n")
    exit()

#Now make a chunks for this transcript

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

if(transcript):
    chunks = splitter.create_documents([transcript])
else:
    print("Cannot continue without transcript.")
    exit()

#Now make embeddings of these chunks and then store this into the vector store

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks,embeddings)

#now make a retreiver to retreive the relevant documents according to the questions or query

retreiver = vector_store.as_retriever(search_type = "mmr" , search_kwargs = {'k' : 4,'fetch_k':6})

# Here 'k' means how many documents/chunks you finally want to return.
# And fetch_k is used with the search type mmr (maximum marginal relevance) which fetch the 6 related document and then fetch again from them top k documents having less repetitions and more unique. 

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

question = "About which topic this video is about?"

retrieve_docs = retreiver.invoke(question)

context = "\n".join(doc.page_content for doc in retrieve_docs)

# Now we have the transcript context, which contains the relevant document chunks
# retrieved from our vector store based on the user's question.
# In the augmentation phase, we combine the retrieved context with the question
# to create the final prompt that will be sent to the LLM.

final_prompt = prompt.invoke({'question':question , 'context' : context})

answer = model.invoke(final_prompt)

print("\n",answer.content)
