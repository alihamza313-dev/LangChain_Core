from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from bs4 import BeautifulSoup


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task= "text-generation",
)

model = ChatHuggingFace(llm = llm)

url = 'https://www.datacamp.com/tutorial/docling?utm_cid=19589720824&utm_aid=157098106775&utm_campaign=230119_1-ps-other~dsa-tofu~all_2-b2c_3-apac_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=1011082-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~apac-en~dsa~tofu~tutorial~artificial-intelligence&gad_source=1&gad_campaignid=19589720824&gbraid=0AAAAADQ9WsEtwwtMddFjA2qPPAn6Zz_k9&gclid=CjwKCAjwmJjSBhB-EiwAkZgxi7LtIW_rpwlQPMqd_-NmIvIgbDj4Af76EM3q6I1I9lN1f2-KnpZXdxoCjRYQAvD_BwE'
loader = WebBaseLoader(url)

docs = loader.load()

prompt = PromptTemplate(template='Answer the questions {question} from this text {text}',
                        input_variables=['question','text'])

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'question':'what are the require Prerequisites?', 'text' : docs}))