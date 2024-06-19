from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.document_loaders import BSHTMLLoader
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
#from IPython.display import display, Markdown
import os

os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_BASE'] = ""
os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"
os.environ['OPENAI_API_KEY'] = ""

# openai.api_type = "azure"
# openai.api_base = "https://devopsvalidation.openai.azure.com/"
# openai.api_version = "2023-03-15-preview"
# openai.api_key = '27940702532b4623b2c59a296bfe484b'
 
# loader = BSHTMLLoader("ZAP_Report.html")
# data = loader.load()
# print(data)

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader(r'C:\Users\nithin.y.c\OneDrive - Accenture\Documents\CaC_Code\langchain\report\SSL_Enablement_Guide.pdf')
docs = loader.load()

# from langchain.document_loaders import BSHTMLLoader
# loader = BSHTMLLoader(".\ZAP_Report.html")
# docs = loader.load()
# print(data)

print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(
chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

# import json
# from pathlib import Path
# from pprint import pprint


# file_path='./SnykVuln.json'
# data = json.loads(Path(file_path).read_text())

# pprint(data)
# print(len(data))


#text = "What would be a good company name for a company that makes colorful socks?"

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=100, chunk_overlap=0
# )
# texts = text_splitter.split_text(data)
# print(texts)

# text_splitter = RecursiveCharacterTextSplitter(
# chunk_size=2000, chunk_overlap=0)



embeddings = OpenAIEmbeddings(chunk_size=1)

from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

#llm = ChatOpenAI(temperature=0.0,model_kwargs={'engine': 'neww'},max_tokens=2000)
llm = OpenAI(
    model_kwargs={'engine': 'devops'},
    max_tokens=2000,
 #   model_name="text-davinci-003",
    #verbose=True,
    temperature=0.0
)

# llm = AzureOpenAI(
#     model= "neww",
#     max_tokens=2000,
#     model_name="text-davinci-003",
#     #verbose=True,
#     temperature=0.0
# )

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="refine", 
    retriever=retriever, 
    verbose=True
)

query="Explain Steps to create a record set"
#query="Generate the issue summary"
response = qa_stuff.run(query)
print(response)
