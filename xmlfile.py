from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import zipfile
import sys
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_BASE'] = ""
os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"
os.environ['OPENAI_API_KEY'] = ""

from langchain.document_loaders import UnstructuredXMLLoader
loader = UnstructuredXMLLoader(r'C:\Users\nithin.y.c\OneDrive - Accenture\Documents\CaC_Code\langchain\report\index.xml')
texting = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
chunk_size=2000, chunk_overlap=0)
texts1 = text_splitter.split_documents(texting)
# print(texts)
llm = ChatOpenAI(temperature=0.0,model_kwargs={'engine': 'neww'},max_tokens=2000)

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

#prompt_template = "Explain the class hierarchy demonstrated in {texts}"
prompt_template = """Explain the class hierarchy from {texts1} to legible json output format"""
#prompt_template = "List the class hierarchy relationships between different classes in the project from {texts}"
#prompt_template = """Explain the static dependencies from the {texts} and provide the output in legible json format'"""
#prompt_template = """Explain the java class hierarchy from {texts} and provide the output in legible json format'"""
prompt = PromptTemplate.from_template(prompt_template)

llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
llm_chain = LLMChain(llm=llm, prompt=prompt)
resp = llm_chain.run(texts1)
print(resp)