from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from bs4 import BeautifulSoup
import os
import json
from pathlib import Path
import sys
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_BASE'] = ""
os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"
os.environ['OPENAI_API_KEY'] = ""

path = (r'C:\Users\nithin.y.c\Downloads\bookstore_output\bookstore_output\html')
folder = os.listdir(path)
textsnew = " "
for file in folder:
    if file.endswith('.html'):
        htmlpath = os.path.join(path,file)
        from langchain.document_loaders import BSHTMLLoader
        loader = BSHTMLLoader(htmlpath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
        textsnew += str(texts)
        
path1 = (r'C:\Users\nithin.y.c\Downloads\bookstore_output')
fname = (r"C:\\Users\\nithin.y.c\\Downloads\\bookstore_output\\demofile3.txt")
#f = open("C:\\Users\\nithin.y.c\\Downloads\\bookstore_output\\demofile3.txt","w")
with open(fname, 'w', encoding='utf-8') as f:
    #for item in textsnew:
    f.write(textsnew)
f.close()
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
loader = DirectoryLoader(path1 , glob="**/demofile3.txt", loader_cls=TextLoader)
text = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=2000, chunk_overlap=0)
classesdoc = text_splitter.split_documents(text)
prompt_template = """Explain the class linkage hirearchy from {classesdoc} to legible json output format"""
#prompt_template = """Generate the 'Summary of the test report:' from {texts} \n '"""
prompt = PromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
llm_chain = LLMChain(llm=llm, prompt=prompt)
resp = llm_chain.run(classesdoc)
print(resp)