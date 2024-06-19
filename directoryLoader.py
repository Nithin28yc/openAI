from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

import os
import sys
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_BASE'] = "https://devopsvalidation.openai.azure.com/"
os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"
os.environ['OPENAI_API_KEY'] = "27940702532b4623b2c59a296bfe484b"

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader

# def write_to_text_file(filename, data):
#     try:
#         with open(filename, 'w') as file:
#             file.write(data)
#     except Exception as e:
#         print(f"An error occurred: {e}")

# Specify the file name and data you want to write
#file_name = "output.txt"
#data_to_write = "Summary: The test case sessionManagement has failed due to the application failing the Unsecure Session Management Vulnerability. This vulnerability can allow an attacker to hijack a user's session and gain unauthorized access to sensitive information."

# Call the function to write data to the text file
# write_to_text_file(file_name, data_to_write)
#    path = os.getcwd()
path = r'C:\Users\nithin.y.c\OneDrive - Accenture\Documents\Code Coverage\ct_genai_testcoverage\codeexp'

loader = DirectoryLoader(path, glob="**/code_explanation.txt", loader_cls=TextLoader)
# #loader = DirectoryLoader(r'C:\Users\nithin.y.c\OneDrive - Accenture\Documents\CaC_Code\langchain\report', glob="**/*.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
chunklen=len(texts)     
embeddings = OpenAIEmbeddings(chunk_size=1)

from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 1})
#llm = ChatOpenAI(temperature=0.0,model_kwargs={'engine': 'neww'})
llm = OpenAI(
    model_kwargs={'engine': 'devops'},
    max_tokens=2000,
 #   model_name="text-davinci-003",
    #verbose=True,
    temperature=0.0
)

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="refine", 
    retriever=retriever, 
    verbose=True
)

query="Give a descriptive 1000 words explanation with clear structure. Begin by providing an overview of what the code does, why it was written, and how it fits into project. Break down the code into smaller pieces and explain each one in a logical order. Use comments, headings to separate and label them. To show how the code works, use examples, scenarios, or demonstrations to show its tasks and how it handles inputs and outputs. Summarize code by repeating the main points and outcomes of code. highlight any challenges, limitations, or improvements from {texts}"
response = qa_stuff.run(query)
print(response)


# llm = ChatOpenAI(temperature=0.0,model_kwargs={'engine': 'development'},max_tokens=3500)
# text_splitter = CharacterTextSplitter()
# print("Reading the contents of the design document")
# loader =  DirectoryLoader(path, glob="**/code_explanation.txt", loader_cls=TextLoader)
# docs = loader.load()
# texts = text_splitter.split_documents(docs)
# print("Started generating summary from the document")
# prompt_template = """Give a descriptive 1000 words explanation from {text} with clear structure. Begin by providing an overview of what the code does, why it was written, and how it fits into project. Break down the code into smaller pieces and explain each one in a logical order. Use comments, headings to separate and label them. To show how the code works, use examples, scenarios, or demonstrations to show its tasks and how it handles inputs and outputs. Summarize code by repeating the main points and outcomes of code. highlight any challenges, limitations, or improvements"""
# PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
# chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
# doc_sum=chain({"input_documents": texts}, return_only_outputs=True)
# summary=doc_sum["output_text"]
# print('Summary:',summary)



# try:
#     loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
# #loader = DirectoryLoader(r'C:\Users\nithin.y.c\OneDrive - Accenture\Documents\CaC_Code\langchain\report', glob="**/*.pdf")
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=2000, chunk_overlap=0)
#     texts = text_splitter.split_documents(docs)
#     from langchain.chains.llm import LLMChain
#     from langchain.prompts import PromptTemplate

#     prompt_template="Give a descriptive 1000 words explanation from {texts} with clear structure. Begin by providing an overview of what the code does, why it was written, and how it fits into project. Break down the code into smaller pieces and explain each one in a logical order. Use comments, headings to separate and label them. To show how the code works, use examples, scenarios, or demonstrations to show its tasks and how it handles inputs and outputs. Summarize code by repeating the main points and outcomes of code. highlight any challenges, limitations, or improvements"
#     prompt = PromptTemplate.from_template(prompt_template)

#     llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
  
#     llm_chain = LLMChain(llm=llm, prompt=prompt)
#     resp = llm_chain.run(texts)
#     print(resp)

# except Exception as e:
#     print(e)

