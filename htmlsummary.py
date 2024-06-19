from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
#from IPython.display import display, Markdown
import os
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_BASE'] = ""
os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"
os.environ['OPENAI_API_KEY'] = ""

import zipfile
with zipfile.ZipFile('Dev_Workspace_DefaultProject_cactest_App_Sec_Controls_Error_handling_2_Error_Handling.zip', 'r') as zip_ref:
    zip_ref.extractall()
print('Zip file extracted successfully.')

path = os.getcwd()
folder = os.listdir(path)
for file in folder:
    if file.endswith('.html'):
        htmlpath = os.path.join(path,file)
        print(htmlpath)

from langchain.document_loaders import BSHTMLLoader
loader = BSHTMLLoader(htmlpath)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

# Define prompt
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
prompt_template = """Generate the overall "Summary of the test report:"\n
                         {texts}\n
                   """
prompt = PromptTemplate.from_template(prompt_template)

llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
llm_chain = LLMChain(llm=llm, prompt=prompt)
resp = llm_chain.run(texts)
print(resp)
# embeddings = OpenAIEmbeddings(chunk_size=1)

# from langchain.vectorstores import Chroma
# db = Chroma.from_documents(texts, embeddings)

# retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})

# llm = ChatOpenAI(temperature=0.0,model_kwargs={'engine': 'neww'},max_tokens=2000)

# # llm = OpenAI(
# #     model_kwargs={'engine': 'devops'},
# #     max_tokens=2000,
# #  #   model_name="text-davinci-003",
# #     #verbose=True,
# #     temperature=0.0
# # )

# qa_stuff = RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="refine", 
#     retriever=retriever, 
#     verbose=True
# )

# query="Generate the overall summary of the test report"
# response = qa_stuff.run(query)
# print(response)