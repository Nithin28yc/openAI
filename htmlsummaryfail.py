from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import zipfile
import sys
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_BASE'] = ""
os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"
os.environ['OPENAI_API_KEY'] = ""

#with zipfile.ZipFile('Dev_Workspace_DefaultProject_cactest_Hosting_Cloud_Infra_Compliance_Ports_Management_5_Ports_Management.zip', 'r') as zip_ref:
# with zipfile.ZipFile('Dev_Workspace_DefaultProject_cactest_App_Sec_Controls_Session_Management_1_Session_Management.zip', 'r') as zip_ref:
#     zip_ref.extractall()
# print('Zip file extracted successfully.')

#path = os.getcwd()
path = (r'C:\Users\nithin.y.c\OneDrive - Accenture\Documents\CaC_Code\langchain\projectsummary')
folder = os.listdir(path)

for file in folder:
    # if file.endswith('.html'):
    #     htmlpath = os.path.join(path,file)
    #     print(htmlpath)
    #     from langchain.document_loaders import BSHTMLLoader
    #     loader = BSHTMLLoader(htmlpath)
    #     docs = loader.load()

    #     text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=2000, chunk_overlap=0)
    #     texts = text_splitter.split_documents(docs)

    if file.endswith('.txt'):
        txtpath = os.path.join(path,file)
        print(txtpath)

        from langchain.document_loaders import DirectoryLoader
        from langchain.document_loaders import TextLoader
        loader = DirectoryLoader(path , glob="**/*.txt", loader_cls=TextLoader)
        text = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(text)
try:

    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    controlStatus = sys.argv[1]
    if controlStatus==('Non-Compliant'):
        prompt_template = """Generate the overall 'Summary of the test report:' {texts}
                        and Generate the 'Remediation for how to fix the issue:'"""
        
    if controlStatus==('Compliant'):
        #prompt_template = """Explain about the {texts} in one line opening statement about it in 20 words and Generate the issue summary under it in numbered points within 80 words from : {texts}'"""
        prompt_template = """Provide issue summary in a line with 30 words and create a overall bullet-pointed list of vulnerabilities, create a bullet-pointed list of overall Remediation steps recommended for the isues and generate a overall conclusion from {texts}"""
    prompt = PromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    resp = llm_chain.run(texts)
    print(resp)

except Exception as e:
    print(e)