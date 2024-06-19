from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
import os
import json
from pathlib import Path
import sys
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_BASE'] = ""
os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"
os.environ['OPENAI_API_KEY'] = ""

#with zipfile.ZipFile('Dev_Workspace_DefaultProject_cactest_Hosting_Cloud_Infra_Compliance_Ports_Management_5_Ports_Management.zip', 'r') as zip_ref:
# with zipfile.ZipFile('Dev_Workspace_DefaultProject_cactest_Code_OSS_review_9_OSS_review.zip', 'r') as zip_ref:
#     zip_ref.extractall()
# print('Zip file extracted successfully.')

# path = os.getcwd()
controlStatus = sys.argv[1]
path = (r'C:\Users\nithin.y.c\OneDrive - Accenture\Documents\CaC_Code\langchain\multiplereport')
#path = (r'C:\Users\nithin.y.c\Downloads\bookstore_output\bookstore_output\html')
#print(path)
folder = os.listdir(path)
#print(folder)
textsnew = [""]
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
      #  result_string = '\n'.join(texts)
        #textsnew.append(texts)
        #print(result_string)
        if controlStatus==('Non-Compliant'):
            prompt_template = """Generate the overall 'Summary of the test report:' {texts}
                        and Generate the 'Remediation for how to fix the issue:'"""       
        if controlStatus==('Compliant'):
            #prompt_template = """Explain the class hierarchy from {texts1} to legible json output format"""
            prompt_template = """Explain about the {texts} in one line opening statement about it and Generate the issue summary of it in numbered points within 80 words from : {texts}
                            """
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        resp = llm_chain.run(texts)
        print(resp)

    if file.endswith('.txt'):
        txtpath = os.path.join(path,file)
        print(txtpath)
        from langchain.document_loaders import DirectoryLoader
        from langchain.document_loaders import TextLoader
        loader = DirectoryLoader(path , glob="**/*.txt", loader_cls=TextLoader)
        text = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0)
        texts2 = text_splitter.split_documents(text)
        if controlStatus==('Non-Compliant'):
            prompt_template = """Generate the overall 'Summary of the test report:' {texts2}
                        and Generate the 'Remediation for how to fix the issue:'"""      
        if controlStatus==('Compliant'):
            #prompt_template = """Explain the class hierarchy from {texts2} to legible json output format"""
            prompt_template = """Explain about the {texts2} in one line opening statement about it and Generate the issue summary in numbered points within 80 words from {texts2}"""
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=1, model_kwargs={'engine': 'neww'})
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        resp = llm_chain.run(texts2)
        print(resp)

    if file.endswith('.pdf'):
        pdfpath = os.path.join(path,file)
        print(pdfpath)
        from langchain.document_loaders.pdf import PyPDFLoader
        loader = PyPDFLoader(pdfpath)
        pdf = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0)
        texts3 = text_splitter.split_documents(pdf)  

        if controlStatus==('Non-Compliant'):
            prompt_template = """Generate the overall 'Summary of the test report:' {texts3}
                        and Generate the 'Remediation for how to fix the issue:'"""
        if controlStatus==('Compliant'):
            prompt_template = """Generate one line opening statement about it and summary in numbered points within 80 words from :{texts3}
                            """
        prompt = PromptTemplate.from_template(prompt_template)

        llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        resp = llm_chain.run(texts3)
        print(resp)

    if file.endswith('.json'):
        jsonpath = os.path.join(path,file)
        print(jsonpath)
        from langchain.document_loaders.json_loader import JSONLoader
    #    loader = JSONLoader(file_path=jsonpath, jq_schema='.messages[].content', json_lines=True )
        data = json.loads(Path(jsonpath).read_text())
        json = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0)
        texts4 = text_splitter.split_documents(json)
        # print(texts)
        if controlStatus==('Non-Compliant'):
            prompt_template = """Generate the overall 'Summary of the test report:' {texts4}
                        and Generate the 'Remediation for how to fix the issue:'"""
        if controlStatus==('Compliant'):
            prompt_template = """Explain about the {texts4} in one line opening statement about it and Generate the issue summary in numbered points within 80 words from :{texts4}
                            '"""
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        resp = llm_chain.run(texts4)
        print(resp)

    if file.endswith('.xml'):
        xmlpath = os.path.join(path,file)
        print(xmlpath)
        from langchain.document_loaders import UnstructuredXMLLoader
        loader = UnstructuredXMLLoader(xmlpath)
        xml = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0)
        texts5 = text_splitter.split_documents(xml)
        # print(texts)
        if controlStatus==('Non-Compliant'):
            prompt_template = """Generate the overall 'Summary of the test report:' {texts5}
                        and Generate the 'Remediation for how to fix the issue:'"""
        if controlStatus==('Compliant'):
            prompt_template = """Explain about the {texts5} in one line opening statement about it and Generate the issue summary in numbered points within 80 words from :{texts5}
                            """
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        resp = llm_chain.run(texts5)
        print(resp)

    if file.endswith('.csv'):
        csvpath = os.path.join(path,file)
        print(csvpath)
        from langchain.document_loaders import CSVLoader
        loader = CSVLoader(csvpath)
        csv = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0)
        texts6 = text_splitter.split_documents(csv)
        # print(texts)
        if controlStatus==('Non-Compliant'):
            prompt_template = """Generate the overall 'Summary of the test report:' {texts6}
                        and Generate the 'Remediation for how to fix the issue:'"""
        if controlStatus==('Compliant'):
            prompt_template = """Explain about the {texts6} in one line opening statement about it and Generate the issue summary in numbered points within 80 words from : {texts6} \n
                            '"""
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        resp = llm_chain.run(texts6)
        print(resp)

# path1 = (r'C:\Users\nithin.y.c\Downloads\bookstore_output')
# fname = 'C:\\Users\\nithin.y.c\\Downloads\\bookstore_output\\demofile3.txt'
#f = open("C:\\Users\\nithin.y.c\\Downloads\\bookstore_output\\demofile3.txt","w")
# with open(fname, 'w') as f:
#     for item in textsnew:
#         f.write(item + '\n')
# for item in textsnew:
#     # write each item on a new line
#     f.write("%s\n" % item)

#f.write("".join(string(textsnew)))
# f.close()
# from langchain.document_loaders import DirectoryLoader
# from langchain.document_loaders import TextLoader
# loader = DirectoryLoader(path1 , glob="**/*.txt", loader_cls=TextLoader)
# text = loader.load()
#new = "".join(map(str, textsnew))
# text_splitter = RecursiveCharacterTextSplitter(
# chunk_size=2000, chunk_overlap=0)
# classesdoc = text_splitter.split_text(text)
# prompt_template = """Explain the class hierarchy included in {classesdoc} to legible json output format"""
#prompt_template = """Generate the 'Summary of the test report:' from {texts} \n '"""
# prompt = PromptTemplate.from_template(prompt_template)
# llm = ChatOpenAI(temperature=0, model_kwargs={'engine': 'neww'})
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# resp = llm_chain.run(classesdoc)
# print(resp)