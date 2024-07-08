"""
使用千帆大模型实现QA系统
根据本地的文档（格式如pdf/txt/docx等）来回答问题
"""

import os
from dotenv import load_dotenv, find_dotenv

# 加载环境变量文件中的设置，通常包含OpenAI API密钥
_ = load_dotenv(find_dotenv())
# os.environ['QIANFAN_AK'] = ""
# os.environ['QIANFAN_SK'] = ""

# 1.Load 导入Document Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

# 加载Documents
base_dir = 'OneFlower'  # 文档的存放目录
documents = []
for file in os.listdir(base_dir): 
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# 2.Split 将Documents切分成块以便后续进行嵌入和向量存储
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

# 3.Store 将分割嵌入并存储在矢量数据库Chroma中
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

# vectorstore = Chroma.from_documents(documents=chunked_documents, embedding=QianfanEmbeddingsEndpoint())

# from langchain.vectorstores import Chroma
# from langchain.embeddings import QianfanEmbeddingsEndpoint
# 持久化向量数据库的数据
# vectorstore = Chroma.from_documents(documents=chunked_documents, embedding=QianfanEmbeddingsEndpoint(), persist_directory="./vector_store") 
# vectorstore.persist()
# 从向量数据库中加载数据
vectorstore = Chroma(persist_directory="./vector_store", embedding_function=QianfanEmbeddingsEndpoint())

# 4. Retrieval 准备模型和Retrieval链
import logging # 导入Logging工具
from langchain_community.chat_models import QianfanChatEndpoint # 千帆模型
from langchain.chains import RetrievalQA # RetrievalQA链

# 设置Logging
logging.basicConfig()
logging.getLogger('langchain.*').setLevel(logging.INFO)

# 实例化一个大模型工具 - 千帆
llm = QianfanChatEndpoint(streaming=True)
# llm = QianfanChatEndpoint(streaming=True, model="ERNIE-Bot-4")
retriever=vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.0})  # 相似度搜索文档




# 实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
# qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)



question = '介绍一下易速鲜花'

result = qa_chain({"query": question})
print(result)



# # 5. Output 问答系统的UI实现
# from flask import Flask, request, render_template
# app = Flask(__name__) # Flask APP

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':

#         # 接收用户输入作为问题
#         question = request.form.get('question')        
        
#         # RetrievalQA链 - 读入问题，生成答案
#         result = qa_chain({"query": question})
        
#         # 把大模型的回答结果返回网页进行渲染
#         return render_template('index.html', result=result)
    
#     return render_template('index.html')

# if __name__ == "__main__":
#     app.run(host='0.0.0.0',debug=True,port=5000)