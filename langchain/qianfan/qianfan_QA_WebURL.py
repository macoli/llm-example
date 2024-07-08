
# 根据Web URL的内容回答问题

import os
from dotenv import load_dotenv, find_dotenv

# 加载环境变量文件中的设置，通常包含OpenAI API密钥
_ = load_dotenv(find_dotenv())

# os.environ['QIANFAN_AK'] = ""
# os.environ['QIANFAN_SK'] = ""

# 此处为 Langsmith 相关功能开关。当且仅当你知道这是做什么用时，可删除注释并设置变量以使用 Langsmith 相关功能
# os.environ['LANGCHAIN_TRACING_V2'] = "true"
# os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
# os.environ['LANGCHAIN_API_KEY'] = "your_langchian_api_key"
# os.environ['LANGCHAIN_PROJECT'] = "your_project_name"

is_chinese = True

if is_chinese:
    WEB_URL = "https://zhuanlan.zhihu.com/p/85289282"
    CUSTOM_PROMPT_TEMPLATE = """
        使用下面的语料来回答本模板最末尾的问题。如果你不知道问题的答案，直接回答 "我不知道"，禁止随意编造答案。
        为了保证答案尽可能简洁，你的回答必须不超过三句话。
        请注意！在每次回答结束之后，你都必须接上 "感谢你的提问" 作为结束语
        以下是一对问题和答案的样例：
            请问：秦始皇的原名是什么
            秦始皇原名嬴政。感谢你的提问。
        
        以下是语料：
        
        {context}
        
        请问：{question}
    """
    QUESTION1 = "明朝的开国皇帝是谁"
    QUESTION2 = "朱元璋是什么时候建立的明朝"
else:
    WEB_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    CUSTOM_PROMPT_TEMPLATE = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum and keep the answer as concise as possible. 
        Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:
    """
    QUESTION1 = "How do agents use Task decomposition?"
    QUESTION2 = "What are the various ways to implemet memory to support it?"



# 1、加载数据。从web的文档中加载数据，加载成Document
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(WEB_URL) 
data = loader.load()


# 2、Split。把Document 切分成块
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 384, chunk_overlap = 0, separators=["\n\n", "\n", " ", "", "。", "，"])
all_splits = text_splitter.split_documents(data)


# 3、Storage。将上面切分的块存储到向量数据库中
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits, embedding=QianfanEmbeddingsEndpoint())


# 4、Retrieve。用上面的向量数据库根据问题进行相似度搜索
# 基于prompt问题查找相似文档
print("prompt问题："+QUESTION1)  
docs = vectorstore.similarity_search_with_relevance_scores(QUESTION1)
# [(document.page_content, score) for document, score in docs]


# 5、Generate。将问题和从向量数据库中查找到的数据一同传递给大模型
from langchain.chains import RetrievalQA
from langchain.chat_models import QianfanChatEndpoint
from langchain.prompts import PromptTemplate

QA_CHAIN_PROMPT = PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)  # 模版

llm = QianfanChatEndpoint(streaming=True)  # 大模型，默认是ERNIE-Bot-turbo
retriever=vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.0})  # 相似度搜索文档
                                   
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
print(qa_chain({"query": QUESTION1}))

# 使用参数return_source_documents，可以将把问题参考的文档内容也返回
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, return_source_documents=True)
result = qa_chain({"query": QUESTION1})
# len(result['source_documents'])
# result['source_documents']
print(result)



# 6、Chat。通过加入 Memory 模块并替换使用 ConversationalRetrievalChain 来实现记忆化的对话式查询
# from langchain.memory import ConversationSummaryMemory
# from langchain.chains import ConversationalRetrievalChain

# memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
# qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT})
# print(qa(QUESTION1))
# print(qa(QUESTION2))