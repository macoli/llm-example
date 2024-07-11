# 加载环境变量
from dotenv import load_dotenv, find_dotenv
import os
from langchain_community.chat_models import QianfanChatEndpoint, ChatBaichuan
from langchain_community.embeddings import QianfanEmbeddingsEndpoint, BaichuanTextEmbeddings

_ = load_dotenv(find_dotenv())

# 导入ChatOpenAI类和OpenAIEmbeddings类，用于初始化大语言模型
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# 导入Chroma类，用于建立长时记忆的向量数据库
from langchain_community.vectorstores import Chroma
# 导入Document类，用于定义文档结构
from langchain.schema import Document

# 导入AutoGPT类，用于实现对话代理
from Agent.AutoGPT import AutoGPT
# 导入工具模块，用于扩展代理的功能
from Tools import *
# 导入PythonTool模块，特别是ExcelAnalyser类，用于分析Excel文件
from Tools.PythonTool import ExcelAnalyser


# 定义启动代理函数，负责与用户进行交互
def launch_agent(agent: AutoGPT):
    """
    启动对话代理，与用户进行循环对话，直到用户选择退出。

    参数:
    - agent: AutoGPT类型的对象，对话代理实例。
    """
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"
    while True:
        # 获取用户输入，并检查是否要退出
        task = input(f"{ai_icon}：有什么可以帮您？\n{human_icon}：")
        if task.strip().lower() == "quit":
            break
        # 使用代理处理用户任务，并打印回复
        reply = agent.run(task, verbose=True)
        print(f"{ai_icon}：{reply}\n")


# 定义主函数，用于初始化并启动对话代理
def main():
    """
    主函数，负责初始化对话代理并启动交互。
    """
    # 初始化大语言模型（LLM），使用gpt-4-1106-preview模型，设置随机种子为42
    # llm = ChatOpenAI(
    #     model="gpt-4-1106-preview",
    #     temperature=0,
    #     model_kwargs={
    #         "seed": 42
    #     },
    # )
    # 初始化长时记忆向量数据库，并配置为检索器
    # db = Chroma.from_documents([Document(page_content="")], OpenAIEmbeddings(model="text-embedding-ada-002"))

    # 初始化大语言模型（LLM）， 默认是ERNIE-Bot-turbo
    # llm = QianfanChatEndpoint(
    #     model="ERNIE-Bot-turbo",
    #     streaming=True
    # )
    #
    # # 初始化长时记忆向量数据库，并配置为检索器
    # db = Chroma.from_documents(documents=[Document(page_content="")], embedding=QianfanEmbeddingsEndpoint())

    # 初始化大语言模型（LLM），使用Baichuan4模型，设置随机种子为42
    llm = ChatBaichuan(
        model="Baichuan4",
        temperature=0,
        model_kwargs={
            "seed": 42
        },
        streaming=True,
    )

    # 初始化长时记忆向量数据库，并配置为检索器
    db = Chroma.from_documents([Document(page_content="")], BaichuanTextEmbeddings())

    retriever = db.as_retriever(
        search_kwargs={"k": 1}
    )
    # 初始化自定义工具集具，用于扩展LLM的功能，包括文档问答、文档生成、邮件处理、Excel分析等
    tools = [
        document_qa_tool,
        document_generation_tool,
        email_tool,
        excel_inspection_tool,
        directory_inspection_tool,
        finish_placeholder,
        ExcelAnalyser(
            prompts_path="./prompts/tools",
            prompt_file="excel_analyser.json",
            verbose=True
        ).as_tool()
    ]
    # 初始化智能体，配置LLM、prompt路径、工具集合、工作目录等参数
    agent = AutoGPT(
        llm=llm,
        prompts_path="./prompts/main",
        tools=tools,
        work_dir="./data",
        main_prompt_file="main.json",
        final_prompt_file="final_step.json",
        max_thought_steps=20,
        memory_retriever=retriever
    )
    # 启动智能体
    launch_agent(agent)


if __name__ == "__main__":
    main()
