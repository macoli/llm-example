# 没有openai接口，暂时不能运行
import openai
import faiss
import numpy as np

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import json

_ = load_dotenv(find_dotenv())

client = OpenAI()

# 设置 OpenAI API 密钥
openai.api_key = 'your-api-key'

# 示例文档
documents = [
    "This is the content of document 1.",
    "This is the content of document 2.",
    "This is some additional information in document 3."
]

# 将文档向量化
def get_embeddings(texts):
    response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
    embeddings = [item['embedding'] for item in response['data']]
    return np.array(embeddings)

# 创建向量数据库（Faiss 索引）
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# 检索最相似的文档
def search_similar_docs(index, query_embedding, k=2):
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0]

# 生成响应
def generate_response(query, docs):
    prompt = f"Query: {query}\n\nRetrieved Docs:\n" + "\n".join(docs) + "\n\nResponse:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# 向量化文档
doc_embeddings = get_embeddings(documents)

# 创建并添加到 Faiss 索引
index = create_faiss_index(doc_embeddings)

# 示例查询
query = "What information is in document 1?"
query_embedding = get_embeddings([query])[0]

# 检索最相似的文档
similar_doc_indices = search_similar_docs(index, query_embedding)
retrieved_docs = [documents[i] for i in similar_doc_indices]

# 生成响应
response = generate_response(query, retrieved_docs)
print("Response:", response)
