#!/usr/bin/env python
# coding: utf-8


# In[2]:


import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
print(api_key)
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# # In[3]:


from openai import OpenAI

deepseek_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# # In[ ]:


from pymilvus import model as milvus_model

embedding_model = milvus_model.DefaultEmbeddingFunction()

# # In[ ]:


test_embedding = embedding_model.encode_queries(["This is a test"])[0]
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])

from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./milvus_demo.db")

collection_name = "my_rag_collection"
if milvus_client.has_collection(collection_name):
    print(f"Dropping existing collection: {collection_name}")
    milvus_client.drop_collection(collection_name)


from glob import glob

text_lines = []

for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")


from tqdm import tqdm

data = []

doc_embeddings = embedding_model.encode_documents(text_lines)

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": doc_embeddings[i], "text": line})

milvus_client.insert(collection_name=collection_name, data=data)


question = "How is data stored in milvus?"

search_res = milvus_client.search(
    collection_name=collection_name,
    data=embedding_model.encode_queries([question]),  # 将问题转换为嵌入向量
    limit=3,  # 返回前3个结果
    search_params={"metric_type": "IP", "params": {}},  # 内积距离
    output_fields=["text"],  # 返回 text 字段
)

import json

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))
