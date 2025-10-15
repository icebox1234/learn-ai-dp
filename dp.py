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

from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./test.db")
