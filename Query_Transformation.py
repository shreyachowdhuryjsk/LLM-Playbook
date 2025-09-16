# ============================================================
# 1. Load Libraries
# ============================================================

import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langchain-api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "query"
pip install -qU langchain-openai langchain-community chromadb langchain tiktoken langchainhub
from langsmith import utils
utils.tracing_is_enabled()
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# 2. Load the Document
# ============================================================

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
print(docs)

# ============================================================
# 3. Split Document
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000 ,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)

# ============================================================
# 4. Create Vector DB
# ============================================================

from langchain_openai import OpenAIEmbeddings , ChatOpenAI
embedding_model = OpenAIEmbeddings (model = "text-embedding-3-small")
vect = Chroma.from_documents (documents = chunks , embedding = embedding_model , persist_directory = "/content/db")
retriever = vect.as_retriever()

# ============================================================
# 5. Setup Prompt and Model
# ============================================================

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template ("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
""")
chat_model = ChatOpenAI (
    model_name = "gpt-4o-mini" ,
    temperature = 0.0
)

# ============================================================
# 6. Format Docs
# ============================================================

def format_docs(chunks):
  for i in docs:
    return "\n\n" .join (i.page_content)

# ============================================================
# 7. Create Chains
# ============================================================

from langchain.schema.runnable import RunnablePassthrough
answer_chain = prompt | chat_model | StrOutputParser()
rewrite_prompt = ChatPromptTemplate.from_template(
"""You are a helpful assistant that improves user queries for information retrieval.
Given a user's input, rewrite it into a clear, standalone question that can be used to retrieve relevant documents.
Remove any conversational fluff or non-essential context.
User input:
{question}
Rewritten question:""")
rewrite_chain = rewrite_prompt | chat_model | StrOutputParser()

# ============================================================
# 8. Run Query
# ============================================================

query = "I am a technology enthusiast, I take a lot of trainings & right now learning langchain. can you tell what is task decomposition for LLM agents?"
rewritten_query = rewrite_chain.invoke({"question" : query})
print("Rewritten Query : " , rewritten_query)
retrieve_docs = retriever.invoke(rewritten_query)
context = format_docs(retrieve_docs)
answer = answer_chain.invoke({"question" : rewritten_query , "context"  : context})
print("Original Query : " , query)
print("Answer : " , answer)