# ============================================================
# 1. Set API Keys
# ============================================================

import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langchain-api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "query"

# ============================================================
# 2. Install Required Libraries
# ============================================================

pip install -qU langchain-openai langchain-community chromadb langchain tiktoken langchainhub

# ============================================================
# 3. Enable LangSmith Tracing
# ============================================================

from langsmith import utils
utils.tracing_is_enabled()

# ============================================================
# 4. Load the Document
# ============================================================

from langchain_community.document_loaders import WebBaseLoader
import bs4

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
# 5. Split Document
# ============================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# ============================================================
# 6. Create Vector DB
# ============================================================

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vect = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="/content/db")
retriever = vect.as_retriever()

chat_model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0
)

# ============================================================
# 7. Generate Query Variations
# ============================================================

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """You are an AI language model assistant. Your task is to generate the 5 different versions of the given user question."""
prompt_perspective = ChatPromptTemplate.from_template(template)
generate_queries = prompt_perspective | chat_model | StrOutputParser() | (lambda y: y.split("\n"))
print(generate_queries)

# ============================================================
# 8. Deduplicate Documents
# ============================================================

from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    flattened_docs = []
    for sublist in documents:
        for doc in sublist:
            flattened_docs.append(dumps(doc))
    unique_docs = list(set(flattened_docs))
    loading_docs = []
    for doc in unique_docs:
        loading_docs.append(loads(doc))
    return loading_docs

# ============================================================
# 9. Retriever Chain
# ============================================================

question = "What is task decomposition for LLM agents?"
retriever_chain = generate_queries | retriever.map() | get_unique_union
response = retriever_chain.invoke({"Question": question})
len(response)
print(response)

# ============================================================
# 10. Final RAG Chain
# ============================================================

from operator import itemgetter

template = """Answer the following question based on this context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
final_rag_chain = (
    {"context": retriever_chain,
     "question": itemgetter("question")}
    | prompt
    | chat_model
    | StrOutputParser()
)

final_rag_chain.invoke({"question": question})