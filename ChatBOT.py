# ============================================================
# 1. Load PDF Document
# ============================================================

#!pip install pypdf
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("/content/Washer_Manual.pdf")
pdf_doc = loader.load()
print(pdf_doc)
print("📄 Total Pages:", len(pdf_doc))

# ============================================================
# 2. Split Document into Chunks
# ============================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pdf_doc)
print(docs)
print("🔹 Total Chunks:", len(docs))

# ============================================================
# 3. Create Embeddings
# ============================================================

# !pip install "sentence-transformers>=2.7.0"
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunk(chunk_text):
    return embedding_model.encode([chunk_text], normalize_embeddings=True)

sample_embeddings = embed_chunk(docs[0].page_content)
print("✅ Sample Embedding Shape:", sample_embeddings.shape)

# ============================================================
# 4. Setup OpenAI + Vector Database (Chroma)
# ============================================================

# !pip install langchain_community
# !pip install -qU langchain-openai langchain-community langchain-core langchain
# !pip install -qU chromadb

from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_db = Chroma.from_documents(docs, embedding=openai_embeddings, persist_directory="/tmp/chromadb/")

chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8)

chat_history = []
chat_history.append(SystemMessage(content="You are a washing machine expert"))

vector_db._collection.get(include=['embeddings','documents'])

# ============================================================
# 5. Load HuggingFace Model for Generation
# ============================================================

from transformers import pipeline
from huggingface_hub import login
login()

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
print("✅ HuggingFace LLaMA Model Loaded")

# ============================================================
# 6. RAG Function: Retrieve + Generate Answer
# ============================================================

def retrieve_and_generate(query, threshold=1):
    search_results = vector_db.similarity_search_with_score(query, k=1)

    if not search_results or search_results[0][1] > threshold:
        return "I don't know the answer"

    retrieve_context = search_results[0][0].page_content
    similarity_score = search_results[0][1]

    print(f"Similarity Score: {similarity_score}")
    print(f"Retrieved Context: {retrieve_context}")

    prompt = f"Answer the question using the given context.\nContext: {retrieve_context}\nQuestion: {query}\nAnswer:"
    
    response = pipe(prompt, max_new_tokens=100)
    return response[0]["generated_text"]

# ============================================================
# 7. Test RAG System
# ============================================================

ques = "If the washer displays an error code related to water drainage, what troubleshooting steps should I follow before calling customer service?"
response = retrieve_and_generate(ques)
print("\nQ:", ques)
print("A:", response)

ques = "How to clean the washing machine"
response = retrieve_and_generate(ques)
print("\nQ:", ques)
print("A:", response)

# ============================================================
# 8. Interactive Chat Mode (Conversation with GPT-4o-mini)
# ============================================================

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("\n🛑 Chat Ended. Chat History:")
        print(chat_history)
        print("Have a good day 👋")
        break
    
    chat_history.append(HumanMessage(content=user_input))
    
    response = chat_model.invoke(chat_history)
    print(f"ChatBot: {response.content}")
    
    chat_history.append(AIMessage(content=response.content))
