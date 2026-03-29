import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import shutil
import json
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

STORAGE_DIR = "server_storage"
VECTOR_INDEX_DIR = os.path.join(STORAGE_DIR, "faiss_index")
os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(title="RAG and Data Agent API", version="1.0.0")

global_vectorstore = None
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

@app.get("/")
async def root():
    return {"status": "online", "service": "RAG & Data Agent API v1.0"}

@app.get("/health")
async def health():
    return {"status": "ok", "vectorstore_loaded": global_vectorstore is not None}

@app.on_event("startup")
def startup_event():

    global global_vectorstore
    if os.path.exists(VECTOR_INDEX_DIR):
        try:
            global_vectorstore = FAISS.load_local(VECTOR_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            print("Loaded FAISS index.")
        except Exception as e:
            print(f"No FAISS index found. {e}")

# API Schemas
class ChatRequest(BaseModel):
    query: str
    provider: str
    api_key: str
    model_name: str
    temperature: float = 0.2
    use_fallback: bool = False
    chat_history: List[dict] = []

class ChatResponse(BaseModel):
    response: str

class FinanceAnalysisRequest(BaseModel):
    data_summary: str
    provider: str
    api_key: str
    model_name: str

class FinanceAnalysisResponse(BaseModel):
    analysis: str

def get_llm(provider: str, model_name: str, api_key: str, temperature: float = 0.2):
    if provider == "OpenAI":
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)
    elif provider == "Claude":
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=temperature)
    elif provider == "Groq":
        return ChatGroq(model=model_name, api_key=api_key, temperature=temperature)
    raise ValueError(f"Unknown provider {provider}")

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Process uploaded PDFs and update FAISS."""
    global global_vectorstore
    raw_text = ""
    for file in files:
        temp_file_path = f"{STORAGE_DIR}/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        reader = PdfReader(temp_file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                raw_text += extracted + "\n"
        os.remove(temp_file_path)
    
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text found in PDFs.")
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128, length_function=len)
    chunks = splitter.split_text(raw_text)
    
    new_vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    new_vectorstore.save_local(VECTOR_INDEX_DIR)
    
    global_vectorstore = new_vectorstore
    return {"message": "Success", "chunks": len(chunks)}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Execute RAG chat."""
    global global_vectorstore
    if not global_vectorstore:
        raise HTTPException(status_code=400, detail="Upload PDFs first.")
        
    try:
        primary_llm = get_llm(request.provider, request.model_name, request.api_key, request.temperature)
        
        # Securely retrieve similar chunks from FAISS without relying on broken Agent logic
        docs = global_vectorstore.similarity_search(request.query, k=5)
        context = "\n---\n".join([d.page_content for d in docs])
        
        # Build strict conversation history
        langchain_history = []
        for msg in request.chat_history:
            langchain_history.append(
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            )
            
        system_instruction = f"""You are a precise Enterprise RAG AI. 
Answer the user's query utilizing ONLY the following extracted document context. If the answer is not in the context, explicitly state so.

[RETRIEVED CONTEXT]:
{context}
"""
        messages = [SystemMessage(content=system_instruction)] + langchain_history + [HumanMessage(content=request.query)]
        
        response = primary_llm.invoke(messages)
        return ChatResponse(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_csv")
async def upload_csv_endpoint(file: UploadFile = File(...)):
    temp_file_path = f"{STORAGE_DIR}/current_finance.csv"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "CSV loaded."}

@app.post("/analyze_finance", response_model=FinanceAnalysisResponse)
async def analyze_finance_endpoint(request: FinanceAnalysisRequest):
    """Analyze tabular financial data using LLM."""
    try:
        primary_llm = get_llm(request.provider, request.model_name, request.api_key, 0.4)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a data analyst. You are provided with statistical summaries of a transaction dataset (CSV) and the names of graphs generated for the user.
Please:
1. Provide a summary of the data.
2. Explain the meaning of the generated graphs.
3. Identify potential insights, anomalies, or trends from the data summary."""),
            HumanMessage(content=f"Data summary:\n\n{request.data_summary}\n\nPlease analyze.")
        ])
        
        chain = prompt | primary_llm
        response = chain.invoke({})
        return FinanceAnalysisResponse(analysis=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_csv", response_model=ChatResponse)
async def chat_csv_endpoint(request: ChatRequest):
    """Chat specifically with the uploaded CSV dataset."""
    try:
        csv_path = f"{STORAGE_DIR}/current_finance.csv"
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=400, detail="No CSV found.")
            
        primary_llm = get_llm(request.provider, request.model_name, request.api_key, request.temperature)
        df = pd.read_csv(csv_path)
        
        df_context = ""
        keywords = request.query.split(" ")
        for col in df.columns:
            if df[col].dtype == 'object':
                for word in keywords:
                    if len(word) > 3:
                        matches = df[df[col].astype(str).str.contains(word, case=False, na=False)]
                        if not matches.empty:
                            df_context += f"Matches for {word}:\n{matches.head(10).to_string()}\n"
        
        numeric_aggs = df.describe().to_string()
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a data analyst answering queries about a dataset.
Dataset stats:\n{numeric_aggs}\n\nRelevant rows:\n{df_context[:3000]}
Answer the user's question clearly and accurately."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        
        chain = prompt | primary_llm
        langchain_history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in request.chat_history]
        response = chain.invoke({"input": request.query, "chat_history": langchain_history})
        return ChatResponse(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
