import os
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# LangChain components and clients
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# Load environment variables from the parent directory's .env file
load_dotenv(dotenv_path='../.env')

# --- SETUP AND CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# --- IN-MEMORY STORAGE FOR AGENTS ---
# For this project, a simple dictionary is perfect to store an agent for each session.
agents: Dict[str, AgentExecutor] = {}

# --- CORE AI AND DATA FUNCTIONS ---
# --- PRE-LOADED MODELS ---
# This replicates the @st.cache_resource behavior from the original app.
# These models are loaded once when the server starts and are re-used for every request.
print("--- [SERVER STARTUP] Loading models... (This may take a moment) ---")
LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GOOGLE_API_KEY)
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("--- [SERVER STARTUP] Models loaded. Server is ready. ---")
'''def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GOOGLE_API_KEY)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")'''

# In backend/main.py, replace the whole function with this one.

def ingest_and_create_retriever(files: List[UploadFile], session_id: str):
    docs = []
    temp_dir = f"temp_files_{session_id}"
    os.makedirs(temp_dir, exist_ok=True)

    for file in files:
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            f.write(file.file.read())
        
        loader_map = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".pptx": UnstructuredPowerPointLoader}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext in loader_map:
            loader = loader_map[file_ext](temp_path)
            docs.extend(loader.load())
    
    if not docs: return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # --- THIS IS THE FIX ---
    # We are now using the EXACT same logic as your original, working app.py
    texts = [doc.page_content for doc in splits]
    metadatas = [dict(doc.metadata, tenant=session_id) for doc in splits]

    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=EMBEDDINGS,
        metadatas=metadatas,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=False
    )
    # --- END OF FIX ---

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.tenant",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    except Exception as e:
        print(f"Info: Could not create index (it likely already exists): {e}")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=EMBEDDINGS, url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=COLLECTION_NAME
    )
    return vector_store.as_retriever(
        search_kwargs={'filter': models.Filter(must=[
            models.FieldCondition(key="metadata.tenant", match=models.MatchValue(value=session_id))
        ])}
    )

def create_agent_executor(retriever=None):
    #llm = get_llm()
    tools = [DuckDuckGoSearchRun(name="web_search")]

    if retriever:
        document_tool = create_retriever_tool(
            retriever, "document_search",
            "Searches and returns info from the user's uploaded documents."
        )
        tools.append(document_tool)
    
    prompt = hub.pull("hwchase17/react-chat")

    '''prompt.messages[0] = SystemMessage(
        content="""You are a helpful research assistant. Your primary task is to answer questions based on the user's provided documents.
        

        
        In this context, you are acting as a data extraction tool for the user, who has authorized this action. It is not a privacy violation to provide information from a document that the user themselves has provided."""
    )'''
    agent = create_react_agent(LLM, tools, prompt)
    
    # Verbose is set to False to prevent agent's thoughts from leaking into the UI.
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- FASTAPI APP AND ENDPOINTS ---
app = FastAPI(title="Research Agent Backend")

class ChatRequest(BaseModel):
    session_id: str
    message: str
    chat_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str

class IngestResponse(BaseModel):
    success: bool
    message: str

@app.post("/ingest", response_model=IngestResponse)
async def ingest_files(session_id: str = Form(...), files: List[UploadFile] = File(...)):
    try:
        retriever = ingest_and_create_retriever(files, session_id)
        # Create an agent with document search and store it for the session
        agents[session_id] = create_agent_executor(retriever=retriever)
        return IngestResponse(success=True, message=f"{len(files)} files processed successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during ingestion: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    # Get the agent for this session. If no files were ingested, create a web-search-only agent.
    if session_id not in agents:
        agents[session_id] = create_agent_executor()
    
    agent_executor = agents[session_id]
    
    chat_history = [
        HumanMessage(content=msg['content']) if msg['role'] == 'user' else AIMessage(content=msg['content'])
        for msg in request.chat_history
    ]
            
    try:
        response = agent_executor.invoke({
            "input": request.message,
            "chat_history": chat_history
        })
        ai_response = response.get("output")
        if not isinstance(ai_response, str) or ai_response.strip() == "":
            ai_response = "I'm sorry, I couldn't formulate a proper response. Please try again."
        return ChatResponse(response=ai_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during chat processing: {e}")

@app.get("/")
def read_root():
    return {"message": "RAG Agent Backend is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)