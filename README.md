# AI Research Assistant RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that combines document analysis with real-time web search capabilities, enabling intelligent research assistance through conversational AI.

## Live Demo

- **Streamlit App**: https://research-assistant-rag.streamlit.app/

## Table of Contents

- [Overview](#overview)
- [RAG Pipeline Implementation](#rag-pipeline-implementation)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

## Overview

This AI Research Assistant leverages advanced RAG techniques to provide intelligent document analysis and research capabilities. Users can upload documents (PDF, DOCX, PPTX) and interact with them through natural language queries, while the system also provides real-time information from web sources.

### Key Capabilities

- **Multi-format Document Processing**: Supports PDF, Word documents, and PowerPoint presentations
- **Hybrid Information Retrieval**: Combines document content with real-time web search
- **Conversational Interface**: Natural language interaction with context awareness
- **Session-based Multi-tenancy**: Multiple users with isolated document stores
- **Production Deployment**: Scalable cloud infrastructure

## RAG Pipeline Implementation

### Phase 1: Document Ingestion & Preprocessing

```python
# Document Loading Chain
PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader

# Text Chunking Strategy
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Embedding Generation
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384-dimensional vectors
```

**Workflow**: File Upload → Temporary Storage → Document Loading → Text Chunking → Embedding Generation → Vector Storage

### Phase 2: Vector Storage & Indexing

```python
# Vector Database with Multi-tenancy
QdrantVectorStore with tenant-based filtering
search_kwargs={'filter': models.Filter(must=[
    models.FieldCondition(key="metadata.tenant", match=models.MatchValue(value=session_id))
])}
```

**Features**: Tenant Isolation, Payload Indexing, Session-specific Retrievers

### Phase 3: Query Processing & Retrieval

```python
# ReAct Agent Architecture
create_react_agent(llm, tools, prompt)
Tools: [Document Search Tool, Web Search Tool (DuckDuckGo)]
```

**Process**: Query → Agent Executor → Tool Selection → Parallel Execution (Document + Web Search) → Context Compilation

### Phase 4: Response Generation

```python
# LLM Configuration
ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
```

**Workflow**: Context + Chat History → Gemini Processing → Response Validation → Streaming to UI

##  Key Technical Achievements

### Advanced RAG Techniques

- **Contextual Chunking**: Overlap strategy maintains semantic continuity across document segments
- **Metadata Enhancement**: Rich document metadata with session-based tenant filtering
- **Hybrid Retrieval**: Intelligent combination of document content and real-time web information
- **Agent-based Orchestration**: ReAct pattern for autonomous tool selection and execution

### Production-Ready Architecture

- **Multi-Session Management**: UUID-based session isolation with dedicated agent instances
- **Concurrent Processing**: FastAPI async framework enables multiple simultaneous users
- **Model Optimization**: Pre-loaded LLM and embeddings at server startup for performance
- **Cloud Infrastructure**: Scalable deployment on Render.com with Qdrant Cloud vector storage

### Performance & Reliability

- **Memory Efficiency**: Shared model instances across sessions with optimized embedding generation
- **Error Recovery**: Comprehensive exception handling with graceful degradation
- **Data Security**: Session-based data isolation, temporary file cleanup, secure API key management
- **Scalability**: Horizontal scaling support with stateless backend architecture

##  Features

###  Advanced RAG Pipeline

- **Semantic Search**: Vector-based document retrieval using state-of-the-art embeddings
- **Intelligent Chunking**: Context-preserving text segmentation with overlap
- **Multi-source Integration**: Seamless combination of documents and web information
- **Agent-based Architecture**: ReAct pattern for intelligent tool selection

###  Document Processing

- **PDF Analysis**: Extract and process research papers, reports, books
- **Word Document Support**: Handle articles, documentation, notes
- **PowerPoint Processing**: Analyze presentation slides and educational materials
- **Batch Upload**: Process multiple documents simultaneously

### Web Integration

- **Real-time Search**: Live information retrieval from web sources
- **DuckDuckGo Integration**: Privacy-focused web search capabilities
- **Context Fusion**: Intelligent merging of document and web information

### Production Features

- **Cloud Deployment**: Hosted on reliable cloud infrastructure
- **Auto-scaling**: Handles variable user loads automatically
- **Session Management**: UUID-based user sessions with data isolation
- **Error Handling**: Comprehensive error recovery and user feedback

## Architecture

```
┌─────────────────┐    HTTP/REST    ┌─────────────────────┐    Vector API    ┌─────────────────┐
│   Streamlit     │ ◄──────────────► │     FastAPI         │ ◄──────────────► │   Qdrant Cloud  │
│   Frontend      │                  │     Backend         │                  │   Vector Store  │
│                 │                  │                     │                  │                 │
│ • File Upload   │                  │ • RAG Pipeline      │                  │ • Embeddings    │
│ • Chat UI       │                  │ • Agent Executor    │                  │ • Multi-tenant  │
│ • Session Mgmt  │                  │ • Document Process  │                  │ • Similarity    │
└─────────────────┘                  └─────────────────────┘                  └─────────────────┘
                                              │
                                              ▼
                                     ┌─────────────────────┐
                                     │ External Services   │
                                     │                     │
                                     │ • Google Gemini     │
                                     │ • HuggingFace       │
                                     │ • DuckDuckGo        │
                                     └─────────────────────┘
```

## Technology Stack

### Core Frameworks & Runtime

- **[LangChain](https://langchain.com/)**: RAG pipeline orchestration and agent framework
- **[FastAPI](https://fastapi.tiangolo.com/)**: High-performance async web framework with Uvicorn ASGI server
- **[Streamlit](https://streamlit.io/)**: Interactive web application frontend
- **Python 3.13**: Core runtime environment

### AI/ML Components

- **[Google Gemini 1.5 Flash](https://ai.google.dev/)**: Large language model (temperature=0 for deterministic responses)
- **[HuggingFace all-MiniLM-L6-v2](https://huggingface.co/)**: Embedding model producing 384-dimensional dense vectors
- **[Sentence Transformers](https://www.sbert.net/)**: Optimized embedding generation library

### Data Storage & Retrieval

- **[Qdrant Cloud](https://qdrant.tech/)**: Managed vector database with auto-scaling
- **Vector Similarity Search**: Cosine similarity for semantic document retrieval
- **Metadata Filtering**: Tenant-based data isolation using session IDs

### Document Processing Pipeline

- **PyPDF**: PDF text extraction
- **python-docx**: Word document processing
- **python-pptx**: PowerPoint content extraction
- **Unstructured**: Advanced document parsing for complex layouts
- **RecursiveCharacterTextSplitter**: Smart chunking (1000 chars with 200 char overlap)

### Agent Architecture

- **ReAct Pattern**: Reasoning and Acting agent framework
- **Tool Integration**: Document search + Web search capabilities
- **LangChain Agents**: Intelligent tool selection and orchestration
- **HuggingFace Hub**: Pre-trained prompt templates (react-chat)

### External Services & APIs

- **DuckDuckGo Search API**: Privacy-focused web search integration
- **Google AI Studio**: Gemini API for text generation
- **Render.com**: Backend deployment platform with auto-scaling
- **Environment Management**: Secure API key handling with python-dotenv

### Core Dependencies

```python
langchain                    # RAG orchestration
langchain-google-genai       # Gemini integration
langchain-qdrant            # Vector store integration
qdrant-client               # Vector database client
sentence-transformers       # Embedding optimization
streamlit                   # Frontend framework
fastapi                     # Backend API framework
uvicorn[standard]           # ASGI server
python-multipart            # File upload handling
```

## Installation

### Prerequisites

- Python 3.11+
- Git

### Local Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/RhithanyaParthasarathi/research-assistant-rag-2.git
   cd research-assistant-rag-2
   ```

2. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Install backend dependencies**

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**
   ```bash
   cd ../frontend
   pip install -r requirements.txt
   ```

### Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_google_gemini_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=your_collection_name
```

##  Usage

### Local Development

1. **Start the backend server**

   ```bash
   cd backend
   python main.py
   ```

2. **Start the frontend application**

   ```bash
   cd frontend
   streamlit run app.py
   ```

3. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8080
   - API Documentation: http://localhost:8080/docs

### Using the Application

1. **Upload Documents**: Drag and drop PDF, DOCX, or PPTX files
2. **Wait for Processing**: System will extract and vectorize document content
3. **Ask Questions**: Use natural language to query your documents
4. **Get Intelligent Responses**: Receive answers combining document content and web information

### Example Queries

- "Summarize the main findings from the uploaded research paper"
- "What are the current trends in AI mentioned in my documents?"
- "Compare the methodology in my document with recent web sources"

## API Documentation

### Endpoints

#### Document Ingestion

```http
POST /ingest
Content-Type: multipart/form-data

{
  "session_id": "uuid-string",
  "files": [file1, file2, ...]
}
```

#### Chat Interface

```http
POST /chat
Content-Type: application/json

{
  "session_id": "uuid-string",
  "message": "user question",
  "chat_history": [
    {"role": "user", "content": "previous message"},
    {"role": "assistant", "content": "previous response"}
  ]
}
```

### Response Formats

```json
{
  "success": true,
  "message": "2 files processed successfully."
}
```

```json
{
  "response": "Based on your documents and current web information..."
}
```

## Project Structure

```
research-assistant-rag-2/
├── backend/                    # FastAPI backend
│   ├── main.py                # Main application entry
│   ├── requirements.txt       # Backend dependencies
│   └── __pycache__/           # Python cache files
├── frontend/                   # Streamlit frontend
│   ├── app.py                 # Main frontend application
│   └── requirements.txt       # Frontend dependencies
├── .env                       # Environment variables
├── .gitignore                 # Git ignore rules
├── app.py                     # Root application file
├── ingest.py                  # Document ingestion utilities
└──  requirements.txt           # Root dependencies
```
### Key Implementation Details

- **RAG Pipeline**: Document processing, embedding generation, vector storage, and retrieval
- **Agent Architecture**: ReAct pattern with tool selection and orchestration
- **Multi-tenancy**: Session-based data isolation and user management
- **Production Deployment**: Cloud infrastructure and scalability considerations

## Acknowledgments

- **LangChain** for the comprehensive RAG framework
- **Google AI** for Gemini model access
- **Qdrant** for vector database capabilities
- **HuggingFace** for embedding models
- **Streamlit** for the interactive frontend framework
