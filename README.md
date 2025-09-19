# 📚 Research Assistant (FastAPI + Streamlit + LangChain + Gemini + Qdrant)

An **AI-powered research assistant** built with **FastAPI (backend)** and **Streamlit (frontend)**.  
Upload your documents, chat with them, and get contextual answers powered by **Google Gemini** + **LangChain** + **Qdrant**.

## 🚀 Features
- 📄 Upload and process documents (`.pdf`, `.docx`, `.pptx`)  
- 🔍 Semantic search with **Qdrant Vector DB**  
- 🤖 Conversational AI with **LangChain + Google Gemini**  
- ⚡ REST API with FastAPI & interactive UI with Streamlit  
- ☁️ Deployable on **Render / Railway / Streamlit Cloud**  


## 📂 Project Structure
research-assistant-rag-2/
│── backend/
│   ├── main.py              # FastAPI backend application
│   ├── requirements.txt     # Backend dependencies
│
│── frontend/
│   ├── app.py               # Streamlit frontend application
│   ├── requirements.txt     # Frontend dependencies
│
│── app.py                   # (root app.py, optional/legacy)
│── ingest.py                # Script for ingesting documents
│── requirements.txt         # Combined/global dependencies
│── .gitignore

Installation
Clone the repository
git clone https://github.com/RhithanyaParthasarathi/research-assistant-rag-2?authuser=0
cd research-assistant-rag-2

Setup Backend (FastAPI)
cd backend
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt


Run backend:

uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Setup Frontend (Streamlit)

In a new terminal:

cd frontend
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

Run frontend:

streamlit run app.py
