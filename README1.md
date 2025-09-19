# 🔬 AI Research Assistant RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that combines document analysis with real-time web search capabilities, enabling intelligent research assistance through conversational AI.

## 🚀 Live Demo

- **Streamlit App**: [Research Assistant RAG](https://research-assistant-rag.streamlit.app/)

## 📋 Table of Contents

- [Overview](#overview)
- [RAG Pipeline Implementation](#rag-pipeline-implementation)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This AI Research Assistant leverages advanced RAG techniques to provide intelligent document analysis and research capabilities. Users can upload documents (PDF, DOCX, PPTX) and interact with them through natural language queries, while the system also provides real-time information from web sources.

### Key Capabilities

- 📄 Multi-format Document Processing (PDF, DOCX, PPTX)
- 🔎 Hybrid Information Retrieval (Documents + Web Search)
- 💬 Conversational Interface with context awareness
- 👥 Session-based Multi-tenancy (isolated data per user)
- ☁️ Production-ready Cloud Deployment

---

## 🔄 RAG Pipeline Implementation

### Phase 1: Document Ingestion & Preprocessing
```python
# Document Loading Chain
PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader

# Text Chunking
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Embeddings
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
