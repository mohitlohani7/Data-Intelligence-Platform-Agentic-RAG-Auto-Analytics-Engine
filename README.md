# 🏢 Agentic RAG & Financial Data Platform

> **A dual-architecture AI application with an Agentic RAG pipeline and a Tabular Data Analytics dashboard.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)

---

## 🌟 Overview
This project is a decoupled microservice (FastAPI + Streamlit) that handles both unstructured document retrieval (RAG) and structured numerical analysis. It features automatic model routing across OpenAI, Claude, and Groq, semantic caching, and a pipeline for fine-tuning open source models.

## 🚀 Key Features

### 1. 🤖 Tool-Calling Agentic RAG
* Rather than a standard retrieval chain, this uses an autonomous agent capable of deciding when to search the vector index and how to compile the answer.

### 2. 🔀 Multi-API Routing & Fallbacks
* Supports **OpenAI**, **Anthropic (Claude)**, and **Groq** APIs dynamically. Includes fallback logic to ensure queries map via alternate models if a specific endpoint crashes.

### 3. ⚡ Local Semantic Caching
* LangChain `SQLiteCache` intercepts redundant queries, limiting API calls to save costs and reduce latency.

### 4. 📚 Embeddings & Chunking 
* Uses `BAAI/bge-large-en-v1.5` embeddings via FAISS. Features `RecursiveCharacterTextSplitter` logic for optimal paragraph mapping.

### 5. 💰 CSV Data Analytics & Plotting
* Generates automated data visualizations (Matplotlib/Seaborn) from uploaded CSVs.
* Generates downloadable PDF reports containing the charts and an LLM analysis of the tabular data.

### 6. 🎓 QLoRA Fine-Tuning Pipeline
* Included in `llm_finetuning_pipeline/` are utilities to generate custom QA pairs from PDFs using Teacher models, and fine-tune a LLaMA-3 model using PEFT/LoRA.

---

## 🏗️ Architecture
* **`backend/main.py` (FastAPI):** Handles embeddings, FAISS storage, validation, LLM routing, and endpoints.
* **`frontend/app.py` (Streamlit):** User interface with custom CSS rendering, interacting with the backend via REST API.

---

## 🚀 How to Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Start the application (Double click the batch file):
```bash
run_enterprise_platform.bat
```
*(This will host the FastAPI backend securely in the background and launch Streamlit).*
