# OpenSCADA-Lite RAG Assistant

This repository contains a custom **Retrieval-Augmented Generation (RAG)** pipeline built for the OpenSCADA Lite project. The pipeline allows you to query your codebase, documentation, and project-related ChatGPT conversations using a project-specific LLM, enabling fast knowledge transfer and onboarding.

All data stays internal — nothing is sent externally, keeping your code and documentation safe.

---

## Features

- **Custom LLM on your code:** Train or run a model specifically for your project.
- **RAG (Retrieval-Augmented Generation):** Uses FAISS embeddings for context-aware answers.
- **Supports CPU & older GPUs:** Designed to work even on limited hardware (e.g., i7 2600 + GTX 1050 Ti).
- **Handles multiple data types:** Code files, documentation, and prior ChatGPT interactions.
- **Safe & internal:** All processing is local.

---

## Project Structure
```bash
rag_scada/
├── scripts/
│ └── rag_scada.py # Main script to build index and query
├── data/
│ ├── code/ # Project source code
│ │ └── openscada_lite/
│ ├── docs/ # Documentation, README, notes, etc.
│ └── chats/ # ChatGPT logs or development conversations
├── models/ # GGUF LLM models (CPU/GPU compatible)
├── requirements.txt # Python dependencies
└── README.md # This file
```
---

## Setup

**Clone repository**
```bash
git clone <repo-url>
cd rag_scada
```
**Install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
⚠️ Some models require llama.cpp or other GGUF-compatible runtime for GPU/CPU inference.

**Prepare your data**

Organize your project files inside data/:

```bash
data/
├── code/       # Your source code
├── docs/       # README.md, guides, documentation
└── chats/      # ChatGPT development logs
```
**Download a compatible LLM model**
```bash
CPU: Models optimized for inference in smaller memory.

GPU (older CUDA like 6.1): Check GGUF models compatible with llama.cpp.
```
Example path:
```bash
models/deepseek-coder-6.7b-instruct-q5_k_m.gguf
```
Run the RAG pipeline
```bash
python3 scripts/rag_scada.py
```
Select the LLM model path when prompted.

Choose an action:
```bash
Rebuild the index (chunking, embeddings, FAISS)

Query the model
```
How It Works

**Chunking**

Splits code, documentation, and chats into meaningful blocks.

Ensures context is preserved in queries.

**Embeddings**

Uses SentenceTransformers (multi-qa-MiniLM-L6-cos-v1) to convert chunks to vectors.

FAISS Index

Stores embeddings for fast similarity search.

**Querying**

Retrieves relevant chunks from the FAISS index.

Feeds them to the LLM for context-aware answers.

**Example Queries**
```bash
> What is the name of the project?
OpenSCADA Lite

> Can I use Docker?
Yes, Docker is supported.

```

Instructions and example code are returned, directly referencing project classes and methods.

**Hardware Notes**
```bash
CPU: i7 2600, RAM: 32GB

GPU: GTX 1050 Ti (CUDA 6.1) — not compatible with latest PyTorch CUDA builds

Works with llama.cpp GGUF models for older GPUs or CPU-only inference.
```

**References**

[Hugging Face GGUF Models](https://huggingface.co/)

[SentenceTransformers](https://www.sbert.net/)

[FAISS](https://github.com/facebookresearch/faiss)

**License**

[MIT License](https://mit-license.org/)
