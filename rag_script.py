import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

DATA_PATH = "/opt/ai_project_assistant/data"
INDEX_PATH = "/opt/ai_project_assistant/dataset"

CODE_PATH = DATA_PATH + "/code"
DOCS_PATH = DATA_PATH + "/docs"
CHATS_PATH = DATA_PATH + "/chats"

ALLOWED_EXT = [
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".js",
]

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def chunk_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def read_files(base):
    files = []

    for root, _, filenames in os.walk(base):
        for f in filenames:
            ext = os.path.splitext(f)[1]
            if ext in ALLOWED_EXT:
                files.append(os.path.join(root, f))

    return files


def collect_data():
    files = []
    files += read_files(CODE_PATH)
    files += read_files(DOCS_PATH)
    files += read_files(CHATS_PATH)

    return files


def build_index():
    print("Loading embedding model...")
    embedder = SentenceTransformer(
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )

    files = collect_data()

    if not files:
        raise RuntimeError("No files found in data folder")

    texts = []
    meta = []

    print("Chunking project...")

    for file in files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            chunks = chunk_text(content)

            for c in chunks:
                texts.append(c)
                meta.append({"file": file, "text": c})

        except:
            pass

    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)

    faiss.write_index(index, INDEX_PATH + "/index.faiss")

    with open(INDEX_PATH + "/meta.json", "w") as f:
        json.dump(meta, f)

    print("Index built:", len(texts), "chunks")


def load_index():
    index = faiss.read_index(INDEX_PATH + "/index.faiss")

    with open(INDEX_PATH + "/meta.json") as f:
        meta = json.load(f)

    return index, meta


def retrieve(question, embedder, index, meta, k=6):
    q = embedder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q)

    scores, ids = index.search(q, k)

    contexts = []
    for i in ids[0]:
        contexts.append(meta[i]["text"])

    return "\n\n".join(contexts)


def ask(llm, embedder, index, meta, question):
    context = retrieve(question, embedder, index, meta)

    prompt = f"""
You are an assistant for a SCADA Python project.

Use the context below to answer accurately.

Context:
{context}

Question:
{question}

Answer:
"""

    out = llm(
        prompt,
        max_tokens=300,
        temperature=0.2,
    )

    return out["choices"][0]["text"]


def main():
    model_path = input(
        "Path to GGUF model (example: /models/deepseek.gguf): "
    )

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=20,
        n_ctx=4096
    )

    embedder = SentenceTransformer(
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        device="cpu"
    )

    print("1 = rebuild index")
    print("2 = query")

    mode = input("> ")

    if mode == "1":
        build_index()
        return

    index, meta = load_index()

    while True:
        q = input("\nQuestion: ")
        print("\nAnswer:\n", ask(llm, embedder, index, meta, q))


if __name__ == "__main__":
    main()
    
