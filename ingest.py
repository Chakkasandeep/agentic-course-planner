"""
ingest.py — Document Ingestion & FAISS Index Builder
Loads raw_catalog.txt → chunks → embeds → saves FAISS index.
"""

import os
import re
import json
import pickle
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CATALOG = os.path.join(DATA_DIR, "raw_catalog.txt")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.pkl")

# ── Chunking config ──────────────────────────────────────────────────────────
# Strategy: We use a RecursiveCharacterTextSplitter with chunk_size=500 and overlap=100.
# Explanation: 500 characters keeps chunks dense and focused on singular rules or 
# course prerequisites without diluting the semantic meaning. The 100 character overlap
# ensures that cross-sentence context (like "Prerequisites for CS 3345:\n(continued rules)")
# is not severed between chunks, reducing lost context during retrieval.

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ── Embedding model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_raw_catalog(path: str) -> List[Document]:
    """
    Parse raw_catalog.txt into LangChain Documents.
    Each document entry has: source_url, section_heading, category, content.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"raw_catalog.txt not found at {path}. "
            "Run scraper.py first: python scraper.py"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on scraper document separator line
    records = re.split(r"^={10,}\n", raw, flags=re.MULTILINE)
    documents: List[Document] = []

    for record in records:
        if not record.strip():
            continue

        source_url = ""
        section_heading = "Unknown"
        category = "unknown"
        content_lines = []
        in_content = False

        for line in record.splitlines():
            if line.startswith("SOURCE: "):
                source_url = line[len("SOURCE: "):].strip()
            elif line.startswith("SECTION: "):
                section_heading = line[len("SECTION: "):].strip()
            elif line.startswith("CATEGORY: "):
                category = line[len("CATEGORY: "):].strip()
            elif line.startswith("CONTENT:"):
                extra = line[len("CONTENT:"):].strip()
                if extra:
                    content_lines.append(extra)
                in_content = True
            elif in_content and not line.startswith("-" * 20):
                content_lines.append(line)
            elif line.startswith("-" * 20):
                in_content = False

        content = " ".join(content_lines).strip()
        if not content or not source_url:
            continue

        doc = Document(
            page_content=content,
            metadata={
                "source_url": source_url,
                "section_heading": section_heading,
                "category": category,
            },
        )
        documents.append(doc)

    print(f"Parsed {len(documents)} sections from raw_catalog.txt")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks.
    Preserves original metadata and adds chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    for doc_idx, doc in enumerate(documents):
        splits = splitter.split_text(doc.page_content)
        for chunk_idx, split_text in enumerate(splits):
            chunk_id = f"doc{doc_idx:04d}_chunk{chunk_idx:03d}"
            chunk = Document(
                page_content=split_text,
                metadata={
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                },
            )
            chunks.append(chunk)

    print(f"Created {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_faiss_index(chunks: List[Document]) -> FAISS:
    """Embed chunks and build FAISS vector store."""
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Building FAISS index for {len(chunks)} chunks ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("FAISS index built.")
    return vectorstore


def save_index(vectorstore: FAISS, chunks: List[Document]):
    """Persist FAISS index and chunks to disk."""
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved FAISS index -> {FAISS_DIR}")
    print(f"Saved chunks      -> {CHUNKS_FILE}")


def load_index() -> FAISS:
    """Load FAISS index from disk."""
    if not os.path.exists(FAISS_DIR):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_DIR}. "
            "Run ingest.py first: python ingest.py"
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"Loaded FAISS index from {FAISS_DIR}")
    return vectorstore


if __name__ == "__main__":
    print("=" * 60)
    print("Ingest Pipeline — Course Planning Assistant")
    print("=" * 60)
    docs = parse_raw_catalog(RAW_CATALOG)
    chunks = chunk_documents(docs)
    vectorstore = build_faiss_index(chunks)
    save_index(vectorstore, chunks)
    print("\nIngestion complete!")
