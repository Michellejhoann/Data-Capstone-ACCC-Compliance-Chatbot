import json
import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "data/accc_corpus_clean.json"
VECTORDB_PATH = "./accc_vectordb"
COLLECTION_NAME = "accc_cases"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 64


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += size - overlap
    return chunks


def build_chunks(corpus):
    all_chunks = []
    for case_idx, case in enumerate(corpus):
        chunks = chunk_text(case["text"])
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"case{case_idx:04d}_chunk{chunk_idx:02d}",
                "text": chunk,
                "metadata": {
                    "title": case["title"],
                    "date": case["date"],
                    "case_year": case.get("case_year") or 0,
                    "category": case["category"],
                    "case_url": case["case_url"],
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                }
            })
    return all_chunks


def main():
    # load corpus
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} cases from {CORPUS_PATH}")

    # build chunks
    chunks = build_chunks(corpus)
    avg_words = sum(len(c["text"].split()) for c in chunks) / len(chunks)
    print(f"Built {len(chunks)} chunks, avg {avg_words:.0f} words each")

    # wipe old DB to avoid stale embeddings
    if Path(VECTORDB_PATH).exists():
        shutil.rmtree(VECTORDB_PATH)
        print(f"Removed old {VECTORDB_PATH}")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # embed in batches
    print(f"Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    n_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        batch = chunks[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metas = [c["metadata"] for c in batch]
        embs = embedder.encode(texts, show_progress_bar=False).tolist()
        collection.add(documents=texts, embeddings=embs, ids=ids, metadatas=metas)
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            print(f"  batch {b+1}/{n_batches} done")

    print(f"Done. {collection.count()} chunks indexed in {VECTORDB_PATH}")


if __name__ == "__main__":
    main()