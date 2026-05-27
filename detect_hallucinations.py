"""
detect_hallucinations.py

Automated hallucination detection for ACCC compliance chatbot.

Method: corpus-grounded entity verification.
Reference: QuCo-RAG (2024, arXiv:2512.19134) and HalluGraph (2025, arXiv:2512.01659).

Each generated answer is parsed for named entities (company names,
case references, legal terms). Each entity is then verified against
the source corpus. Entities that do not appear in the corpus are
flagged as potential hallucinations.
"""

import json
import re
import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

CHROMA_PATH = "./accc_vectordb"
COLLECTION_NAME = "accc_cases"
CORPUS_PATH = "data/accc_corpus_clean.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "google/flan-t5-large"
INITIAL_K = 20
FINAL_K = 4

print("Loading models...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)
tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
llm = T5ForConditionalGeneration.from_pretrained(LLM_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# load the full corpus as a single lowercase blob for substring search
print("Loading corpus...")
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)
corpus_text = " ".join(case["text"] for case in corpus).lower()
print(f"Corpus loaded: {len(corpus)} cases, {len(corpus_text):,} characters")

# 17 in-domain + 3 out-of-domain queries
TEST_QUERIES = [
    "What penalty did Webjet receive?",
    "What enforcement action did the ACCC take against Optus?",
    "Has the ACCC fined any company for greenwashing?",
    "What is misleading was/now pricing?",
    "What penalty did EnergyAustralia receive?",
    "Has the ACCC issued infringement notices for false advertising?",
    "What is a court enforceable undertaking?",
    "What kind of conduct violates Australian Consumer Law?",
    "What enforcement actions involve Telstra?",
    "What penalties has Volkswagen faced?",
    "What does the ACCC do about drip pricing?",
    "What recent cases involve sustainability claims?",
    "What infringement notices have been issued in 2024?",
    "What companies have received warning notices?",
    "How does the ACCC handle unconscionable conduct?",
    "What are recent court orders from the Federal Court?",
    "What is section 87B of the Competition and Consumer Act?",
    "What enforcement actions involve retail companies?",
    "What is RRP misleading pricing?",
    "What penalties has the ACCC imposed for false country of origin claims?",
]


def retrieve_and_rerank(query):
    query_emb = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=INITIAL_K)
    docs = results["documents"][0]
    pairs = [(query, doc) for doc in docs]
    rerank_scores = reranker.predict(pairs)
    scored = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)[:FINAL_K]
    return [s[0] for s in scored]


def generate_answer(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_length=200,
            num_beams=6,
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_entities(text):
    # extracts candidate named entities from the answer
    entities = set()

    # company-style names ending in Ltd / Pty / Inc / Limited / Group
    pat_company = re.compile(
        r"\b([A-Z][A-Za-z0-9&'\.\-]+(?:\s+[A-Z][A-Za-z0-9&'\.\-]+){0,6}\s+(?:Pty\s+)?(?:Ltd|Limited|Inc|Group|Corporation|Co\.))\b"
    )
    for m in pat_company.finditer(text):
        entities.add(m.group(1).strip())

    # case-style references like "ACCC v X"
    pat_case = re.compile(r"\b(ACCC\s+v\s+[A-Z][A-Za-z0-9&'\.\-\s]{2,40})\b")
    for m in pat_case.finditer(text):
        entities.add(m.group(1).strip())

    # generic 2+ word capitalised proper nouns
    pat_proper = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")
    for m in pat_proper.finditer(text):
        candidate = m.group(1).strip()
        skip_starts = {"The", "Australian", "Federal", "In", "On", "Date", "Section", "Topics"}
        first_word = candidate.split()[0]
        if first_word not in skip_starts:
            entities.add(candidate)

    return entities


def entity_in_corpus(entity, corpus_text):
    # case-insensitive substring match
    return entity.lower() in corpus_text


print("\nRunning hallucination detection")
print(f"Test set: {len(TEST_QUERIES)} queries\n")

results = []
total_entities = 0
total_hallucinations = 0

for i, query in enumerate(TEST_QUERIES, 1):
    print(f"[{i}/{len(TEST_QUERIES)}] {query}")
    chunks = retrieve_and_rerank(query)
    answer = generate_answer(query, chunks)
    entities = extract_entities(answer)
    hallucinations = [e for e in entities if not entity_in_corpus(e, corpus_text)]

    print(f"  entities: {len(entities)} | flagged: {len(hallucinations)}")
    if hallucinations:
        for h in hallucinations:
            print(f"    [flag] {h}")

    results.append({
        "query": query,
        "answer": answer,
        "entities_found": sorted(list(entities)),
        "hallucinations": sorted(hallucinations),
        "num_entities": len(entities),
        "num_hallucinations": len(hallucinations),
    })
    total_entities += len(entities)
    total_hallucinations += len(hallucinations)


print("\nSummary")
print("-" * 50)
print(f"Total queries: {len(TEST_QUERIES)}")
print(f"Total entities extracted: {total_entities}")
print(f"Total hallucinations flagged: {total_hallucinations}")
print(f"Hallucination rate: {100*total_hallucinations/max(total_entities,1):.1f}%")

clean_queries = sum(1 for r in results if r["num_hallucinations"] == 0)
print(f"Queries with zero hallucinations: {clean_queries}/{len(TEST_QUERIES)}")

with open("eval_results/hallucination_audit.json", "w", encoding="utf-8") as f:

    json.dump({
        "method": "corpus-grounded entity verification (QuCo-RAG 2024, HalluGraph 2025)",
        "total_queries": len(TEST_QUERIES),
        "total_entities": total_entities,
        "total_hallucinations": total_hallucinations,
        "hallucination_rate_percent": round(100*total_hallucinations/max(total_entities,1), 2),
        "queries_with_zero_hallucinations": clean_queries,
        "results": results,
    }, f, indent=2, ensure_ascii=False)

print("\nSaved to eval_results/hallucination_audit.json")
