import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# config
CHROMA_PATH = "./accc_vectordb"
COLLECTION_NAME = "accc_cases"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "google/flan-t5-large"

# retrieve 20 first, then rerank to keep top 4
INITIAL_K = 20
FINAL_K = 4


@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
    llm = T5ForConditionalGeneration.from_pretrained(LLM_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    return embedder, reranker, tokenizer, llm, collection


def retrieve_and_rerank(query, embedder, reranker, collection):
    # first stage: fast retrieval with embeddings
    query_emb = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=INITIAL_K)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # second stage: rerank the 20 with cross-encoder
    pairs = [(query, doc) for doc in docs]
    rerank_scores = reranker.predict(pairs)
    scored = sorted(
        zip(docs, metas, dists, rerank_scores),
        key=lambda x: x[3],
        reverse=True,
    )[:FINAL_K]

    final_docs = [s[0] for s in scored]
    final_metas = [s[1] for s in scored]
    final_sims = [1 - s[2] for s in scored]
    final_scores = [float(s[3]) for s in scored]
    return final_docs, final_metas, final_sims, final_scores


def generate_answer(query, chunks, tokenizer, llm):
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


# UI
st.set_page_config(page_title="ACCC Compliance Chatbot", page_icon="⚖️", layout="wide")

st.title("⚖️ ACCC Compliance Chatbot")
st.markdown(
    "RAG assistant over 1,017 ACCC enforcement cases (2010-2026). "
    "Ask about misleading conduct, pricing deception, advertising claims, court penalties, and undertakings."
)

with st.spinner("Loading models and vector data (first run can take 2-3 min)..."):
    embedder, reranker, tokenizer, llm, collection = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Retrieved sources ({len(msg['sources'])} cases)"):
                for i, src in enumerate(msg["sources"], 1):
                    doc, meta, sim, score = src
                    st.markdown(
                        f"**{i}. {meta.get('title', 'Untitled')}**  \n"
                        f"*{meta.get('category', '')} - {meta.get('date', '')} - "
                        f"similarity: {sim:.2f} - rerank score: {score:.2f}*"
                    )
                    if meta.get("url"):
                        st.markdown(f"[View on accc.gov.au]({meta['url']})")
                    st.text(doc[:400] + ("..." if len(doc) > 400 else ""))
                    st.divider()

# new query
if prompt := st.chat_input("Ask about ACCC enforcement cases..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + re-ranking..."):
            docs, metas, sims, scores = retrieve_and_rerank(prompt, embedder, reranker, collection)
        with st.spinner("Generating answer..."):
            answer = generate_answer(prompt, docs, tokenizer, llm)

        st.markdown(answer)
        retrieved = list(zip(docs, metas, sims, scores))
        with st.expander(f"Retrieved sources ({len(retrieved)} cases)"):
            for i, src in enumerate(retrieved, 1):
                doc, meta, sim, score = src
                st.markdown(
                    f"**{i}. {meta.get('title', 'Untitled')}**  \n"
                    f"*{meta.get('category', '')} - {meta.get('date', '')} - "
                    f"similarity: {sim:.2f} - rerank score: {score:.2f}*"
                )
                if meta.get("url"):
                    st.markdown(f"[View on accc.gov.au]({meta['url']})")
                st.text(doc[:400] + ("..." if len(doc) > 400 else ""))
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved,
    })