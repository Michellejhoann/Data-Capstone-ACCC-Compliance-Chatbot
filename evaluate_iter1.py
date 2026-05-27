import json
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

CHROMA_PATH = "./accc_vectordb"
COLLECTION_NAME = "accc_cases"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
TOP_K = 4

print("Loading models...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
llm = T5ForConditionalGeneration.from_pretrained(LLM_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print("Models loaded.\n")

# 7 in-domain + 3 out-of-domain
TEST_QUESTIONS = [
    {"question": "What enforcement action did the ACCC take against Webjet?", "expected_answer": "The ACCC took Federal Court action against Webjet for making misleading pricing representations to consumers. The Court ordered Webjet to pay pecuniary penalties for breaching the Australian Consumer Law by failing to disclose the full price of bookings, including compulsory fees and charges.", "expected_relevance": "high"},
    {"question": "Has the ACCC fined any company for greenwashing?", "expected_answer": "Yes, the ACCC has taken multiple enforcement actions against companies for greenwashing and misleading environmental claims. These actions target false or unsubstantiated representations about sustainability, recyclability, and green credentials, with the ACCC issuing internet sweeps and pursuing penalties under the Australian Consumer Law.", "expected_relevance": "high"},
    {"question": "What is misleading was/now pricing?", "expected_answer": "Misleading was/now pricing occurs when a business advertises a comparison between a previous price and a current price, but the previous price was not genuinely the regular selling price. This practice creates a false impression of savings or discount, breaching the Australian Consumer Law and resulting in ACCC enforcement action and penalties.", "expected_relevance": "high"},
    {"question": "What penalty did EnergyAustralia receive?", "expected_answer": "EnergyAustralia was ordered by the Federal Court to pay pecuniary penalties for misleading consumers about energy plans, discount claims, and pricing representations. The ACCC took action over conduct that breached the Australian Consumer Law, including misleading discount-off-what representations in energy promotions.", "expected_relevance": "high"},
    {"question": "Has the ACCC issued infringement notices for false advertising?", "expected_answer": "Yes, the ACCC regularly issues infringement notices to businesses for false or misleading advertising claims under the Australian Consumer Law. These notices apply to a wide range of conduct including misleading product claims, false representations about country of origin, and unsubstantiated health or environmental claims.", "expected_relevance": "medium"},
    {"question": "What is a court enforceable undertaking?", "expected_answer": "A court enforceable undertaking under section 87B of the Competition and Consumer Act is a written agreement where a company commits to corrective actions, compliance programs, and consumer remediation to resolve ACCC concerns without proceeding to a contested trial. The undertaking is registered on a public register and enforceable by the courts.", "expected_relevance": "medium"},
    {"question": "What kind of conduct violates Australian Consumer Law?", "expected_answer": "Conduct that misleads or deceives consumers, makes false or misleading representations about goods or services, engages in unconscionable behavior, breaches consumer guarantees, or uses bait advertising violates the Australian Consumer Law. The ACCC enforces these provisions through court action, infringement notices, and enforceable undertakings.", "expected_relevance": "medium"},
    {"question": "What is the best pizza recipe in Italy?", "expected_answer": "OUT_OF_DOMAIN", "expected_relevance": "out_of_domain"},
    {"question": "How do I train a dog to sit?", "expected_answer": "OUT_OF_DOMAIN", "expected_relevance": "out_of_domain"},
    {"question": "What is the capital of France?", "expected_answer": "OUT_OF_DOMAIN", "expected_relevance": "out_of_domain"},
]


def retrieve(query):
    query_emb = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=TOP_K)
    docs = results["documents"][0]
    dists = results["distances"][0]
    return docs, [1 - d for d in dists]


def generate_answer(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def calculate_bleu(reference, candidate):
    if not candidate or not reference:
        return 0.0
    ref_tokens = [reference.lower().split()]
    cand_tokens = candidate.lower().split()
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoothie)


def calculate_rouge_l(reference, candidate):
    if not candidate or not reference:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure


def calculate_faithfulness(answer, chunks):
    if not answer.strip():
        return 0.0
    answer_emb = embedder.encode([answer])
    chunks_emb = embedder.encode(chunks)
    similarities = cosine_similarity(answer_emb, chunks_emb)[0]
    return float(np.max(similarities))


print("Iteration 1 evaluation (flan-t5-base, no re-ranking)")
print(f"Test set: {len(TEST_QUESTIONS)} questions\n")

results = []
for i, item in enumerate(TEST_QUESTIONS, 1):
    print(f"[{i}/{len(TEST_QUESTIONS)}] {item['question']}")
    chunks, similarities = retrieve(item['question'])
    avg_retrieval_sim = np.mean(similarities)
    answer = generate_answer(item['question'], chunks)

    if item['expected_answer'] != "OUT_OF_DOMAIN":
        bleu = calculate_bleu(item['expected_answer'], answer)
        rouge_l = calculate_rouge_l(item['expected_answer'], answer)
    else:
        bleu, rouge_l = None, None

    faithfulness = calculate_faithfulness(answer, chunks)

    results.append({
        "question": item['question'],
        "expected_relevance": item['expected_relevance'],
        "answer": answer,
        "avg_retrieval_similarity": round(avg_retrieval_sim, 3),
        "bleu": round(bleu, 3) if bleu is not None else "N/A",
        "rouge_l": round(rouge_l, 3) if rouge_l is not None else "N/A",
        "faithfulness": round(faithfulness, 3),
    })
    print(f"   retrieval={avg_retrieval_sim:.3f} faithfulness={faithfulness:.3f}\n")


print("\nSummary metrics (Iteration 1: flan-t5-base, no re-ranking)")
print("-" * 50)

in_domain = [r for r in results if r['expected_relevance'] != 'out_of_domain']
ood = [r for r in results if r['expected_relevance'] == 'out_of_domain']

if in_domain:
    avg_bleu = np.mean([r['bleu'] for r in in_domain if r['bleu'] != "N/A"])
    avg_rouge = np.mean([r['rouge_l'] for r in in_domain if r['rouge_l'] != "N/A"])
    avg_retrieval = np.mean([r['avg_retrieval_similarity'] for r in in_domain])
    avg_faithful = np.mean([r['faithfulness'] for r in in_domain])
    print(f"\nIn-domain (n={len(in_domain)}):")
    print(f"  BLEU:                 {avg_bleu:.3f}")
    print(f"  ROUGE-L:              {avg_rouge:.3f}")
    print(f"  Retrieval similarity: {avg_retrieval:.3f}")
    print(f"  Faithfulness:         {avg_faithful:.3f}")

if ood:
    avg_retrieval_ood = np.mean([r['avg_retrieval_similarity'] for r in ood])
    print(f"\nOut-of-domain (n={len(ood)}):")
    print(f"  Retrieval similarity: {avg_retrieval_ood:.3f}")

if in_domain and ood:
    print(f"\nDomain gap: {avg_retrieval - avg_retrieval_ood:.3f}")

with open("eval_results/iteration_1_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nSaved to eval_results/iteration_1_results.json")
