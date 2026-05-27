# ACCC Compliance Chatbot

RAG chatbot over 1,017 enforcement cases from the Australian Competition and Consumer Commission (ACCC), scraped from accc.gov.au and covering 2010 to 2026.

You ask it something like "what penalty did Webjet receive" and it pulls up the actual cases from a vector database, re-ranks them, and generates a short answer that points back to the source releases.

## How it works

1. User types a question
2. The query gets embedded with MiniLM (all-MiniLM-L6-v2)
3. ChromaDB returns the top 20 most similar case chunks
4. A cross-encoder re-ranks those 20 and keeps the best 4
5. flan-t5-large generates the answer using those 4 chunks as context
6. Streamlit shows the answer plus the cited cases

The two-stage retrieval is the main trick. The first stage is fast but not super accurate, so the second stage uses a smarter model on a small set.

## Stack

- ChromaDB for the vector store (persistent, local)
- sentence-transformers/all-MiniLM-L6-v2 for embeddings
- cross-encoder/ms-marco-MiniLM-L-6-v2 for re-ranking
- google/flan-t5-large for generation (780M params, runs on CPU)
- Streamlit for the UI
- 1,017 ACCC media releases scraped with requests + BeautifulSoup

## Setup

I built this on Python 3.13 / Windows but it should run on Linux and Mac too.

```
pip install -r requirements.txt
```

The first run downloads about 4 GB of model weights from HuggingFace. After that everything is cached.

## Usage

If you want to rebuild the dataset from scratch:

```
python scraper.py
python clean_corpus.py
```

Otherwise just build the index from the cleaned corpus (one-time, takes 5 to 10 minutes):

```
python build_vectordb.py
```

This creates an `accc_vectordb/` folder.

Then run the app:

```
streamlit run app.py
```

It opens in the browser at localhost:8501.

## Data

The repo includes both the raw scraped data and the cleaned version, so you can inspect the pipeline:

- `data/accc_raw.json` is the direct output from `scraper.py`. Contains everything pulled from accc.gov.au including some duplicates and noisy text.
- `data/accc_corpus_clean.json` is the same data after running `clean_corpus.py`. This is what feeds into the vector database.

Cleaning removes PDF markers, fixes smart quotes, drops orphan headers from the HTML scrape, re-categorizes cases based on the URL pattern, and deduplicates identical content.

## Evaluation

```
python evaluate_iter1.py
python evaluate_iter2.py
python detect_hallucinations.py
```

Each script runs the test set, prints the metrics summary, and writes the detailed results to `eval_results/`.

### Iterations

The system went through two iterations:

- **Iteration 1** (`evaluate_iter1.py`): flan-t5-base, no re-ranking. Baseline.
- **Iteration 2** (`evaluate_iter2.py`): flan-t5-large + cross-encoder re-ranking + BERTScore. Final version.

### Results (Iteration 2)

- BERTScore F1: 0.833
- Faithfulness (cosine between answer and retrieved chunks): 0.432
- Retrieval similarity in-domain: 0.547
- Retrieval similarity out-of-domain: 0.147
- Domain gap: 0.400

Both iterations also report BLEU and ROUGE-L, but they stayed below 0.12 in every run. The issue is that flan-t5-large paraphrases instead of copying, and n-gram metrics don't like that. BERTScore (Zhang et al. 2020) uses contextual embeddings so it picks up on semantic similarity instead of exact word matches, which is what you actually want for a generative model.

### Hallucination audit

`detect_hallucinations.py` runs an additional 20-query audit using corpus-grounded entity verification, the approach described in QuCo-RAG (2024) and HalluGraph (2025). Each answer is parsed for named entities, and each entity is checked against the full corpus. Results: 19/20 queries clean, 1 entity flagged ("Informal Merger Act"), giving a 5.9% entity-level hallucination rate. For comparison, Magesh et al. (2025) measured 17 to 33% on commercial legal AI tools and 43% on general-purpose GPT-4.

## Project structure

```
.
├── app.py                       streamlit chatbot
├── scraper.py                   scrapes accc.gov.au media releases
├── clean_corpus.py              cleans the scraped data
├── build_vectordb.py            builds the chromadb index
├── evaluate_iter1.py            iteration 1 baseline (flan-t5-base)
├── evaluate_iter2.py            iteration 2 final (flan-t5-large + re-rank + BERTScore)
├── detect_hallucinations.py     hallucination audit
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── accc_raw.json                       raw scraper output
│   └── accc_corpus_clean.json              cleaned corpus
└── eval_results/
    ├── iteration_1_results.json
    ├── iteration_2_results.json
    └── hallucination_audit.json
```

## Known limitations

- Running flan-t5-large on CPU is slow. About 30 to 60 seconds per query on my laptop. A small GPU or a smaller model would help.
- Source is limited to ACCC media releases. Court judgments and full case documents aren't in there.
- The re-ranker is a generic web re-ranker, not legal-specific. A legal-bert variant might do better.
- Answers tend to be short. flan-t5-large is fine but not great at long generation. A 7B model like mistral or llama-3 would give much longer answers.

## License

MIT
