# VeriRAG

A hybrid Retrieval-Augmented Generation system for PDF question answering with cross-encoder reranking, confidence gating, and automatic web fallback.

Built with Python, Streamlit, FAISS, Sentence Transformers, Groq API, and LLaMA 3.1.

---

## Why I Built This

Most PDF chat systems rely entirely on semantic vector search and often return irrelevant context or hallucinated answers when retrieval quality is poor. VeriRAG was built to improve reliability by combining keyword retrieval (BM25), semantic retrieval (FAISS), cross-encoder reranking, confidence-based gating, and automatic web fallback when document context is insufficient.

---

## What it does

Upload PDFs, ask questions, get cited answers. If the documents don't have the answer, it falls back to web search automatically.

## Tech Stack

- **Backend:** Python
- **UI:** Streamlit
- **Retrieval:** BM25, FAISS, Reciprocal Rank Fusion
- **Reranking:** Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- **LLM:** LLaMA 3.1 via Groq API
- **Deployment:** Docker, Streamlit Cloud

What makes it different from a basic RAG:

- **Hybrid retrieval** — BM25 keyword search + FAISS semantic search combined via Reciprocal Rank Fusion
- **Cross-encoder reranking** — second-stage model that reads query and chunk together for precise scoring
- **Confidence gating** — refuses to answer from weak retrieval instead of hallucinating
- **Failure routing** — detects vague LLM responses and routes to web search (Brave → DuckDuckGo)
- **Multiple reliability checks** — PDF validation, chunk sanitization, drift guards, empty response handling

---

## Features

| Feature | Description |
|---|---|
| Multi-PDF Upload | Query across multiple documents simultaneously |
| Hybrid Search | BM25 (lexical) + FAISS (semantic) with RRF fusion |
| Cross-Encoder Reranking | `ms-marco-MiniLM-L-6-v2` for precision scoring |
| Dual-Query Retrieval | Searches with both original and rewritten query |
| Conversational Memory | Query rewriting resolves pronouns from history |
| Semantic Drift Guard | Cosine check rejects bad query rewrites (> 0.85) |
| Confidence Scoring | Per-chunk sigmoid scores from cross-encoder (0–1) |
| Context Sufficiency Check | LLM-based YES/NO gate before answer generation |
| Confidence Override | Forces web fallback if scores are too low |
| Web Search Fallback | Brave API (primary) + DuckDuckGo (backup) |
| Summary Mode | Detects broad questions, samples across entire document |
| Source Citations | Answers include `[1] (filename.pdf)` references |

---

## System Design Patterns

| Pattern | Where | Why |
|---|---|---|
| **Singleton** | `models.py` | ML models are 100s of MB — one instance, thread-safe double-checked locking |
| **Circuit Breaker** | `circuit_breaker.py` | Prevents cascading failures on Groq/web API outages — fails fast |
| **Strategy** | `retriever.py` via `config.py` | Retrieval methods are swappable through config |
| **Memoization** | `cache.py` | LRU cache for query embeddings — skips redundant inference |
| **Pipeline** | `pipeline.py` | Central orchestration layer; UI knows nothing about retrieval logic |
| **Separation of Concerns** | `app.py` → `pipeline.py` → modules | UI is a thin layer; business logic is in pipeline and modules |
| **Centralized Config** | `config.py` | Single source of truth for all hyperparameters |

---

## Architecture

```
User Question
     |
     v
Query Rewriter + Drift Guard (cosine > 0.85 or reject)
     |
     v
Summary question? -- YES --> sample broadly across document
     |                        
     NO
     |
     v
Hybrid Search
  BM25 + FAISS (dual-query)
  RRF Fusion --> Threshold Filter --> Dedup --> Cross-Encoder Rerank
     |
     v
Confidence Gate (mean < 0.35 OR max < 0.4 --> force fallback)
Sufficiency Check (LLM: "is context enough?")
     |             |
    YES            NO
     |              |
  Generate       Web Search (Brave -> DuckDuckGo)
  Answer         Combine contexts -> Generate
     |              |
  Pattern        Both failed?
  Check          Show failure message
  ("I don't know"? -> web search)
     |
  Display answer with citations
```

---

## Project Structure

```
config.py           Central config — all hyperparameters in one place
models.py           Singleton model manager — thread-safe ML model loading
cache.py            LRU query cache with hit/miss tracking
circuit_breaker.py  API resilience (CLOSED/OPEN/HALF_OPEN states)
pipeline.py         Core orchestrator — retriever + LLM + web search
app.py              Streamlit UI — delegates all logic to pipeline
retriever.py        Hybrid search — BM25 + FAISS + RRF + Cross-Encoder
llm.py              LLM integration — rewriting, sufficiency check, generation
chunking.py         Text chunking with sliding window overlap
web_search.py       Brave API (primary) + DuckDuckGo (fallback)
evaluation.py       Custom PDF evaluation with retrieval + answer metrics
benchmark.py        Standard benchmark (8 passages, 20 QA pairs)
Dockerfile          Containerized deployment
requirements.txt    Python dependencies
.env.example        Template for API keys
```

---

## Setup

### 1. Clone and create environment

```bash
git clone <your-repo-url>
cd VeriRag
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here       # Required — get from console.groq.com
BRAVE_API_KEY=your_brave_api_key_here     # Optional — falls back to DuckDuckGo
```

### 4. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Benchmark Results

```bash
python benchmark.py
```

Self-contained benchmark with 8 technical passages and 20 complex QA pairs (multi-hop, comparison, synthesis). No external dataset needed.

| Metric | Score |
|---|---|
| Recall@5 | 100.00% |
| MRR | 1.0000 |
| Answer F1 | 0.5072 |
| Keyword Accuracy | 100.00% |
| Hallucination Rate | 0.00% |
| Avg Latency | 4.86s |

All 20 questions retrieved the correct passage at rank 1. Zero hallucination.

For custom PDF evaluation:

```bash
python evaluation.py    # place test PDFs in docs/ first
```

---

## Docker

```bash
docker build -t verirag .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key_here verirag
```

---

## Configuration

All hyperparameters live in `config.py`:

| Setting | Key | Default |
|---|---|---|
| LLM Model | `LLM_MODEL` | `llama-3.1-8b-instant` |
| Bi-Encoder | `BI_ENCODER_MODEL` | `all-MiniLM-L6-v2` |
| Cross-Encoder | `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chunk Size | `CHUNK_SIZE` | 800 chars |
| Chunk Overlap | `CHUNK_OVERLAP` | 100 chars |
| Top-K Results | `TOP_K` | 5 |
| RRF Constant | `RRF_K` | 60 |
| Drift Threshold | `DRIFT_THRESHOLD` | cosine > 0.85 |
| Confidence Gate | `CONFIDENCE_MEAN_THRESHOLD` | mean < 0.35 OR max < 0.4 |
| Context Budget | `CONTEXT_BUDGET` | 3500 chars |
| LLM Temperature | `LLM_TEMPERATURE` | 0.2 |

---

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/) → **New app** → connect your repo → set main file to `app.py` → Deploy
3. Add secrets in **Settings → Secrets**:

```toml
GROQ_API_KEY = "your_key"
BRAVE_API_KEY = "your_key"   # optional
```

---

## Reliability Features

| Layer | What It Does |
|---|---|
| PDF Validation | Skips corrupted/encrypted/empty PDFs |
| Chunk Sanitization | Removes whitespace-only chunks |
| Query Drift Guard | Rejects rewrites with cosine similarity < 0.85 |
| RRF Threshold | Filters candidates below 30% of top score |
| Hard Cap | Limits cross-encoder input to initial_k candidates |
| Confidence Override | Forces fallback if mean < 0.35 OR max < 0.4 |
| Sufficiency Check | LLM verifies context before answering |
| Failure Detection | "I don't know" patterns route to web search |
| Document Guard | Skips web for questions about the file itself |
| Empty Response Check | Catches blank LLM outputs |

---

## Limitations

- Retrieval can miss chunks if they don't match query keywords or semantics
- Query rewriting may be imprecise for highly ambiguous conversations
- Confidence score reflects retrieval relevance, not answer correctness
- Evaluation uses keyword matching (RAGAS would provide semantic evaluation)
