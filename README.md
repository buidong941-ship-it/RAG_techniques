# RAG Learning Project

Học RAG từ baseline đến advanced techniques, theo roadmap [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques).

**Stack**: Python · FAISS · BGE-M3 · BM25 · BGE-reranker-v2-m3 · Ollama (Qwen2.5-7B) · FastAPI

---

## Kiến trúc

```
[Local Machine]                    [GPU Server]
  src/                               server/
  ├── client.py         ──HTTP──▶   ├── main.py          (FastAPI)
  ├── evaluation/                    ├── indexer.py       (PDF→Chunk→Embed→FAISS)
  └── experiments/                   ├── retriever.py     (dense/sparse/hybrid)
                                     ├── reranker.py      (BGE cross-encoder)
  data/                              ├── generator.py     (Ollama)
  └── evaluation/                    └── data/
      └── eval_dataset.json               ├── documents/   ← PDFs
                                          └── faiss_index/ ← persisted
  experiments/results/
  └── <experiment_name>/
      └── metrics.json
```

---

## GPU Requirements (minimum)

| Component | VRAM |
|---|---|
| BGE-M3 embedding | ~1.5 GB |
| BGE-reranker-v2-m3 | ~1.2 GB |
| Qwen2.5-7B (4-bit via Ollama) | ~5.5 GB |
| **Total minimum** | **~8–9 GB** |
| **Recommended** | **12–16 GB** |

---

## Setup

### 1. Server (GPU Machine — Ubuntu)

```bash
# Clone project
git clone <repo> && cd RAG_testing

# Create .env
cp .env.example .env
# Edit .env: set RAG_OLLAMA_BASE_URL, RAG_PORT, etc.

# Install dependencies
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b

# Start server
python -m server.main
# Or with uvicorn directly:
# uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### 2. Local Machine (Client)

```bash
pip install -r requirements_client.txt
```

### 3. Forward server port (optional)

```bash
# On local machine, forward server port via SSH:
ssh -L 8000:localhost:8000 user@server_ip
# Then use http://localhost:8000 as server URL
```

---

## Workflow

### Step 1: Upload & Index Documents

```bash
# Upload PDFs from local folder to server and index them
python -m src.experiments.ingest_docs \
    --server http://localhost:8000 \
    --docs /path/to/your/pdfs
```

### Step 2: Provide Evaluation Dataset

Edit `data/evaluation/eval_dataset.json` with your 50-100 QA pairs:

```json
[
  {
    "question": "Câu hỏi của bạn?",
    "ground_truth": "Đáp án đúng.",
    "source": "document.pdf",
    "page": 3,
    "relevant_chunks": ["document.pdf::chunk_12", "document.pdf::chunk_13"]
  }
]
```

**⚠️ QUAN TRỌNG**: Không thay đổi dataset sau khi đã bắt đầu benchmark!

### Step 3: Run Baseline

```bash
python -m src.experiments.run_baseline --server http://localhost:8000
```

Results saved to `experiments/results/baseline/metrics.json`

### Step 4: Run Experiments

```bash
# Example: test hybrid retrieval
python -m src.experiments.run_experiment \
    --server http://localhost:8000 \
    --name hybrid_retrieval \
    --mode hybrid \
    --top-k 10

# Example: test reranking
python -m src.experiments.run_experiment \
    --server http://localhost:8000 \
    --name reranking \
    --rerank

# Example: test smaller chunk size
python -m src.experiments.run_experiment \
    --server http://localhost:8000 \
    --name chunk_size_200 \
    --config '{"chunk_size": 200, "chunk_overlap": 20}'
```

### Step 5: Compare Results

```bash
# Compare specific experiments
python -m src.experiments.compare_results \
    --experiments baseline hybrid_retrieval reranking

# Compare all
python -m src.experiments.compare_results --all
```

---

## Project Roadmap

### Level 1 — Understand RAG
- [x] **Baseline RAG** — dense retrieval, flat chunking
- [ ] **Chunk Size** — experiment with different chunk_size values
- [ ] **Proposition Chunking** — LLM-assisted atomic fact extraction
- [ ] **Semantic Chunking** — split on embedding similarity breakpoints

### Level 2 — Improve Retrieval
- [ ] **Query Transformations** — multi-query, step-back
- [ ] **HyDE** — Hypothetical Document Embeddings
- [ ] **Fusion Retrieval** — BM25 + dense with RRF
- [ ] **Reranking** — BGE cross-encoder
- [ ] **Filtering** — metadata-based filters
- [ ] **Hierarchical Indices** — summary + detail index

### Level 3 — Improve Context
- [ ] **Contextual Headers** — section headers in chunk metadata
- [ ] **Relevant Segment Extraction** — RSE
- [ ] **Context Window Enhancement** — expand with surrounding sentences
- [ ] **Contextual Compression** — compress before LLM
- [ ] **Document Augmentation** — synthetic QA in chunk metadata

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/ingest` | POST | Ingest document (JSON + base64) |
| `/ingest/upload` | POST | Ingest via multipart upload |
| `/retrieve` | POST | Retrieve chunks only |
| `/query` | POST | Full RAG query |
| `/index/info` | GET | Index statistics |
| `/index` | DELETE | Clear index |
| `/config` | POST | Update runtime config |
| `/_judge` | POST | LLM judge (internal) |

Interactive docs: `http://localhost:8000/docs`

---

## Evaluation Metrics

### Retrieval
| Metric | Meaning |
|---|---|
| Recall@K | % of relevant chunks retrieved |
| Precision@K | % of retrieved chunks that are relevant |
| MRR | How high up is the first relevant result? |
| NDCG@K | Quality of ranking order |

### Generation (LLM-as-judge)
| Metric | Meaning |
|---|---|
| answer_correctness | Answer vs ground truth |
| faithfulness | Answer supported by context? |
| context_relevance | Context relevant to question? |
| answer_relevance | Answer addresses the question? |

### Failure Analysis
- **retrieval_failure**: Retriever didn't find the right context → RAG trả lời sai
- **generation_failure**: Context đúng nhưng LLM vẫn sai
- **both_failure**: Cả hai đều sai
- **ok**: Cả hai đều đạt
