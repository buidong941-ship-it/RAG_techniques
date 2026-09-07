# RAG Laboratory — Modular Techniques Implementation Plan

## Tổng quan

Xây dựng hệ thống **pluggable strategy architecture** cho phép ON/OFF từng technique
qua API config, so sánh cùng query với nhiều cấu hình khác nhau, và chạy benchmark có progress UI.

---

## Kiến trúc: Strategy Pattern

Mỗi bước trong RAG pipeline là một **strategy** có thể swap:

```
Query
  │
  ▼
[Query Transformer]  ← passthrough | multi_query | hyde | step_back
  │
  ▼
[Retriever]          ← dense | sparse | hybrid | hierarchical
  │
  ▼
[Reranker]           ← none | bge
  │
  ▼
[Context Processor]  ← passthrough | compress | window_expand | rse | headers
  │
  ▼
[Generator]          ← Ollama (prompt template có thể custom)
  │
  ▼
Answer + Chunks
```

**Pipeline Config** — client gửi JSON, server chọn đúng strategy:

```json
{
  "experiment_name": "hyde_hybrid_rerank",
  "query_transformer": "hyde",
  "retriever": "hybrid",
  "reranker": "bge",
  "context_processor": "compress",
  "top_k": 5
}
```

---

## Proposed Changes

### Server

---

#### [NEW] `server/techniques/` — thư mục technique modules

#### [NEW] `server/techniques/chunkers.py`

| Strategy | Mô tả |
|---|---|
| `sliding_window` | Baseline hiện tại |
| `semantic` | Split dựa trên cosine similarity breakpoints giữa sentences |
| `proposition` | LLM tách text thành atomic facts (mỗi chunk = 1 fact) |

#### [NEW] `server/techniques/query_transformers.py`

| Strategy | Mô tả |
|---|---|
| `passthrough` | Baseline — giữ nguyên query |
| `multi_query` | LLM generate N query variants, retrieve cho mỗi cái, RRF để merge |
| `hyde` | LLM generate hypothetical document, embed để search thay query gốc |
| `step_back` | LLM rephrase sang câu hỏi tổng quát hơn trước khi search |

#### [NEW] `server/techniques/context_processors.py`

| Strategy | Mô tả |
|---|---|
| `passthrough` | Baseline — giữ nguyên chunks |
| `window_expand` | Mở rộng mỗi chunk thêm ±k sentences xung quanh |
| `compress` | LLM giữ lại phần relevant nhất trong mỗi chunk |
| `rse` | Relevant Segment Extraction — ghép các chunks liên tiếp thành segment |
| `contextual_headers` | Prepend section heading vào mỗi chunk trước khi đưa LLM |

#### [NEW] `server/pipeline.py`

Orchestrator nhận `PipelineConfig`, gọi đúng strategy cho từng bước.

#### [MODIFY] `server/main.py`

Thêm 2 nhóm endpoints:

```
POST /pipeline/query      — RAG query với arbitrary pipeline config
POST /pipeline/compare    — Cùng query với N configs, trả về N kết quả song song
POST /benchmark/run       — Chạy full evaluation, stream progress qua SSE
GET  /benchmark/status    — Trạng thái benchmark đang chạy
GET  /benchmark/list      — Danh sách kết quả đã chạy
```

#### [MODIFY] `server/models.py`

Thêm `PipelineConfig`, `CompareRequest`, `BenchmarkRunRequest` schemas.

#### [MODIFY] `server/indexer.py`

Thêm support `ChunkerStrategy` parameter khi ingest (để experiment semantic/proposition chunking).

---

### Frontend

---

#### [NEW] `frontend/src/pages/PipelineLabPage.jsx` — 🔬 Pipeline Lab

- **Query input** chung
- **Config builder**: dropdown strategy cho mỗi bước
- **"+ Add Config"**: tối đa 3 cấu hình cạnh nhau
- **Run All**: gửi query đến tất cả configs đồng thời
- **Side-by-side view**: so sánh answer + chunks
- **Preset buttons**: "Baseline", "HyDE", "Semantic+Rerank", v.v.

#### [NEW] `frontend/src/pages/BenchmarkPage.jsx` — 🏃 Benchmark

- Config panel chọn pipeline config
- Đặt tên experiment
- **Run Benchmark** → trigger SSE streaming từ server
- **Live progress bar**: "Sample X/N"
- **Live metrics**: cập nhật realtime
- Auto-load kết quả vào Dashboard sau khi xong

#### [MODIFY] `frontend/src/App.jsx`

Thêm 2 nav items: 🔬 Pipeline Lab, 🏃 Benchmark.

#### [MODIFY] `frontend/src/api.js`

Thêm: `pipelineQuery`, `pipelineCompare`, `runBenchmark`, `benchmarkStatus`.

---

## Techniques Roadmap

| # | Technique | Level | LLM cần? | Độ phức tạp |
|---|---|---|---|---|
| 1 | **Semantic Chunking** | L1 | ❌ | Thấp |
| 2 | **Context Window Expand** | L3 | ❌ | Thấp |
| 3 | **Contextual Headers** | L3 | ❌ | Thấp |
| 4 | **HyDE** | L2 | ✅ | Trung bình |
| 5 | **Multi-Query** | L2 | ✅ | Trung bình |
| 6 | **Step-Back** | L2 | ✅ | Trung bình |
| 7 | **Contextual Compression** | L3 | ✅ | Trung bình |
| 8 | **RSE** | L3 | ❌ | Trung bình |
| 9 | **Proposition Chunking** | L1 | ✅ | Cao |
| 10 | **Hierarchical Indices** | L2 | ✅ | Cao |

---

## Open Questions

> [!IMPORTANT]
> **Proposition Chunking** cần LLM call cho MỖI chunk khi ingest → rất chậm với dataset lớn.
> Bạn có muốn implement không?

> [!NOTE]
> **Hierarchical Indices** cần 2 FAISS index riêng (summary + detail) — implement sau cùng?

> [!NOTE]
> SSE streaming cho Benchmark progress: FastAPI native support — straightforward.
