Tôi muốn học RAG (Retrieval-Augmented Generation) theo hướng thực hành + thực nghiệm, sử dụng repository NirDiamant/RAG_Techniques làm roadmap chính.

Hãy đóng vai một RAG Engineer/Mentor và giúp tôi xây dựng một mini-project RAG chạy local ( kết hợp backend ở server riêng ), sau đó lần lượt triển khai các kỹ thuật RAG trong repository này để tôi có thể đo lường và trực tiếp thấy sự khác biệt giữa từng kỹ thuật.

1. Mục tiêu

Tôi không muốn chỉ đọc notebook/code trong repository.

Tôi muốn:

Hiểu lý thuyết của từng RAG technique.
Tự implement technique đó vào mini-project của mình.
Chạy cùng một evaluation dataset.
So sánh kết quả với baseline.
Phân tích tại sao technique cải thiện hoặc làm giảm kết quả.
Hiểu technique giải quyết failure mode nào của RAG.

Repository tham khảo chính:

NirDiamant/RAG_Techniques

Hãy sử dụng repository này làm nguồn tham khảo cho danh sách và thứ tự các technique, nhưng không copy code một cách máy móc. Hãy giúp tôi xây dựng một project có kiến trúc riêng, đơn giản và dễ hiểu.

2. Công nghệ

Ưu tiên chạy hoàn toàn local:

Python
FAISS
BGE-M3 hoặc embedding model phù hợp cho tiếng Việt
BM25 nếu technique yêu cầu
Reranker chạy local
Ollama
Qwen hoặc một LLM local phù hợp
PyMuPDF hoặc thư viện tương đương để đọc PDF

Không sử dụng API trả phí nếu không cần thiết.

Nếu repository sử dụng framework như LangChain/LlamaIndex, hãy giải thích trước abstraction đó và nếu có thể hãy implement phiên bản đơn giản bằng Python thuần để tôi hiểu bản chất.

3. Dataset

Tôi sẽ sử dụng một tập tài liệu PDF/document làm knowledge base.

Project cần có:

data/
├── documents/
└── evaluation/

Evaluation dataset khoảng 50–100 câu hỏi. ( cái này tôi sẽ cung cấp bạn chỉ cần tạo file trống thôi )

Mỗi sample nên có:

{
"question": "...",
"ground_truth": "...",
"source": "...",
"page": "...",
"relevant_chunks": [...]
}

Evaluation dataset phải được cố định trong toàn bộ quá trình benchmark.

Không được thay đổi dataset giữa các experiment.

4. Baseline RAG

Trước tiên hãy xây dựng một baseline RAG đơn giản:

Document
→ Text Extraction
→ Chunking
→ Embedding
→ FAISS
→ Top-K Retrieval
→ LLM
→ Answer

Ví dụ configuration:

chunk_size = 500
chunk_overlap = 50
embedding = BGE-M3
top_k = 5

Baseline phải thật đơn giản để tôi hiểu rõ:

document ingestion
chunking
embedding
vector indexing
similarity search
context construction
prompt
generation
5. Evaluation Framework

Hãy xây dựng một evaluation framework dùng chung cho tất cả experiment.

Retrieval metrics

Đánh giá:

Recall@K
Precision@K
MRR
NDCG@K
Generation metrics

Đánh giá:

Answer correctness
Faithfulness
Context relevance
Answer relevance
System metrics

Đánh giá thêm:

latency
embedding time
retrieval time
generation time
memory usage
number of retrieved chunks
token usage nếu có thể đo được

Quan trọng:

Phải phân biệt:

Retrieval Failure

và

Generation Failure.

Ví dụ:

Retriever lấy sai context
→ RAG trả lời sai

khác với:

Retriever lấy đúng context
→ LLM vẫn trả lời sai.

6. Sử dụng NirDiamant/RAG_Techniques làm roadmap

LEVEL 1 — Understand RAG
Basic RAG
Chunk Size
Proposition Chunking
Semantic Chunking

LEVEL 2 — Improve Retrieval
Query Transformations
HyDE
Fusion Retrieval
Reranking
Filtering
Hierarchical Indices

LEVEL 3 — Improve Context
Contextual Headers
Relevant Segment Extraction
Context Window Enhancement
Contextual Compression
Document Augmentation
