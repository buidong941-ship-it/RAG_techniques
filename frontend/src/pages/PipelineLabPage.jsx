import { useState, useEffect, useRef } from 'react'
import { getPresets, pipelineQuery, pipelineCompare } from '../api.js'

// ── Strategy options ────────────────────────────────────────────────────────
const STRATEGY_OPTIONS = {
  query_transformer: [
    { value: 'passthrough',  label: 'Passthrough (baseline)' },
    { value: 'multi_query',  label: 'Multi-Query' },
    { value: 'hyde',         label: 'HyDE' },
    { value: 'step_back',    label: 'Step-Back' },
  ],
  retriever: [
    { value: 'dense',   label: 'Dense (FAISS)' },
    { value: 'sparse',  label: 'Sparse (BM25)' },
    { value: 'hybrid',  label: 'Hybrid (RRF)' },
  ],
  reranker: [
    { value: 'none', label: 'None' },
    { value: 'bge',  label: 'BGE Reranker' },
  ],
  context_processor: [
    { value: 'passthrough',        label: 'Passthrough (baseline)' },
    { value: 'window_expand',      label: 'Window Expand' },
    { value: 'compress',           label: 'Contextual Compress' },
    { value: 'rse',                label: 'RSE (Segment Merge)' },
    { value: 'contextual_headers', label: 'Contextual Headers' },
  ],
}

const TECHNIQUE_DESCRIPTIONS = {
  passthrough:        'Không thay đổi — baseline mặc định',
  multi_query:        'LLM tạo N query khác nhau, retrieve cho mỗi cái, merge bằng RRF',
  hyde:               'LLM tạo hypothetical document, embed nó để search thay query gốc',
  step_back:          'LLM rephrase sang câu hỏi tổng quát hơn trước khi search',
  dense:              'FAISS cosine similarity với BGE-M3 embeddings',
  sparse:             'BM25 keyword matching (rank_bm25)',
  hybrid:             'RRF kết hợp dense + sparse',
  none:               'Không rerank',
  bge:                'BAAI/bge-reranker-v2-m3 cross-encoder reranking',
  window_expand:      'Mở rộng mỗi chunk thêm ±k câu xung quanh để thêm context',
  compress:           'LLM giữ lại phần liên quan nhất trong mỗi chunk',
  rse:                'Ghép các chunk liên tiếp thành segment dài hơn',
  contextual_headers: 'Thêm tên document + section vào đầu mỗi chunk',
}

const DEFAULT_CONFIG = {
  query_transformer: 'passthrough',
  retriever: 'dense',
  reranker: 'none',
  context_processor: 'passthrough',
  top_k: 5,
  multi_query_n: 3,
  window_size: 2,
}

const COLORS = ['var(--accent)', 'var(--green)', 'var(--amber)', 'var(--blue)']

// ── Config builder panel ───────────────────────────────────────────────────
function ConfigPanel({ config, index, onUpdate, onRemove, canRemove, color }) {
  const Field = ({ label, field }) => (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        {label}
      </div>
      <select
        value={config[field]}
        onChange={e => onUpdate({ ...config, [field]: e.target.value })}
        style={{ width: '100%', fontSize: 12.5 }}
      >
        {STRATEGY_OPTIONS[field].map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3, lineHeight: 1.4 }}>
        {TECHNIQUE_DESCRIPTIONS[config[field]]}
      </div>
    </div>
  )

  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: `1px solid ${color}44`,
      borderRadius: 'var(--radius-lg)',
      padding: 16,
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0 }} />
        <div style={{ fontSize: 13, fontWeight: 600 }}>Config {index + 1}</div>
        {canRemove && (
          <button
            onClick={onRemove}
            className="btn btn-ghost"
            style={{ marginLeft: 'auto', padding: '2px 8px', fontSize: 11 }}
          >×</button>
        )}
      </div>
      <Field label="Query Transform" field="query_transformer" />
      <Field label="Retriever" field="retriever" />
      <Field label="Reranker" field="reranker" />
      <Field label="Context Processor" field="context_processor" />
      <div style={{ display: 'flex', gap: 8 }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Top-K</div>
          <input type="number" min={1} max={20} value={config.top_k}
            onChange={e => onUpdate({ ...config, top_k: Number(e.target.value) })}
            style={{ fontSize: 12 }} />
        </div>
        {config.query_transformer === 'multi_query' && (
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Queries (N)</div>
            <input type="number" min={2} max={6} value={config.multi_query_n}
              onChange={e => onUpdate({ ...config, multi_query_n: Number(e.target.value) })}
              style={{ fontSize: 12 }} />
          </div>
        )}
        {config.context_processor === 'window_expand' && (
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Window ±k</div>
            <input type="number" min={1} max={5} value={config.window_size}
              onChange={e => onUpdate({ ...config, window_size: Number(e.target.value) })}
              style={{ fontSize: 12 }} />
          </div>
        )}
      </div>
    </div>
  )
}

// ── Result card ────────────────────────────────────────────────────────────
function ResultCard({ result, index, color, query }) {
  const [showChunks, setShowChunks] = useState(false)
  if (!result) return null
  if (result.error) return (
    <div style={{
      background: 'var(--bg-surface)', border: `1px solid ${color}44`,
      borderRadius: 'var(--radius-lg)', padding: 16
    }}>
      <div className="alert alert-error">⚠️ {result.error}</div>
    </div>
  )

  const cfg = result.pipeline_config || {}
  return (
    <div style={{
      background: 'var(--bg-surface)', border: `1px solid ${color}44`,
      borderRadius: 'var(--radius-lg)', padding: 16, display: 'flex', flexDirection: 'column', gap: 10,
    }}>
      {/* Config badge row */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {[
          cfg.query_transformer !== 'passthrough' && cfg.query_transformer,
          cfg.retriever,
          cfg.reranker !== 'none' && `rerank:${cfg.reranker}`,
          cfg.context_processor !== 'passthrough' && cfg.context_processor,
        ].filter(Boolean).map(label => (
          <span key={label} className="badge" style={{ background: `${color}22`, color, fontSize: 11 }}>
            {label}
          </span>
        ))}
      </div>

      {/* Answer */}
      <div style={{ fontSize: 13.5, lineHeight: 1.7, color: 'var(--text-primary)' }}>
        {result.answer || <span style={{ color: 'var(--text-muted)' }}>No answer</span>}
      </div>

      {/* Query variants (if multi-query or hyde) */}
      {result.query_variants?.length > 1 && (
        <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
          <b>Query variants:</b> {result.query_variants.map((q, i) => (
            <div key={i} style={{ marginLeft: 8, marginTop: 2 }}>• {q.slice(0, 80)}…</div>
          ))}
        </div>
      )}

      {/* Meta */}
      <div style={{ display: 'flex', gap: 12, fontSize: 11.5, color: 'var(--text-muted)' }}>
        {result.retrieval_time != null && <span>retrieval {(result.retrieval_time * 1000).toFixed(0)}ms</span>}
        {result.generation_time != null && <span>gen {(result.generation_time * 1000).toFixed(0)}ms</span>}
        {result.chunks?.length > 0 && (
          <span style={{ cursor: 'pointer', color: 'var(--accent-light)' }}
            onClick={() => setShowChunks(v => !v)}>
            {showChunks ? '▲' : '▼'} {result.chunks.length} chunks
          </span>
        )}
      </div>

      {/* Chunks */}
      {showChunks && result.chunks?.length > 0 && (
        <div className="chunks-panel">
          {result.chunks.map((c, i) => (
            <div key={c.chunk_id} className="chunk-item">
              <div className="chunk-header">
                <span className="chunk-source">#{i + 1} {c.metadata?.doc_id}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-light)' }}>
                  {c.score.toFixed(4)}
                </span>
              </div>
              <p className="chunk-text">{c.text.slice(0, 240)}…</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function PipelineLabPage() {
  const [configs, setConfigs] = useState([{ ...DEFAULT_CONFIG }])
  const [query, setQuery]     = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [presets, setPresets] = useState({})
  const textareaRef = useRef(null)

  useEffect(() => {
    getPresets().then(d => setPresets(d.presets || {})).catch(() => {})
  }, [])

  const addConfig = () => {
    if (configs.length >= 4) return
    setConfigs(prev => [...prev, { ...DEFAULT_CONFIG }])
  }

  const updateConfig = (i, cfg) => {
    setConfigs(prev => prev.map((c, idx) => idx === i ? cfg : c))
  }

  const removeConfig = (i) => {
    setConfigs(prev => prev.filter((_, idx) => idx !== i))
    setResults(prev => prev.filter((_, idx) => idx !== i))
  }

  const loadPreset = (presetName) => {
    const preset = presets[presetName]
    if (!preset) return
    setConfigs([{ ...DEFAULT_CONFIG, ...preset }])
    setResults([])
  }

  const handleRun = async () => {
    if (!query.trim() || loading) return
    setLoading(true)
    setError(null)
    setResults([])
    try {
      if (configs.length === 1) {
        const res = await pipelineQuery(query, configs[0])
        setResults([res])
      } else {
        const res = await pipelineCompare(query, configs)
        setResults(res.results || [])
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleRun() }
  }

  const colWidth = configs.length === 1 ? '100%'
    : configs.length === 2 ? 'calc(50% - 6px)'
    : configs.length === 3 ? 'calc(33.3% - 8px)'
    : 'calc(25% - 9px)'

  return (
    <div className="page">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* Presets */}
        {Object.keys(presets).length > 0 && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Presets:</span>
            {Object.keys(presets).map(name => (
              <button key={name} className="btn btn-ghost"
                style={{ padding: '4px 12px', fontSize: 12 }}
                onClick={() => loadPreset(name)}>
                {name}
              </button>
            ))}
          </div>
        )}

        {/* Query input */}
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
            <textarea
              ref={textareaRef}
              id="lab-query"
              placeholder="Nhập câu hỏi để test tất cả configs… (Enter để chạy)"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKey}
              rows={2}
              style={{ flex: 1 }}
            />
            <button id="lab-run" className="btn btn-primary"
              onClick={handleRun}
              disabled={!query.trim() || loading}
              style={{ height: 56 }}>
              {loading
                ? <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                : `▶ Run ${configs.length > 1 ? `(${configs.length})` : ''}`
              }
            </button>
          </div>
        </div>

        {error && <div className="alert alert-error">⚠️ {error}</div>}

        {/* Config panels + results */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          {configs.map((cfg, i) => (
            <div key={i} style={{ width: colWidth, display: 'flex', flexDirection: 'column', gap: 10 }}>
              <ConfigPanel
                config={cfg}
                index={i}
                color={COLORS[i % COLORS.length]}
                onUpdate={c => updateConfig(i, c)}
                onRemove={() => removeConfig(i)}
                canRemove={configs.length > 1}
              />
              {results[i] && (
                <ResultCard result={results[i]} index={i} color={COLORS[i % COLORS.length]} query={query} />
              )}
              {loading && !results[i] && (
                <div style={{ display: 'flex', justifyContent: 'center', padding: 20 }}>
                  <div className="spinner" />
                </div>
              )}
            </div>
          ))}

          {/* Add config button */}
          {configs.length < 4 && (
            <div style={{ width: colWidth }}>
              <button
                id="lab-add-config"
                className="btn btn-ghost"
                onClick={addConfig}
                style={{
                  width: '100%', height: 120,
                  border: '2px dashed var(--border)',
                  borderRadius: 'var(--radius-lg)',
                  flexDirection: 'column', gap: 6,
                  fontSize: 13, color: 'var(--text-muted)',
                }}>
                <div style={{ fontSize: 24 }}>+</div>
                <div>Add Config</div>
              </button>
            </div>
          )}
        </div>

        {/* Empty state */}
        {!loading && results.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">🔬</div>
            <div className="empty-state-text">
              Cấu hình pipeline bên trên, nhập câu hỏi và nhấn Run.<br />
              Thêm nhiều configs để so sánh kết quả song song.
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
