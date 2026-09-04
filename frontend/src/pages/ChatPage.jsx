import { useState, useRef, useEffect } from 'react'
import { query as apiQuery } from '../api.js'

/* ── Score colouring ─────────────────────────────────────────── */
function scoreColor(s) {
  if (s >= 0.7) return 'var(--green)'
  if (s >= 0.4) return 'var(--amber)'
  return 'var(--red)'
}

/* ── Single retrieved chunk ────────────────────────────────────── */
function ChunkCard({ chunk, rank }) {
  return (
    <div className="chunk-item">
      <div className="chunk-header">
        <span className="chunk-source">
          #{rank}  {chunk.metadata?.doc_id}
          {chunk.metadata?.page ? ` · p.${chunk.metadata.page}` : ''}
        </span>
        <span
          className="badge"
          style={{
            background: 'transparent',
            color: scoreColor(chunk.score),
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            padding: '0',
          }}
        >
          score {chunk.score.toFixed(4)}
        </span>
      </div>
      <p className="chunk-text">{chunk.text.slice(0, 280)}{chunk.text.length > 280 ? '…' : ''}</p>
    </div>
  )
}

/* ── Message bubble ───────────────────────────────────────────── */
function Message({ msg }) {
  const [showChunks, setShowChunks] = useState(false)
  const isUser = msg.role === 'user'

  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-bubble">{msg.content}</div>

      {!isUser && (
        <div className="message-meta">
          {msg.model && <span>{msg.model}</span>}
          {msg.retrieval_time != null && (
            <span>retrieval {(msg.retrieval_time * 1000).toFixed(0)}ms</span>
          )}
          {msg.generation_time != null && (
            <span>gen {(msg.generation_time * 1000).toFixed(0)}ms</span>
          )}
          {msg.chunks?.length > 0 && (
            <span
              className="chunks-toggle"
              onClick={() => setShowChunks(v => !v)}
            >
              {showChunks ? '▲' : '▼'} {msg.chunks.length} chunks
            </span>
          )}
        </div>
      )}

      {showChunks && msg.chunks?.length > 0 && (
        <div className="chunks-panel">
          {msg.chunks.map((c, i) => (
            <ChunkCard key={c.chunk_id} chunk={c} rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  )
}

/* ── Chat page ────────────────────────────────────────────────── */
export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)

  // Options
  const [topK,   setTopK]   = useState(5)
  const [mode,   setMode]   = useState('dense')
  const [rerank, setRerank] = useState(false)

  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSend = async () => {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setError(null)

    setMessages(prev => [...prev, { role: 'user', content: q }])
    setLoading(true)

    try {
      const res = await apiQuery({
        query: q,
        top_k: topK,
        mode,
        rerank,
      })
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: res.answer,
          chunks: res.chunks,
          retrieval_time: res.retrieval_time,
          generation_time: res.generation_time,
          model: res.model,
        },
      ])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
      setTimeout(() => textareaRef.current?.focus(), 50)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-wrap">
      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">💬</div>
            <div className="empty-state-text">
              Đặt câu hỏi về tài liệu của bạn.<br />
              Câu trả lời sẽ hiển thị cùng các chunks được truy xuất.
            </div>
          </div>
        )}

        {messages.map((msg, i) => <Message key={i} msg={msg} />)}

        {loading && (
          <div className="message assistant">
            <div className="message-bubble" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div className="spinner" />
              <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>Đang xử lý…</span>
            </div>
          </div>
        )}

        {error && (
          <div className="alert alert-error" style={{ maxWidth: 600 }}>
            ⚠️ {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="chat-input-area">
        {/* Options row */}
        <div className="chat-options">
          <div className="option-group">
            <span>Mode</span>
            <select value={mode} onChange={e => setMode(e.target.value)} id="chat-mode">
              <option value="dense">Dense</option>
              <option value="sparse">Sparse (BM25)</option>
              <option value="hybrid">Hybrid</option>
            </select>
          </div>

          <div className="option-group">
            <span>Top-K</span>
            <input
              id="chat-topk"
              type="number"
              min={1} max={20}
              value={topK}
              onChange={e => setTopK(Number(e.target.value))}
            />
          </div>

          <div className="option-group toggle-row">
            <label className="toggle">
              <input
                id="chat-rerank"
                type="checkbox"
                checked={rerank}
                onChange={e => setRerank(e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
            <span>Rerank</span>
          </div>

          {messages.length > 0 && (
            <button
              className="btn btn-ghost"
              style={{ marginLeft: 'auto', padding: '4px 12px', fontSize: 12 }}
              onClick={() => setMessages([])}
            >
              Xoá chat
            </button>
          )}
        </div>

        {/* Text + send */}
        <div className="chat-input-row">
          <textarea
            ref={textareaRef}
            id="chat-input"
            placeholder="Nhập câu hỏi… (Enter để gửi, Shift+Enter để xuống dòng)"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            rows={1}
          />
          <button
            id="chat-send"
            className="btn btn-primary"
            onClick={handleSend}
            disabled={!input.trim() || loading}
            style={{ height: 44 }}
          >
            {loading ? <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> : '↑ Gửi'}
          </button>
        </div>
      </div>
    </div>
  )
}
