import { useState, useRef, useCallback } from 'react'
import { indexInfo, ingestFile, clearIndex } from '../api.js'

const ICONS = { pdf: '📄', txt: '📝', md: '📋', default: '📁' }

function getIcon(name) {
  const ext = name.split('.').pop()?.toLowerCase()
  return ICONS[ext] || ICONS.default
}

function DocRow({ name, onRemove }) {
  return (
    <div className="doc-row">
      <span className="doc-icon">{getIcon(name)}</span>
      <span className="doc-name">{name}</span>
    </div>
  )
}

function StatCard({ label, value, sub, color }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color: color || 'var(--text-primary)' }}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

export default function DocumentsPage() {
  const [info, setInfo]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError]     = useState(null)
  const [success, setSuccess] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [chunkSize, setChunkSize]     = useState(500)
  const [chunkOverlap, setChunkOverlap] = useState(50)
  const fileInputRef = useRef(null)

  const fetchInfo = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await indexInfo()
      setInfo(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleUpload = async (files) => {
    if (!files || files.length === 0) return
    setUploading(true)
    setError(null)
    setSuccess(null)
    let ok = 0, fail = 0
    for (const file of files) {
      try {
        const res = await ingestFile(file, chunkSize, chunkOverlap)
        ok++
      } catch (e) {
        fail++
        setError(`${file.name}: ${e.message}`)
      }
    }
    setUploading(false)
    if (ok > 0) {
      setSuccess(`✓ Đã index ${ok} file${ok > 1 ? 's' : ''}.`)
      fetchInfo()
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleUpload(Array.from(e.dataTransfer.files))
  }

  const handleClear = async () => {
    if (!window.confirm('Xoá toàn bộ index? Hành động này không thể hoàn tác.')) return
    try {
      await clearIndex()
      setInfo(null)
      setSuccess('Index đã được xoá.')
    } catch (e) {
      setError(e.message)
    }
  }

  // Fetch on mount
  useState(() => { fetchInfo() }, [])

  return (
    <div className="page">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 900 }}>

        {/* Stats */}
        {info && (
          <div className="grid-3">
            <StatCard label="Tổng chunks" value={info.num_chunks.toLocaleString()} color="var(--accent-light)" />
            <StatCard label="Documents" value={info.num_documents} />
            <StatCard
              label="Index size"
              value={`${info.index_size_mb.toFixed(1)} MB`}
              sub={info.faiss_index_type}
            />
          </div>
        )}

        {/* Alerts */}
        {error   && <div className="alert alert-error">⚠️ {error}</div>}
        {success && <div className="alert alert-success">{success}</div>}

        {/* Upload section */}
        <div className="card">
          <div className="card-title">Upload Documents</div>

          {/* Chunking options */}
          <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
            <div className="option-group" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
              <label style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Chunk size</label>
              <input
                id="doc-chunk-size"
                type="number"
                value={chunkSize}
                onChange={e => setChunkSize(Number(e.target.value))}
                style={{ width: 100 }}
              />
            </div>
            <div className="option-group" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
              <label style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Chunk overlap</label>
              <input
                id="doc-chunk-overlap"
                type="number"
                value={chunkOverlap}
                onChange={e => setChunkOverlap(Number(e.target.value))}
                style={{ width: 100 }}
              />
            </div>
            <div style={{ marginLeft: 'auto', alignSelf: 'flex-end' }}>
              <button id="doc-refresh" className="btn btn-ghost" onClick={fetchInfo} disabled={loading}>
                {loading ? <div className="spinner" style={{ width: 14, height: 14 }} /> : '↻'} Refresh
              </button>
            </div>
          </div>

          {/* Drop zone */}
          <div
            id="doc-dropzone"
            className={`drop-zone ${dragging ? 'dragging' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="drop-zone-icon">{uploading ? '⏳' : '📂'}</div>
            <div className="drop-zone-text">
              {uploading ? 'Đang upload và index…' : 'Kéo thả PDF / TXT vào đây, hoặc click để chọn file'}
            </div>
            <div className="drop-zone-sub">Hỗ trợ: .pdf, .txt, .md</div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.md"
            style={{ display: 'none' }}
            onChange={e => handleUpload(Array.from(e.target.files))}
          />
        </div>

        {/* Document list */}
        {info?.documents?.length > 0 && (
          <div className="card">
            <div className="card-title">Documents in Index ({info.documents.length})</div>
            <div className="doc-grid">
              {info.documents.map(name => <DocRow key={name} name={name} />)}
            </div>
          </div>
        )}

        {/* Danger zone */}
        <div className="card" style={{ borderColor: 'rgba(248,113,113,0.2)' }}>
          <div className="card-title" style={{ color: 'var(--red)' }}>Danger Zone</div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 4 }}>Xoá toàn bộ index</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Xoá tất cả vectors và metadata. Documents gốc vẫn còn trên server.
              </div>
            </div>
            <button id="doc-clear-index" className="btn btn-danger" onClick={handleClear}>
              🗑 Xoá Index
            </button>
          </div>
        </div>

      </div>
    </div>
  )
}
