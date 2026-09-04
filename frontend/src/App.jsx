import { useState, useEffect, useCallback } from 'react'
import ChatPage      from './pages/ChatPage.jsx'
import DocumentsPage from './pages/DocumentsPage.jsx'
import DashboardPage from './pages/DashboardPage.jsx'
import { health, getServerUrl, setServerUrl } from './api.js'

const NAV = [
  { id: 'chat',      icon: '💬', label: 'Chat' },
  { id: 'documents', icon: '📁', label: 'Documents' },
  { id: 'dashboard', icon: '📊', label: 'Dashboard' },
]

const PAGE_TITLES = {
  chat:      { title: 'Chat', sub: 'Retrieval-Augmented Generation' },
  documents: { title: 'Documents', sub: 'Upload & manage knowledge base' },
  dashboard: { title: 'Experiment Dashboard', sub: 'Compare evaluation results' },
}

export default function App() {
  const [page, setPage]           = useState('chat')
  const [serverUrl, _setServerUrl] = useState(getServerUrl)
  const [serverStatus, setStatus] = useState('checking') // 'checking' | 'online' | 'offline'

  /* ── Health check ──────────────────────────────────────── */
  const checkHealth = useCallback(async () => {
    setStatus('checking')
    try {
      await health()
      setStatus('online')
    } catch {
      setStatus('offline')
    }
  }, [])

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30_000)
    return () => clearInterval(interval)
  }, [checkHealth])

  const handleServerUrlChange = (url) => {
    _setServerUrl(url)
    setServerUrl(url)
    checkHealth()
  }

  const { title, sub } = PAGE_TITLES[page]

  return (
    <div className="app">
      {/* ── Sidebar ─────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">🧪</div>
          <div>
            <div className="sidebar-logo-text">RAG Lab</div>
            <div className="sidebar-logo-sub">Learning Platform</div>
          </div>
        </div>

        {NAV.map(item => (
          <div
            key={item.id}
            id={`nav-${item.id}`}
            className={`nav-item ${page === item.id ? 'active' : ''}`}
            onClick={() => setPage(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </div>
        ))}

        <div className="sidebar-footer">
          <div className="server-status">
            <div className={`status-dot ${serverStatus}`} />
            <div>
              <div style={{ fontSize: 11.5, fontWeight: 500 }}>
                {serverStatus === 'online' ? 'Server online' : serverStatus === 'offline' ? 'Server offline' : 'Connecting…'}
              </div>
              <div style={{ fontSize: 10.5, marginTop: 1, fontFamily: 'var(--font-mono)', wordBreak: 'break-all' }}>
                {serverUrl}
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main ─────────────────────────────────────────── */}
      <div className="main">
        {/* Top bar */}
        <div className="topbar">
          <div>
            <div className="topbar-title">{title}</div>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 1 }}>{sub}</div>
          </div>

          {/* Server URL control */}
          <div className="server-url-input">
            <label htmlFor="server-url-field">Server:</label>
            <input
              id="server-url-field"
              type="text"
              value={serverUrl}
              onChange={e => handleServerUrlChange(e.target.value)}
              placeholder="http://server-ip:8000"
            />
            <button
              id="server-reconnect"
              className="btn btn-ghost"
              style={{ padding: '5px 10px', fontSize: 12 }}
              onClick={checkHealth}
            >
              ↺
            </button>
          </div>
        </div>

        {/* Page content */}
        {page === 'chat'      && <ChatPage />}
        {page === 'documents' && <DocumentsPage />}
        {page === 'dashboard' && <DashboardPage />}
      </div>
    </div>
  )
}
