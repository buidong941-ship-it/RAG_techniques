import { useState } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, Cell,
} from 'recharts'

/* ─── Colour pool for experiments ────────────────────────────── */
const COLORS = [
  '#7c6af7', '#34d399', '#fbbf24', '#60a5fa',
  '#f87171', '#a78bfa', '#6ee7b7', '#fcd34d',
]

/* ─── Metric display names ────────────────────────────────────── */
const RETRIEVAL_KEYS = ['recall@5', 'precision@5', 'mrr', 'ndcg@5']
const GEN_KEYS       = ['answer_correctness', 'faithfulness', 'context_relevance', 'answer_relevance']

/* ─── File drop / load ────────────────────────────────────────── */
function loadJson(file) {
  return new Promise((res, rej) => {
    const reader = new FileReader()
    reader.onload = e => { try { res(JSON.parse(e.target.result)) } catch { rej(new Error('Invalid JSON')) } }
    reader.onerror = () => rej(new Error('Read error'))
    reader.readAsText(file)
  })
}

/* ─── Metric value cell with delta ───────────────────────────── */
function MetricCell({ value, baseline }) {
  if (value == null || isNaN(value)) return <td style={{ color: 'var(--text-muted)' }}>—</td>
  const delta = baseline != null ? value - baseline : null
  return (
    <td>
      <span style={{ fontFamily: 'var(--font-mono)' }}>{value.toFixed(4)}</span>
      {delta != null && delta !== 0 && (
        <span className={delta > 0 ? 'delta-positive' : 'delta-negative'}>
          {delta > 0 ? '+' : ''}{delta.toFixed(3)}
        </span>
      )}
    </td>
  )
}

/* ─── Section tabs ────────────────────────────────────────────── */
function Tabs({ active, onChange }) {
  const tabs = [
    { id: 'retrieval',   label: '📡 Retrieval' },
    { id: 'generation',  label: '🤖 Generation' },
    { id: 'system',      label: '⏱ System' },
    { id: 'failure',     label: '⚠️ Failure' },
    { id: 'radar',       label: '🕸 Radar' },
  ]
  return (
    <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
      {tabs.map(t => (
        <button
          key={t.id}
          className={`btn ${active === t.id ? 'btn-primary' : 'btn-ghost'}`}
          style={{ padding: '6px 14px', fontSize: 12.5 }}
          onClick={() => onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

/* ─── Retrieval metrics table ─────────────────────────────────── */
function RetrievalTable({ experiments }) {
  const baseline = experiments.find(e => e.experiment === 'baseline')
  const bm = baseline?.retrieval_metrics || {}

  return (
    <div className="card">
      <div className="card-title">Retrieval Metrics</div>
      <table className="metric-table">
        <thead>
          <tr>
            <th>Experiment</th>
            {RETRIEVAL_KEYS.map(k => <th key={k}>{k}</th>)}
          </tr>
        </thead>
        <tbody>
          {experiments.map((exp, i) => (
            <tr key={exp.experiment} className={exp.experiment === 'baseline' ? 'baseline-row' : ''}>
              <td className="exp-name-col">
                <span style={{
                  display: 'inline-block', width: 8, height: 8,
                  borderRadius: '50%', background: COLORS[i % COLORS.length],
                  marginRight: 8,
                }} />
                {exp.experiment}
              </td>
              {RETRIEVAL_KEYS.map(k => (
                <MetricCell
                  key={k}
                  value={exp.retrieval_metrics?.[k]}
                  baseline={exp.experiment !== 'baseline' ? bm[k] : null}
                />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ─── Generation metrics table ────────────────────────────────── */
function GenerationTable({ experiments }) {
  const baseline = experiments.find(e => e.experiment === 'baseline')
  const bm = baseline?.generation_metrics || {}

  const hasGen = experiments.some(e => Object.keys(e.generation_metrics || {}).length > 0)
  if (!hasGen) return (
    <div className="empty-state">
      <div className="empty-state-icon">🤖</div>
      <div className="empty-state-text">Chưa có generation metrics. Chạy evaluation với --judge flag.</div>
    </div>
  )

  return (
    <div className="card">
      <div className="card-title">Generation Metrics (LLM-as-judge, 0–1)</div>
      <table className="metric-table">
        <thead>
          <tr>
            <th>Experiment</th>
            {GEN_KEYS.map(k => <th key={k}>{k.replace(/_/g, ' ')}</th>)}
          </tr>
        </thead>
        <tbody>
          {experiments.map((exp, i) => (
            <tr key={exp.experiment} className={exp.experiment === 'baseline' ? 'baseline-row' : ''}>
              <td className="exp-name-col">
                <span style={{
                  display: 'inline-block', width: 8, height: 8,
                  borderRadius: '50%', background: COLORS[i % COLORS.length],
                  marginRight: 8,
                }} />
                {exp.experiment}
              </td>
              {GEN_KEYS.map(k => (
                <MetricCell
                  key={k}
                  value={exp.generation_metrics?.[k]}
                  baseline={exp.experiment !== 'baseline' ? bm[k] : null}
                />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ─── System metrics table ────────────────────────────────────── */
function SystemTable({ experiments }) {
  return (
    <div className="card">
      <div className="card-title">System Metrics — Mean Latency (seconds)</div>
      <table className="metric-table">
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Retrieval</th>
            <th>Generation</th>
            <th>Total</th>
            <th>Chunks</th>
          </tr>
        </thead>
        <tbody>
          {experiments.map((exp, i) => {
            const s = exp.system_metrics || {}
            return (
              <tr key={exp.experiment} className={exp.experiment === 'baseline' ? 'baseline-row' : ''}>
                <td className="exp-name-col">
                  <span style={{
                    display: 'inline-block', width: 8, height: 8,
                    borderRadius: '50%', background: COLORS[i % COLORS.length],
                    marginRight: 8,
                  }} />
                  {exp.experiment}
                </td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{s.retrieval_time_s?.mean?.toFixed(3) ?? '—'}</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{s.generation_time_s?.mean?.toFixed(3) ?? '—'}</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{s.total_latency_s?.mean?.toFixed(3) ?? '—'}</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{s.num_retrieved_chunks?.mean?.toFixed(1) ?? '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/* ─── Failure analysis bar chart ──────────────────────────────── */
function FailureChart({ experiments }) {
  const FAIL_MODES = ['ok', 'retrieval_failure', 'generation_failure', 'both_failure']
  const FAIL_COLORS = {
    ok: '#34d399',
    retrieval_failure: '#f87171',
    generation_failure: '#fbbf24',
    both_failure: '#a78bfa',
  }

  const data = experiments.map(exp => {
    const fa = exp.failure_analysis || {}
    const total = exp.num_samples || 1
    return {
      name: exp.experiment,
      ...Object.fromEntries(FAIL_MODES.map(m => [m, +(((fa[m] || 0) / total) * 100).toFixed(1)])),
    }
  })

  return (
    <div className="card">
      <div className="card-title">Failure Analysis (% of samples)</div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
          <YAxis unit="%" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
          <Tooltip
            contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text-primary)' }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
          {FAIL_MODES.map(m => (
            <Bar key={m} dataKey={m} stackId="a" fill={FAIL_COLORS[m]} name={m.replace(/_/g, ' ')} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ─── Radar chart ─────────────────────────────────────────────── */
function RadarView({ experiments }) {
  const RADAR_KEYS = ['recall@5', 'precision@5', 'mrr', 'ndcg@5']

  const radarData = RADAR_KEYS.map(key => {
    const obj = { metric: key.replace('@5', '') }
    experiments.forEach(exp => {
      obj[exp.experiment] = exp.retrieval_metrics?.[key] ?? 0
    })
    return obj
  })

  return (
    <div className="card">
      <div className="card-title">Retrieval Metrics — Radar Comparison</div>
      <ResponsiveContainer width="100%" height={320}>
        <RadarChart data={radarData}>
          <PolarGrid stroke="var(--border)" />
          <PolarAngleAxis dataKey="metric" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
          {experiments.map((exp, i) => (
            <Radar
              key={exp.experiment}
              name={exp.experiment}
              dataKey={exp.experiment}
              stroke={COLORS[i % COLORS.length]}
              fill={COLORS[i % COLORS.length]}
              fillOpacity={0.12}
              strokeWidth={2}
            />
          ))}
          <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
          <Tooltip
            contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ─── Main Dashboard page ─────────────────────────────────────── */
export default function DashboardPage() {
  const [experiments, setExperiments] = useState([])
  const [tab, setTab]                 = useState('retrieval')
  const [dragging, setDragging]       = useState(false)
  const [error, setError]             = useState(null)

  const handleFiles = async (files) => {
    setError(null)
    const loaded = []
    for (const f of files) {
      if (!f.name.endsWith('.json')) continue
      try {
        const data = await loadJson(f)
        if (!data.experiment) data.experiment = f.name.replace('.json', '')
        loaded.push(data)
      } catch (e) {
        setError(`${f.name}: ${e.message}`)
      }
    }
    if (loaded.length > 0) {
      // Put baseline first
      loaded.sort((a, b) => (a.experiment === 'baseline' ? -1 : b.experiment === 'baseline' ? 1 : 0))
      setExperiments(prev => {
        const names = new Set(loaded.map(e => e.experiment))
        return [...prev.filter(e => !names.has(e.experiment)), ...loaded]
      })
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFiles(Array.from(e.dataTransfer.files))
  }

  const removeExp = (name) => setExperiments(prev => prev.filter(e => e.experiment !== name))

  return (
    <div className="page">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1000 }}>

        {/* Load area */}
        <div
          className={`drop-zone ${dragging ? 'dragging' : ''}`}
          style={{ padding: '24px 20px' }}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => {
            const inp = document.createElement('input')
            inp.type = 'file'; inp.multiple = true; inp.accept = '.json'
            inp.onchange = e => handleFiles(Array.from(e.target.files))
            inp.click()
          }}
        >
          <div className="drop-zone-icon">📊</div>
          <div className="drop-zone-text">Kéo thả <code>metrics.json</code> từ <code>experiments/results/*/</code></div>
          <div className="drop-zone-sub">Hoặc click để chọn file — tải nhiều experiments cùng lúc</div>
        </div>

        {error && <div className="alert alert-error">⚠️ {error}</div>}

        {/* Loaded experiments */}
        {experiments.length > 0 && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Loaded:</span>
            {experiments.map((exp, i) => (
              <span key={exp.experiment} className="badge badge-accent" style={{ gap: 6 }}>
                <span style={{
                  width: 7, height: 7, borderRadius: '50%',
                  background: COLORS[i % COLORS.length], display: 'inline-block',
                }} />
                {exp.experiment}
                <span
                  style={{ cursor: 'pointer', marginLeft: 2, opacity: 0.6 }}
                  onClick={() => removeExp(exp.experiment)}
                >×</span>
              </span>
            ))}
          </div>
        )}

        {experiments.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">📈</div>
            <div className="empty-state-text">
              Tải file <code>metrics.json</code> từ thư mục <code>experiments/results/</code> để xem so sánh.
            </div>
          </div>
        )}

        {experiments.length > 0 && (
          <>
            <Tabs active={tab} onChange={setTab} />
            {tab === 'retrieval'  && <RetrievalTable experiments={experiments} />}
            {tab === 'generation' && <GenerationTable experiments={experiments} />}
            {tab === 'system'     && <SystemTable experiments={experiments} />}
            {tab === 'failure'    && <FailureChart experiments={experiments} />}
            {tab === 'radar'      && <RadarView experiments={experiments} />}
          </>
        )}
      </div>
    </div>
  )
}
