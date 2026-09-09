import { useState, useEffect, useRef, useCallback } from 'react'
import { runBenchmark, benchmarkStatus, benchmarkList, benchmarkResults } from '../api.js'

const DEFAULT_CONFIG = {
  query_transformer: 'passthrough',
  retriever: 'dense',
  reranker: 'none',
  context_processor: 'passthrough',
  top_k: 5,
  multi_query_n: 3,
  window_size: 2,
}

const STRATEGY_OPTIONS = {
  query_transformer: ['passthrough', 'multi_query', 'hyde', 'step_back'],
  retriever: ['dense', 'sparse', 'hybrid'],
  reranker: ['none', 'bge'],
  context_processor: ['passthrough', 'window_expand', 'compress', 'rse', 'contextual_headers'],
}

const JUDGE_METRICS = ['answer_correctness', 'faithfulness', 'context_relevance', 'answer_relevance']

// ── Progress bar ────────────────────────────────────────────────────────────
function ProgressBar({ value, max, color = 'var(--accent)' }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
        <span style={{ color: 'var(--text-muted)' }}>Progress</span>
        <span style={{ color, fontFamily: 'var(--font-mono)' }}>{value} / {max} ({pct}%)</span>
      </div>
      <div className="score-bar-bg" style={{ height: 8 }}>
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}

// ── Live metrics display ────────────────────────────────────────────────────
function LiveMetrics({ metrics }) {
  if (!metrics || Object.keys(metrics).length === 0) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
      {Object.entries(metrics).map(([k, v]) => (
        <div key={k} style={{
          background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)',
          padding: '8px 12px', border: '1px solid var(--border)',
        }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 2 }}>
            {k.replace(/_/g, ' ')}
          </div>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-light)', fontFamily: 'var(--font-mono)' }}>
            {typeof v === 'number' ? v.toFixed(3) : v}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Score chip ──────────────────────────────────────────────────────────────────
function ScoreChip({ value }) {
  if (value == null) return <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>—</span>
  const color = value >= 0.7 ? 'var(--green)' : value >= 0.4 ? 'var(--amber)' : 'var(--red)'
  return (
    <span style={{
      fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 700, color,
      background: value >= 0.7 ? 'var(--green-dim)' : value >= 0.4 ? 'var(--amber-dim)' : 'var(--red-dim)',
      padding: '2px 7px', borderRadius: 99, whiteSpace: 'nowrap',
    }}>
      {value.toFixed(2)}
    </span>
  )
}

// ── Per-sample row ────────────────────────────────────────────────────────────────
function SampleRow({ sample, index, hasJudge }) {
  const [expanded, setExpanded] = useState(false)
  const correctness = sample.answer_correctness
  const isGood = correctness == null ? null : correctness >= 0.7

  return (
    <>
      <tr
        onClick={() => setExpanded(v => !v)}
        style={{ cursor: 'pointer', transition: 'background 120ms' }}
        className="sample-row"
      >
        <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)', width: 36, textAlign: 'center' }}>
          {index + 1}
        </td>
        <td style={{ width: 28, textAlign: 'center' }}>
          {correctness == null
            ? <span title="No judge">·</span>
            : isGood
              ? <span title="Correct" style={{ color: 'var(--green)' }}>✓</span>
              : <span title="Wrong" style={{ color: 'var(--red)' }}>✗</span>
          }
        </td>
        <td style={{ fontSize: 12.5, color: 'var(--text-primary)', maxWidth: 320 }}>
          <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {sample.question}
          </div>
        </td>
        {hasJudge && JUDGE_METRICS.map(m => (
          <td key={m} style={{ textAlign: 'center', width: 80 }}>
            <ScoreChip value={sample[m]} />
          </td>
        ))}
        <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', textAlign: 'right', whiteSpace: 'nowrap' }}>
          {sample.retrieval_time != null ? `${(sample.retrieval_time * 1000).toFixed(0)}ms` : '—'}
        </td>
        <td style={{ width: 24, textAlign: 'center', color: 'var(--text-muted)', fontSize: 11 }}>
          {expanded ? '▲' : '▼'}
        </td>
      </tr>

      {expanded && (
        <tr>
          <td colSpan={hasJudge ? 6 + JUDGE_METRICS.length : 6}
            style={{ padding: 0, background: 'var(--bg-elevated)' }}>
            <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
              {sample.ground_truth && (
                <div>
                  <div style={{ fontSize: 11, color: 'var(--green)', fontWeight: 600, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>✓ Ground Truth</div>
                  <div style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--text-primary)' }}>{sample.ground_truth}</div>
                </div>
              )}
              <div>
                <div style={{ fontSize: 11, color: correctness == null ? 'var(--text-muted)' : isGood ? 'var(--accent-light)' : 'var(--amber)', fontWeight: 600, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>🤖 Model Answer</div>
                <div style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--text-secondary)' }}>
                  {sample.answer || <em style={{ color: 'var(--text-muted)' }}>No answer generated</em>}
                </div>
              </div>
              {hasJudge && JUDGE_METRICS.some(m => sample[m] != null) && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                  {JUDGE_METRICS.filter(m => sample[m] != null).map(m => (
                    <div key={m}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
                        <span style={{ color: 'var(--text-muted)' }}>{m.replace(/_/g, ' ')}</span>
                        <ScoreChip value={sample[m]} />
                      </div>
                      <div className="score-bar-bg">
                        <div className="score-bar-fill" style={{
                          width: `${(sample[m] * 100).toFixed(0)}%`,
                          background: sample[m] >= 0.7 ? 'var(--green)' : sample[m] >= 0.4 ? 'var(--amber)' : 'var(--red)',
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <div style={{ display: 'flex', gap: 16, fontSize: 11.5, color: 'var(--text-muted)' }}>
                {sample.retrieval_time != null && <span>retrieval: {(sample.retrieval_time * 1000).toFixed(0)}ms</span>}
                {sample.generation_time != null && <span>generation: {(sample.generation_time * 1000).toFixed(0)}ms</span>}
                {sample.retrieved_ids?.length > 0 && <span>{sample.retrieved_ids.length} chunks retrieved</span>}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

// ── Per-sample table ───────────────────────────────────────────────────────────────
function SampleTable({ samples }) {
  const [filter, setFilter] = useState('all')
  if (!samples || samples.length === 0) return null

  const hasJudge = samples.some(s => s.answer_correctness != null)

  const filtered = samples.filter(s => {
    if (filter === 'correct') return (s.answer_correctness ?? 0) >= 0.7
    if (filter === 'wrong')   return s.answer_correctness != null && s.answer_correctness < 0.7
    if (filter === 'nojudge') return s.answer_correctness == null
    return true
  })

  const correctCount = samples.filter(s => (s.answer_correctness ?? 0) >= 0.7).length
  const wrongCount   = samples.filter(s => s.answer_correctness != null && s.answer_correctness < 0.7).length

  return (
    <div className="card" style={{ marginTop: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
        <div className="card-title" style={{ margin: 0 }}>Per-Sample Results</div>
        <span className="badge badge-green">{correctCount} correct</span>
        <span className="badge badge-red">{wrongCount} wrong</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          {['all', 'correct', 'wrong', 'nojudge'].map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={`btn btn-ghost`}
              style={{
                padding: '3px 10px', fontSize: 11,
                background: filter === f ? 'var(--accent-dim)' : '',
                color: filter === f ? 'var(--accent-light)' : '',
              }}>
              {f}
            </button>
          ))}
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th style={{ width: 36, padding: '6px 8px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 11 }}>#</th>
              <th style={{ width: 28 }}></th>
              <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 11 }}>QUESTION</th>
              {hasJudge && JUDGE_METRICS.map(m => (
                <th key={m} style={{ width: 80, textAlign: 'center', padding: '6px 4px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.3px' }}>
                  {m.replace('answer_', '').replace(/_/g, ' ')}
                </th>
              ))}
              <th style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 11 }}>RTR</th>
              <th style={{ width: 24 }}></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((s, i) => (
              <SampleRow key={s.sample_idx ?? i} sample={s} index={s.sample_idx ?? i} hasJudge={hasJudge} />
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
        Hiển thị {filtered.length}/{samples.length} samples — click vào hàng để xem chi tiết
      </div>
    </div>
  )
}

// ── Past experiments list ───────────────────────────────────────────────────
function ExperimentsList({ onLoad }) {
  const [experiments, setExperiments] = useState([])
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const data = await benchmarkList()
      setExperiments(data.experiments || [])
    } catch { }
    setLoading(false)
  }, [])

  useEffect(() => { refresh() }, [refresh])

  if (experiments.length === 0 && !loading) return (
    <div className="empty-state" style={{ padding: '24px 0' }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Chưa có experiments nào.</div>
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: 11 }} onClick={refresh}>
          ↻ Refresh
        </button>
      </div>
      {experiments.map(exp => (
        <div key={exp.name} style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '10px 14px', background: 'var(--bg-elevated)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
        }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12.5, flex: 1 }}>{exp.name}</span>
          {exp.has_metrics
            ? <span className="badge badge-green">metrics ✓</span>
            : <span className="badge badge-amber">no metrics</span>
          }
          {exp.has_metrics && (
            <button className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: 11 }}
              onClick={() => onLoad(exp.name)}>
              Load →
            </button>
          )}
        </div>
      ))}
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function BenchmarkPage() {
  const [config, setConfig]           = useState({ ...DEFAULT_CONFIG })
  const [expName, setExpName]         = useState('my_experiment')
  const [runJudge, setRunJudge]       = useState(true)
  const [status, setStatus]           = useState('idle')
  const [progress, setProgress]       = useState(0)
  const [total, setTotal]             = useState(0)
  const [currentMetrics, setCurrentMetrics] = useState({})
  const [finalMetrics, setFinalMetrics]     = useState(null)
  const [perSample, setPerSample]           = useState([])   // ← NEW
  const [error, setError]             = useState(null)
  const [log, setLog]                 = useState([])
  const [resultPath, setResultPath]   = useState(null)
  const logRef = useRef(null)

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [log])

  const addLog = (msg) => setLog(prev => [...prev.slice(-80), msg])

  const handleRun = async () => {
    if (status === 'running') return
    if (!expName.trim()) return

    setStatus('running')
    setProgress(0); setTotal(0)
    setCurrentMetrics({}); setFinalMetrics(null); setPerSample([])
    setError(null); setLog([])
    setResultPath(null)

    addLog(`⏳ Starting experiment: ${expName}`)

    try {
      const response = await runBenchmark({
        experiment_name: expName,
        config,
        run_judge: runJudge,
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || `HTTP ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))
            if (event.type === 'progress') {
              setProgress(event.sample)
              setTotal(event.total)
              setCurrentMetrics(event.metrics || {})
              addLog(`[${event.sample}/${event.total}] ${event.question}`)
            } else if (event.type === 'done') {
              setStatus('done')
              setFinalMetrics(event.metrics)
              setResultPath(event.result_path)
              addLog(`✅ Done! Saved to: ${event.result_path}`)
              // Load per-sample data from saved file
              if (event.result_path) {
                try {
                  const expDir = expName
                  const fullData = await benchmarkResults(expDir)
                  setPerSample(fullData.per_sample || [])
                } catch { /* non-critical */ }
              }
            } else if (event.type === 'error') {
              setStatus('error')
              setError(event.message)
              addLog(`❌ Error: ${event.message}`)
            }
          } catch { /* skip malformed events */ }
        }
      }
    } catch (e) {
      setStatus('error')
      setError(e.message)
      addLog(`❌ ${e.message}`)
    }
  }

  const handleLoadResult = async (name) => {
    try {
      const data = await benchmarkResults(name)
      setExpName(name)
      setFinalMetrics(data.generation_metrics || {})
      setPerSample(data.per_sample || [])           // ← load per_sample
      setTotal(data.num_samples || 0)
      setProgress(data.num_samples || 0)
      setStatus('done')
      setLog([`Loaded results for "${name}" (${data.num_samples} samples)`])
    } catch (e) {
      setError(e.message)
    }
  }

  const ConfigSelect = ({ label, field }) => (
    <div>
      <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <select value={config[field]}
        onChange={e => setConfig(c => ({ ...c, [field]: e.target.value }))}
        style={{ fontSize: 12.5 }}>
        {STRATEGY_OPTIONS[field].map(v => <option key={v} value={v}>{v}</option>)}
      </select>
    </div>
  )

  return (
    <div className="page">
      <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: 20, maxWidth: 1100 }}>

        {/* Left: Config + Past experiments */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Config card */}
          <div className="card">
            <div className="card-title">Pipeline Config</div>

            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>Experiment name</div>
              <input id="bench-exp-name" type="text" value={expName}
                onChange={e => setExpName(e.target.value)}
                placeholder="my_experiment"
                style={{ fontSize: 13 }} />
            </div>

            <ConfigSelect label="Query Transformer" field="query_transformer" />
            <div style={{ height: 8 }} />
            <ConfigSelect label="Retriever" field="retriever" />
            <div style={{ height: 8 }} />
            <ConfigSelect label="Reranker" field="reranker" />
            <div style={{ height: 8 }} />
            <ConfigSelect label="Context Processor" field="context_processor" />

            <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>Top-K</div>
                <input type="number" min={1} max={20} value={config.top_k}
                  onChange={e => setConfig(c => ({ ...c, top_k: Number(e.target.value) }))}
                  style={{ fontSize: 12 }} />
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 12 }}>
              <label className="toggle">
                <input type="checkbox" checked={runJudge}
                  onChange={e => setRunJudge(e.target.checked)} />
                <span className="toggle-slider" />
              </label>
              <span style={{ fontSize: 12.5 }}>LLM-as-Judge metrics</span>
            </div>

            <div style={{ marginTop: 16 }}>
              <button
                id="bench-run"
                className="btn btn-primary"
                style={{ width: '100%' }}
                onClick={handleRun}
                disabled={status === 'running' || !expName.trim()}
              >
                {status === 'running'
                  ? <><div className="spinner" style={{ width: 15, height: 15, borderWidth: 2 }} /> Running…</>
                  : '▶ Run Benchmark'
                }
              </button>
            </div>
          </div>

          {/* Past experiments */}
          <div className="card">
            <div className="card-title">Past Experiments</div>
            <ExperimentsList onLoad={handleLoadResult} />
          </div>
        </div>

        {/* Right: Live progress + results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Status */}
          {status !== 'idle' && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div style={{ fontWeight: 600, fontSize: 14 }}>{expName}</div>
                <span className={`badge ${status === 'done' ? 'badge-green' : status === 'error' ? 'badge-red' : 'badge-accent'}`}>
                  {status}
                </span>
              </div>

              {total > 0 && <ProgressBar value={progress} max={total} />}

              {Object.keys(currentMetrics).length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <div className="card-title">Live Metrics</div>
                  <LiveMetrics metrics={currentMetrics} />
                </div>
              )}
            </div>
          )}

          {/* Final metrics */}
          {finalMetrics && Object.keys(finalMetrics).length > 0 && (
            <div className="card">
              <div className="card-title">Final Metrics</div>
              <LiveMetrics metrics={finalMetrics} />
              {resultPath && (
                <div style={{ marginTop: 12, fontSize: 12, color: 'var(--text-muted)' }}>
                  Saved: <code style={{ fontFamily: 'var(--font-mono)' }}>{resultPath}</code>
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {error && <div className="alert alert-error">⚠️ {error}</div>}

          {/* Log */}
          {log.length > 0 && (
            <div className="card">
              <div className="card-title">Log</div>
              <div ref={logRef} style={{
                fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: 1.6,
                maxHeight: 280, overflowY: 'auto', color: 'var(--text-secondary)',
              }}>
                {log.map((entry, i) => (
                  <div key={i}>{entry}</div>
                ))}
              </div>
            </div>
          )}

          {/* Idle state */}
          {status === 'idle' && (
            <div className="empty-state">
              <div className="empty-state-icon">🏃</div>
              <div className="empty-state-text">
                Cấu hình pipeline, đặt tên experiment và nhấn Run Benchmark.<br/>
                Kết quả sẽ được stream realtime và lưu vào <code>experiments/results/</code>.
              </div>
            </div>
          )}

          {/* Per-sample table */}
          {perSample.length > 0 && <SampleTable samples={perSample} />}
        </div>
      </div>
    </div>
  )
}
