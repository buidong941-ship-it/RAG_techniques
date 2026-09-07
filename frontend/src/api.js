/**
 * api.js — RAG Backend API client
 * All calls go through the Vite proxy (/api → server URL)
 */

let _serverUrl = localStorage.getItem('rag_server_url') || 'http://localhost:8000'

export function setServerUrl(url) {
  _serverUrl = url.replace(/\/$/, '')
  localStorage.setItem('rag_server_url', _serverUrl)
}

export function getServerUrl() { return _serverUrl }

async function request(method, path, body = null) {
  const url = `${_serverUrl}${path}`
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body) opts.body = JSON.stringify(body)
  const res = await fetch(url, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

// ── Health ─────────────────────────────────────────────────────
export const health = () => request('GET', '/health')

// ── RAG Query ──────────────────────────────────────────────────
export const query = (payload) => request('POST', '/query', payload)

// ── Retrieve only ──────────────────────────────────────────────
export const retrieve = (payload) => request('POST', '/retrieve', payload)

// ── Ingest (base64) ────────────────────────────────────────────
export async function ingestFile(file, chunkSize, chunkOverlap) {
  const arrayBuffer = await file.arrayBuffer()
  const bytes = new Uint8Array(arrayBuffer)
  let binary = ''
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i])
  const content_b64 = btoa(binary)
  return request('POST', '/ingest', {
    filename: file.name,
    content_b64,
    chunk_size: chunkSize || undefined,
    chunk_overlap: chunkOverlap || undefined,
  })
}

// ── Index info ─────────────────────────────────────────────────
export const indexInfo = () => request('GET', '/index/info')

// ── Clear index ────────────────────────────────────────────────
export const clearIndex = () => request('DELETE', '/index')

// ── Config update ──────────────────────────────────────────────
export const updateConfig = (cfg) => request('POST', '/config', cfg)

// ── Pipeline ───────────────────────────────────────────────────
export const getPresets = () => request('GET', '/pipeline/presets')

export const pipelineQuery = (query, config) =>
  request('POST', '/pipeline/query', { query, config })

export const pipelineCompare = (query, configs) =>
  request('POST', '/pipeline/compare', { query, configs })

// ── Benchmark ──────────────────────────────────────────────────
export const benchmarkStatus = () => request('GET', '/benchmark/status')
export const benchmarkList   = () => request('GET', '/benchmark/list')
export const benchmarkResults = (name) => request('GET', `/benchmark/results/${name}`)

/**
 * Start a benchmark run and return an EventSource for SSE streaming.
 * The caller should attach .onmessage / .onerror handlers.
 */
export function runBenchmark(payload) {
  // SSE via fetch so we can POST with body
  const url = `${getServerUrl()}/benchmark/run`
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}
