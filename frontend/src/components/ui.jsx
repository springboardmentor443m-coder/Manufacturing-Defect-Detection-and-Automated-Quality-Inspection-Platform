import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react'
import { api } from '../api'

// ---------- status chips ----------
const VALIDATION = {
  valid: { cls: 'bg-pass/10 text-pass ring-pass/25', Icon: CheckCircle2 },
  warning: { cls: 'bg-warn/10 text-warn ring-warn/25', Icon: AlertTriangle },
  rejected: { cls: 'bg-fail/10 text-fail ring-fail/25', Icon: XCircle },
}

export function ValidationBadge({ status }) {
  const v = VALIDATION[status] || VALIDATION.valid
  return <span className={`chip ${v.cls}`}><v.Icon size={11} />{status}</span>
}

export function LabelBadge({ label }) {
  if (!label) return null
  return label === 'good'
    ? <span className="chip bg-sky-400/10 text-sky-300 ring-sky-400/20">good</span>
    : <span className="chip bg-fuchsia-400/10 text-fuchsia-300 ring-fuchsia-400/25">{label.replaceAll('_', ' ')}</span>
}

export function SourceBadge({ source }) {
  const map = {
    upload: 'bg-indigo-400/10 text-indigo-300 ring-indigo-400/20',
    camera: 'bg-emerald-400/10 text-emerald-300 ring-emerald-400/20',
    dataset: 'bg-amber-400/10 text-amber-300 ring-amber-400/20',
  }
  return <span className={`chip capitalize ${map[source]}`}>{source}</span>
}

// ---------- numbers ----------
export function useCountUp(target, ms = 900) {
  const [v, setV] = useState(0)
  useEffect(() => {
    if (typeof target !== 'number') return
    let raf
    const start = performance.now()
    const tick = (t) => {
      const p = Math.min(1, (t - start) / ms)
      setV(Math.round(target * (1 - Math.pow(1 - p, 3))))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, ms])
  return v
}

export function StatCard({ label, value, hint, icon: Icon, tone = 'signal', spark, delay = 0 }) {
  const n = useCountUp(value)
  const tones = { signal: 'text-signal bg-signal/10', pass: 'text-pass bg-pass/10', warn: 'text-warn bg-warn/10', fail: 'text-fail bg-fail/10' }
  return (
    <div className="panel-pad animate-fadeUp group hover:border-ink-600 transition-colors" style={{ animationDelay: `${delay}ms` }}>
      <div className="flex items-start justify-between">
        <div className="text-xs font-medium uppercase tracking-wider text-slate-500">{label}</div>
        {Icon && <div className={`rounded-lg p-1.5 ${tones[tone]}`}><Icon size={15} /></div>}
      </div>
      <div className="mt-3 flex items-end justify-between gap-3">
        <div className="text-3xl font-semibold text-slate-50 tabular-nums">{n.toLocaleString()}</div>
        {spark && <Sparkline data={spark} />}
      </div>
      {hint && <div className="mt-1.5 text-xs text-slate-500">{hint}</div>}
    </div>
  )
}

export function Sparkline({ data, w = 88, h = 28 }) {
  const max = Math.max(1, ...data)
  const pts = data.map((d, i) => `${(i / (data.length - 1 || 1)) * w},${h - (d / max) * (h - 2) - 1}`).join(' ')
  return (
    <svg width={w} height={h} className="overflow-visible">
      <defs>
        <linearGradient id="spark" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#22D3EE" stopOpacity=".35" /><stop offset="100%" stopColor="#22D3EE" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${h} ${pts} ${w},${h}`} fill="url(#spark)" />
      <polyline points={pts} fill="none" stroke="#22D3EE" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  )
}

// ---------- layout helpers ----------
export function PageHeader({ eyebrow, title, desc, actions }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 mb-7 animate-fadeUp">
      <div>
        {eyebrow && <div className="text-[11px] font-mono uppercase tracking-[.2em] text-signal mb-1.5">{eyebrow}</div>}
        <h1 className="text-2xl font-semibold">{title}</h1>
        {desc && <p className="mt-1 text-sm text-slate-400 max-w-2xl">{desc}</p>}
      </div>
      {actions && <div className="flex gap-2">{actions}</div>}
    </div>
  )
}

export function EmptyState({ icon: Icon, title, desc, action }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-14 px-6">
      {Icon && <div className="mb-4 rounded-2xl bg-ink-800 p-3 text-slate-400 ring-1 ring-ink-600"><Icon size={22} /></div>}
      <div className="font-medium text-slate-200">{title}</div>
      {desc && <p className="mt-1 text-sm text-slate-500 max-w-sm">{desc}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}

export function ErrorNote({ error }) {
  if (!error) return null
  return (
    <div className="flex items-start gap-2 rounded-xl border border-fail/30 bg-fail/10 px-3.5 py-2.5 text-sm text-rose-200">
      <XCircle size={16} className="mt-0.5 shrink-0 text-fail" />{error}
    </div>
  )
}

export const Skeleton = ({ className = '' }) => <div className={`skeleton ${className}`} />

// ---------- protected images ----------
const blobCache = new Map() // path -> object URL, kept for the session so grids don't refetch

/** <img> for protected endpoints (fetches with the bearer token, lazy-loads when visible). */
export function AuthImage({ path, className = '', alt = '', style }) {
  const [src, setSrc] = useState(blobCache.get(path) || null)
  const [failed, setFailed] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    if (blobCache.has(path)) { setSrc(blobCache.get(path)); return }
    let alive = true
    const load = () => api.blob(path)
      .then((b) => { const u = URL.createObjectURL(b); blobCache.set(path, u); if (alive) setSrc(u) })
      .catch(() => alive && setFailed(true))
    const el = ref.current
    if (!el || !('IntersectionObserver' in window)) { load(); return () => { alive = false } }
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { io.disconnect(); load() } }, { rootMargin: '200px' })
    io.observe(el)
    return () => { alive = false; io.disconnect() }
  }, [path])

  if (failed) return <div ref={ref} className={`${className} bg-ink-800 grid place-items-center text-xs text-slate-500`}>unavailable</div>
  if (!src) return <div ref={ref} className={`${className} skeleton`} />
  return <img src={src} alt={alt} style={style} className={`${className} animate-[fadeUp_.3s_ease-out]`} draggable={false} />
}

// ---------- formatting ----------
export const fmtDate = (s) => new Date(s).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
export const fmtBytes = (n) => (n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`)
export function timeAgo(s) {
  const d = (Date.now() - new Date(s).getTime()) / 1000
  if (d < 60) return 'just now'
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`
  return `${Math.floor(d / 86400)}d ago`
}
export const initials = (name = '') => name.split(/\s+/).map((p) => p[0]).slice(0, 2).join('').toUpperCase()
