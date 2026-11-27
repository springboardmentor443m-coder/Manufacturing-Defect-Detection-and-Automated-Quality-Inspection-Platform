import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Camera, Database, Layers, Search, UploadCloud, X } from 'lucide-react'
import { api } from '../api'
import { can, useAuth } from '../auth.jsx'
import { EmptyState, ErrorNote, PageHeader, Skeleton, SourceBadge, fmtDate } from '../components/ui.jsx'
import { QualityBar } from './Dashboard.jsx'

const SOURCES = [['', 'All', Layers], ['upload', 'Upload', UploadCloud], ['camera', 'Camera', Camera], ['dataset', 'Dataset', Database]]

export default function Batches() {
  const { user } = useAuth()
  const [params, setParams] = useSearchParams()
  const [batches, setBatches] = useState(null)
  const [error, setError] = useState('')
  const source = params.get('source') || ''
  const category = params.get('category') || ''
  const [q, setQ] = useState('')

  const set = (k, v) => { const p = new URLSearchParams(params); v ? p.set(k, v) : p.delete(k); setParams(p, { replace: true }) }

  useEffect(() => {
    setBatches(null)
    const p = { limit: 200 }
    if (source) p.source = source
    if (category) p.category = category
    api.batches(p).then(setBatches).catch((e) => setError(e.message))
  }, [source, category])

  const shown = useMemo(() => (batches || []).filter((b) => !q || `${b.name} ${b.product_category} ${b.created_by_name}`.toLowerCase().includes(q.toLowerCase())), [batches, q])

  return (
    <>
      <PageHeader eyebrow="Monitor" title="Inspection batches" desc="Every group of images that has come in, whether from an upload, the line camera or the dataset."
        actions={can.inspect(user) && <Link to="/app/upload" className="btn-primary"><UploadCloud size={15} />New batch</Link>} />

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex rounded-xl border border-ink-700 bg-ink-900 p-1">
          {SOURCES.map(([v, label, Icon]) => (
            <button key={label} onClick={() => set('source', v)}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition ${source === v ? 'bg-ink-700 text-slate-50' : 'text-slate-400 hover:text-slate-200'}`}>
              <Icon size={14} />{label}
            </button>
          ))}
        </div>
        <div className="relative w-full sm:w-64">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input className="input pl-9 py-2" placeholder="Search batches…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        {category && (
          <button onClick={() => set('category', '')} className="chip bg-signal/10 text-signal ring-signal/25 py-1">category: {category}<X size={12} /></button>
        )}
      </div>

      <ErrorNote error={error} />
      <div className="panel overflow-hidden animate-fadeUp">
        {!batches ? (
          <div className="p-5 space-y-3">{[0, 1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-12" />)}</div>
        ) : shown.length === 0 ? (
          <EmptyState icon={Layers} title="No batches match" desc="Try another filter, or start a new inspection." />
        ) : (
          <div className="overflow-x-auto">
            <table className="table-base min-w-[760px]">
              <thead><tr><th className="pl-5">Batch</th><th>Category</th><th>Source</th><th>Quality</th><th>Status</th><th>Created</th></tr></thead>
              <tbody>
                {shown.map((b) => (
                  <tr key={b.id} className="group hover:bg-ink-800/50 transition-colors">
                    <td className="pl-5"><Link to={`/app/batches/${b.id}`} className="font-medium text-slate-100 group-hover:text-signal transition-colors">{b.name}</Link><div className="text-xs text-slate-500">#{b.id} · {b.created_by_name}</div></td>
                    <td><span className="font-mono text-xs text-slate-300">{b.product_category}</span></td>
                    <td><SourceBadge source={b.source} /></td>
                    <td><QualityBar b={b} wide /></td>
                    <td><span className="text-xs capitalize text-slate-400">{b.status}</span></td>
                    <td className="whitespace-nowrap text-xs text-slate-500">{fmtDate(b.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  )
}
