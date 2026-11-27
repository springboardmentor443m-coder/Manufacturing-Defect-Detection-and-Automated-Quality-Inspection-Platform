import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Database, Download, FolderOpen, Search } from 'lucide-react'
import { api } from '../api'
import { useToast } from '../components/Toast.jsx'
import { AuthImage, EmptyState, ErrorNote, PageHeader, Skeleton } from '../components/ui.jsx'

export default function Dataset() {
  const toast = useToast()
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(null)
  const [q, setQ] = useState('')

  useEffect(() => { api.dataset().then(setData).catch((e) => setError(e.message)) }, [])

  if (error) return <ErrorNote error={error} />
  if (!data) return <><PageHeader eyebrow="Acquire" title="MVTec AD dataset" /><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">{[...Array(10)].map((_, i) => <Skeleton key={i} className="h-64 rounded-2xl" />)}</div></>

  const cats = data.categories.filter((c) => c.name.includes(q.toLowerCase()))
  const totals = data.categories.reduce((a, c) => ({ train: a.train + c.train_good, good: a.good + c.test_good, bad: a.bad + c.test_defective, types: a.types + Object.keys(c.defect_types).length }), { train: 0, good: 0, bad: 0, types: 0 })

  return (
    <>
      <PageHeader eyebrow="Acquire" title="MVTec AD dataset"
        desc="The industry benchmark for unsupervised anomaly detection. Pick a category to browse its defect types and import labelled images." />

      {!data.available ? (
        <div className="panel"><EmptyState icon={FolderOpen} title="Dataset not found" desc={`Extract MVTec AD to ${data.root}, or run scripts/make_sample_dataset.py. See docs/dataset.md.`} /></div>
      ) : (
        <>
          <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
            {[['Categories', data.categories.length], ['Training (good)', totals.train], ['Test images', totals.good + totals.bad], ['Defect types', totals.types]].map(([k, v]) => (
              <div key={k} className="panel px-4 py-3"><div className="text-[11px] uppercase tracking-wider text-slate-500">{k}</div><div className="text-2xl font-semibold tabular-nums text-slate-50">{v.toLocaleString()}</div></div>
            ))}
          </div>

          <div className="relative mb-4 w-full sm:w-72">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input className="input pl-9 py-2" placeholder="Filter categories…" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {cats.map((c, i) => (
              <button key={c.name} onClick={() => setSelected(c)}
                className="group overflow-hidden rounded-2xl border border-ink-700 bg-ink-900 text-left transition hover:-translate-y-0.5 hover:border-signal/50 hover:shadow-glow animate-fadeUp"
                style={{ animationDelay: `${i * 30}ms` }}>
                <div className="relative grid aspect-[4/3] grid-cols-2 gap-px bg-ink-700">
                  <AuthImage path={`/dataset/${c.name}/preview?kind=good`} className="h-full w-full object-cover" alt="" />
                  <AuthImage path={`/dataset/${c.name}/preview?kind=defect`} className="h-full w-full object-cover" alt="" />
                  <span className="absolute bottom-2 left-2 chip bg-ink-950/80 text-sky-300 ring-sky-400/20">good</span>
                  <span className="absolute bottom-2 right-2 chip bg-ink-950/80 text-fuchsia-300 ring-fuchsia-400/25">defect</span>
                </div>
                <div className="p-4">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm font-medium text-slate-100">{c.name}</span>
                    <ArrowRight size={15} className="text-slate-600 group-hover:text-signal group-hover:translate-x-0.5 transition" />
                  </div>
                  <div className="mt-1 text-xs text-slate-500">{c.train_good + c.test_good + c.test_defective} images · {Object.keys(c.defect_types).length} defect types</div>
                </div>
              </button>
            ))}
          </div>
        </>
      )}

      {selected && <ImportDrawer category={selected} onClose={() => setSelected(null)} toast={toast} />}
    </>
  )
}

function ImportDrawer({ category: c, onClose, toast }) {
  const [split, setSplit] = useState('test')
  const available = split === 'test' ? c.test_good + c.test_defective : c.train_good
  const [limit, setLimit] = useState(Math.min(50, available))
  const [busy, setBusy] = useState(false)
  const [created, setCreated] = useState(null)
  const [error, setError] = useState('')
  const maxType = Math.max(1, ...Object.values(c.defect_types))

  useEffect(() => setLimit((l) => Math.min(Math.max(l, 1), available)), [available])
  useEffect(() => {
    const k = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k)
  }, [onClose])

  const run = async () => {
    setBusy(true); setError('')
    try {
      const b = await api.importDataset({ category: c.name, split, limit: Number(limit) })
      setCreated(b); toast('Dataset imported', { desc: `${b.image_count} ${c.name} images with ground-truth labels` })
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <aside className="relative flex h-full w-full max-w-md flex-col overflow-y-auto border-l border-ink-700 bg-ink-900 animate-[fadeUp_.25s_ease-out]">
        <div className="grid grid-cols-2 gap-px bg-ink-700">
          <AuthImage path={`/dataset/${c.name}/preview?kind=good`} className="aspect-square w-full object-cover" alt="" />
          <AuthImage path={`/dataset/${c.name}/preview?kind=defect`} className="aspect-square w-full object-cover" alt="" />
        </div>
        <div className="flex-1 p-6">
          <div className="text-[11px] font-mono uppercase tracking-[.2em] text-signal">Category</div>
          <h2 className="mt-1 font-mono text-2xl font-semibold">{c.name}</h2>
          <div className="mt-5 grid grid-cols-3 gap-2 text-center">
            {[['train', c.train_good, 'text-slate-200'], ['test good', c.test_good, 'text-sky-300'], ['defective', c.test_defective, 'text-fuchsia-300']].map(([k, v, cl]) => (
              <div key={k} className="rounded-xl bg-ink-800 py-2.5"><div className={`text-lg font-semibold ${cl}`}>{v}</div><div className="text-[10px] uppercase text-slate-500">{k}</div></div>
            ))}
          </div>

          <div className="mt-6 text-[11px] uppercase tracking-wider text-slate-500">Defect types</div>
          <ul className="mt-3 space-y-2">
            {Object.entries(c.defect_types).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
              <li key={k}>
                <div className="flex justify-between text-xs"><span className="text-slate-300">{k.replaceAll('_', ' ')}</span><span className="text-slate-500">{v}</span></div>
                <div className="mt-1 h-1 rounded-full bg-ink-700"><div className="h-full rounded-full bg-gradient-to-r from-fuchsia-500 to-fail" style={{ width: `${(v / maxType) * 100}%` }} /></div>
              </li>
            ))}
          </ul>

          <div className="mt-7 space-y-4 border-t border-ink-700 pt-6">
            <div>
              <div className="label">Split</div>
              <div className="grid grid-cols-2 gap-2">
                {[['test', 'Test', 'good + defective'], ['train', 'Train', 'good only']].map(([v, l, d]) => (
                  <button key={v} onClick={() => setSplit(v)} className={`rounded-xl border p-3 text-left transition ${split === v ? 'border-signal/60 bg-signal/5' : 'border-ink-600 hover:border-ink-500'}`}>
                    <div className="text-sm font-medium text-slate-100">{l}</div><div className="text-xs text-slate-500">{d}</div>
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div className="flex justify-between"><span className="label">Images to import</span><span className="font-mono text-sm text-signal">{limit} / {available}</span></div>
              <input type="range" min="1" max={Math.min(500, available)} value={limit} onChange={(e) => setLimit(Number(e.target.value))} className="w-full accent-cyan-400" />
            </div>
            <ErrorNote error={error} />
            {created ? (
              <Link to={`/app/batches/${created.id}`} className="btn-primary w-full py-3">Open imported batch<ArrowRight size={16} /></Link>
            ) : (
              <button onClick={run} disabled={busy} className="btn-primary w-full py-3">
                {busy ? <><span className="h-4 w-4 animate-spin rounded-full border-2 border-ink-950/30 border-t-ink-950" />Importing…</> : <><Download size={16} />Import {limit} images</>}
              </button>
            )}
            <button onClick={onClose} className="btn-ghost w-full">Close</button>
          </div>
        </div>
      </aside>
    </div>
  )
}
