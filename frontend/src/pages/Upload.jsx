import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, CheckCircle2, FileImage, ImagePlus, Trash2, UploadCloud, X } from 'lucide-react'
import { api } from '../api'
import { useToast } from '../components/Toast.jsx'
import { ErrorNote, PageHeader, fmtBytes } from '../components/ui.jsx'

const ACCEPT = '.png,.jpg,.jpeg,.bmp,.tif,.tiff'
const MAX_FILES = 100
const MAX_MB = 15
const SUGGESTED = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'transistor']

export default function Upload() {
  const toast = useToast()
  const inputRef = useRef(null)
  const [files, setFiles] = useState([])
  const [category, setCategory] = useState('')
  const [name, setName] = useState('')
  const [drag, setDrag] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const previews = useMemo(() => files.map((f) => (f.type.startsWith('image/') && !/tiff?$/i.test(f.name) ? URL.createObjectURL(f) : null)), [files])
  useEffect(() => () => previews.forEach((u) => u && URL.revokeObjectURL(u)), [previews])

  const addFiles = (list) => {
    const incoming = Array.from(list)
    const tooBig = incoming.filter((f) => f.size > MAX_MB * 1024 * 1024)
    if (tooBig.length) toast(`${tooBig.length} file(s) over ${MAX_MB} MB skipped`, { type: 'warn' })
    setFiles((prev) => [...prev, ...incoming.filter((f) => f.size <= MAX_MB * 1024 * 1024)].slice(0, MAX_FILES))
    setResult(null)
  }
  const totalSize = files.reduce((s, f) => s + f.size, 0)

  const submit = async (e) => {
    e.preventDefault()
    if (!files.length) return setError('Add at least one image')
    setBusy(true); setError(''); setResult(null)
    const form = new FormData()
    form.append('product_category', category)
    form.append('batch_name', name)
    files.forEach((f) => form.append('files', f))
    try {
      const r = await api.upload(form)
      setResult(r); setFiles([])
      toast('Batch uploaded', { desc: `${r.accepted} accepted · ${r.rejected} rejected`, type: r.rejected ? 'warn' : 'success' })
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  return (
    <>
      <PageHeader eyebrow="Acquire" title="Upload product images" desc="Each image is checked for file type, resolution, sharpness and exposure, then queued for defect detection." />

      <form onSubmit={submit} className="grid gap-5 xl:grid-cols-[1fr_340px]">
        <div className="space-y-5">
          <div
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
            role="button" tabIndex={0}
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files) }}
            className={`relative cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed p-10 text-center transition-all ${drag ? 'border-signal bg-signal/5 shadow-glow' : 'border-ink-600 hover:border-ink-500 bg-ink-900/50'}`}>
            <input ref={inputRef} type="file" multiple accept={ACCEPT} className="hidden" onChange={(e) => { addFiles(e.target.files); e.target.value = '' }} />
            <div className="pointer-events-none absolute inset-0 grid-bg opacity-50" />
            <div className={`relative mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-signal/10 text-signal ring-1 ring-signal/25 transition-transform ${drag ? 'scale-110' : ''}`}><UploadCloud size={26} /></div>
            <div className="relative mt-4 font-medium text-slate-100">{drag ? 'Drop to add images' : 'Drag product images here'}</div>
            <div className="relative mt-1 text-sm text-slate-400">or <span className="text-signal">browse files</span></div>
            <div className="relative mt-3 text-xs text-slate-500">PNG · JPG · BMP · TIFF — up to {MAX_FILES} files, {MAX_MB} MB each</div>
          </div>

          {files.length > 0 && (
            <div className="panel-pad animate-fadeUp">
              <div className="mb-4 flex items-center justify-between">
                <div className="text-sm"><span className="font-medium text-slate-100">{files.length} images</span><span className="text-slate-500"> · {fmtBytes(totalSize)}</span></div>
                <button type="button" onClick={() => setFiles([])} className="flex items-center gap-1 text-xs text-slate-400 hover:text-fail"><Trash2 size={13} />Clear all</button>
              </div>
              <div className="grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-6">
                {files.map((f, i) => (
                  <div key={`${f.name}-${i}`} className="group relative overflow-hidden rounded-xl border border-ink-700 bg-ink-850">
                    {previews[i] ? <img src={previews[i]} alt="" className="aspect-square w-full object-cover" /> : <div className="grid aspect-square place-items-center text-slate-500"><FileImage size={22} /></div>}
                    <div className="truncate px-2 py-1.5 text-[10px] text-slate-400">{f.name}</div>
                    <button type="button" onClick={() => setFiles((p) => p.filter((_, j) => j !== i))}
                      className="absolute right-1.5 top-1.5 rounded-full bg-ink-950/80 p-1 text-slate-300 opacity-0 group-hover:opacity-100 hover:text-fail transition" aria-label={`Remove ${f.name}`}><X size={12} /></button>
                  </div>
                ))}
                {files.length < MAX_FILES && (
                  <button type="button" onClick={() => inputRef.current?.click()} className="grid aspect-square place-items-center rounded-xl border border-dashed border-ink-600 text-slate-500 hover:border-signal/50 hover:text-signal transition"><ImagePlus size={20} /></button>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="space-y-5">
          <div className="panel-pad space-y-4 xl:sticky xl:top-24">
            <div>
              <label className="label" htmlFor="cat">Product category</label>
              <input id="cat" className="input" placeholder="e.g. bottle" value={category} onChange={(e) => setCategory(e.target.value)} required list="cats" />
              <datalist id="cats">{SUGGESTED.map((c) => <option key={c} value={c} />)}</datalist>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {SUGGESTED.slice(0, 5).map((c) => (
                  <button key={c} type="button" onClick={() => setCategory(c)} className={`rounded-md px-2 py-0.5 font-mono text-[11px] ring-1 transition ${category === c ? 'bg-signal/15 text-signal ring-signal/30' : 'text-slate-400 ring-ink-600 hover:text-slate-200'}`}>{c}</button>
                ))}
              </div>
            </div>
            <div>
              <label className="label" htmlFor="bname">Batch name <span className="normal-case text-slate-600">(optional)</span></label>
              <input id="bname" className="input" placeholder="Line 2 · morning shift" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <ErrorNote error={error} />
            <button className="btn-primary w-full py-3" disabled={busy || !files.length}>
              {busy ? <><span className="h-4 w-4 animate-spin rounded-full border-2 border-ink-950/30 border-t-ink-950" />Validating {files.length} images…</> : <><UploadCloud size={16} />Upload {files.length || ''} {files.length === 1 ? 'image' : 'images'}</>}
            </button>
            <ul className="space-y-2 border-t border-ink-700 pt-4 text-xs text-slate-500">
              <li className="flex gap-2"><CheckCircle2 size={14} className="text-slate-600 shrink-0" />Corrupt, blank or tiny images are rejected</li>
              <li className="flex gap-2"><CheckCircle2 size={14} className="text-slate-600 shrink-0" />Blurry or badly lit images are flagged with a warning</li>
              <li className="flex gap-2"><CheckCircle2 size={14} className="text-slate-600 shrink-0" />Files uploaded before are skipped automatically</li>
            </ul>
          </div>

          {result && (
            <div className="panel-pad animate-fadeUp border-pass/30">
              <div className="flex items-center gap-2 text-sm font-medium text-slate-100"><CheckCircle2 size={17} className="text-pass" />Upload complete</div>
              <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                <div className="rounded-xl bg-pass/10 py-2"><div className="text-lg font-semibold text-pass">{result.accepted}</div><div className="text-[10px] uppercase text-slate-500">accepted</div></div>
                <div className="rounded-xl bg-fail/10 py-2"><div className="text-lg font-semibold text-fail">{result.rejected}</div><div className="text-[10px] uppercase text-slate-500">rejected</div></div>
                <div className="rounded-xl bg-ink-800 py-2"><div className="text-lg font-semibold text-slate-300">{result.duplicates.length}</div><div className="text-[10px] uppercase text-slate-500">duplicates</div></div>
              </div>
              <Link className="btn-ghost mt-4 w-full" to={`/app/batches/${result.batch.id}`}>Review batch<ArrowRight size={15} /></Link>
            </div>
          )}
        </div>
      </form>
    </>
  )
}
