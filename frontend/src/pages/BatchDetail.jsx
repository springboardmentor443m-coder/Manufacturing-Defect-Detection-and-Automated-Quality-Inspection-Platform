import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  ArrowLeft, ChevronLeft, ChevronRight, Grid2x2, Grid3x3, Layers2, Maximize2, Trash2, X, ZoomIn, ZoomOut,
} from 'lucide-react'
import { api } from '../api'
import { can, useAuth } from '../auth.jsx'
import { useToast } from '../components/Toast.jsx'
import {
  AuthImage, EmptyState, ErrorNote, LabelBadge, PageHeader, Skeleton, SourceBadge, ValidationBadge, fmtBytes, fmtDate,
} from '../components/ui.jsx'

export default function BatchDetail() {
  const { id } = useParams()
  const { user } = useAuth()
  const toast = useToast()
  const navigate = useNavigate()
  const [batch, setBatch] = useState(null)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('all')
  const [label, setLabel] = useState('all')
  const [dense, setDense] = useState(false)
  const [open, setOpen] = useState(null) // index into `images`
  const [confirmDelete, setConfirmDelete] = useState(false)

  useEffect(() => { api.batch(id).then(setBatch).catch((e) => setError(e.message)) }, [id])

  const images = useMemo(() => (batch?.images || []).filter((i) =>
    (filter === 'all' || i.validation_status === filter) && (label === 'all' || i.ground_truth_label === label)), [batch, filter, label])
  const labels = useMemo(() => [...new Set((batch?.images || []).map((i) => i.ground_truth_label).filter(Boolean))].sort(), [batch])

  const remove = async () => {
    try { await api.deleteBatch(id); toast('Batch deleted', { desc: batch.name }); navigate('/app/batches') }
    catch (e) { toast('Delete failed', { type: 'error', desc: e.message }) }
  }

  if (error) return <ErrorNote error={error} />
  if (!batch) return <div className="space-y-4"><Skeleton className="h-16 w-1/2" /><div className="grid grid-cols-2 md:grid-cols-5 gap-3">{[...Array(10)].map((_, i) => <Skeleton key={i} className="aspect-square rounded-2xl" />)}</div></div>

  const counts = { all: batch.image_count, valid: batch.valid_count, warning: batch.warning_count, rejected: batch.rejected_count }

  return (
    <>
      <Link to="/app/batches" className="mb-4 inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-100"><ArrowLeft size={15} />All batches</Link>
      <PageHeader eyebrow={`Batch #${batch.id}`} title={batch.name}
        desc={<span className="flex flex-wrap items-center gap-2"><SourceBadge source={batch.source} /><span className="font-mono text-xs">{batch.product_category}</span><span>·</span>{fmtDate(batch.created_at)} by {batch.created_by_name}</span>}
        actions={can.inspect(user) && (confirmDelete
          ? <div className="flex items-center gap-2 animate-fadeUp"><span className="text-sm text-slate-400">Delete {batch.image_count} images?</span><button className="btn-ghost" onClick={() => setConfirmDelete(false)}>Cancel</button><button className="btn bg-fail text-white hover:bg-rose-500" onClick={remove}>Delete</button></div>
          : <button className="btn-danger" onClick={() => setConfirmDelete(true)}><Trash2 size={15} />Delete batch</button>)} />

      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        {[['Images', batch.image_count, 'text-slate-50'], ['Valid', batch.valid_count, 'text-pass'], ['Warnings', batch.warning_count, 'text-warn'], ['Rejected', batch.rejected_count, 'text-fail']].map(([k, v, c]) => (
          <div key={k} className="panel px-4 py-3"><div className="text-[11px] uppercase tracking-wider text-slate-500">{k}</div><div className={`text-2xl font-semibold tabular-nums ${c}`}>{v}</div></div>
        ))}
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        {Object.keys(counts).map((f) => (
          <button key={f} onClick={() => setFilter(f)}
            className={`rounded-full px-3 py-1.5 text-xs font-medium capitalize ring-1 transition ${filter === f ? 'bg-slate-100 text-ink-950 ring-slate-100' : 'text-slate-400 ring-ink-600 hover:ring-ink-500 hover:text-slate-200'}`}>
            {f} <span className="opacity-60">{counts[f]}</span>
          </button>
        ))}
        {labels.length > 0 && (
          <select value={label} onChange={(e) => setLabel(e.target.value)} className="input w-auto py-1.5 text-xs">
            <option value="all">All labels</option>{labels.map((l) => <option key={l} value={l}>{l.replaceAll('_', ' ')}</option>)}
          </select>
        )}
        <div className="ml-auto flex rounded-xl border border-ink-700 p-1">
          <button onClick={() => setDense(false)} className={`rounded-lg p-1.5 ${!dense ? 'bg-ink-700 text-slate-100' : 'text-slate-500'}`} aria-label="Large grid"><Grid2x2 size={15} /></button>
          <button onClick={() => setDense(true)} className={`rounded-lg p-1.5 ${dense ? 'bg-ink-700 text-slate-100' : 'text-slate-500'}`} aria-label="Dense grid"><Grid3x3 size={15} /></button>
        </div>
      </div>

      {images.length === 0 ? <div className="panel"><EmptyState icon={Layers2} title="No images match these filters" /></div> : (
        <div className={`grid gap-3 ${dense ? 'grid-cols-3 sm:grid-cols-5 lg:grid-cols-8' : 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-5'}`}>
          {images.map((img, i) => (
            <button key={img.id} onClick={() => setOpen(i)}
              className="group relative overflow-hidden rounded-2xl border border-ink-700 bg-ink-900 text-left transition hover:border-signal/50 hover:shadow-glow animate-fadeUp"
              style={{ animationDelay: `${Math.min(i, 20) * 25}ms` }}>
              <div className="relative aspect-square overflow-hidden">
                <AuthImage path={`/images/${img.id}/thumb`} className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105" alt={img.original_filename} />
                <div className="absolute inset-0 bg-gradient-to-t from-ink-950/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <Maximize2 size={15} className="absolute right-2.5 top-2.5 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                <span className={`absolute left-2.5 top-2.5 h-2 w-2 rounded-full ring-2 ring-ink-950 ${img.validation_status === 'valid' ? 'bg-pass' : img.validation_status === 'warning' ? 'bg-warn' : 'bg-fail'}`} />
              </div>
              {!dense && (
                <div className="p-2.5">
                  <div className="truncate text-xs text-slate-300">{img.original_filename}</div>
                  <div className="mt-1.5 flex flex-wrap gap-1"><LabelBadge label={img.ground_truth_label} /></div>
                </div>
              )}
            </button>
          ))}
        </div>
      )}

      {open !== null && images[open] && (
        <Lightbox images={images} index={open} onIndex={setOpen} onClose={() => setOpen(null)} />
      )}
    </>
  )
}

function Lightbox({ images, index, onIndex, onClose }) {
  const img = images[index]
  const hasMask = img.ground_truth_label && img.ground_truth_label !== 'good'
  const [maskOpacity, setMaskOpacity] = useState(0.55)
  const [showMask, setShowMask] = useState(true)
  const [zoom, setZoom] = useState(1)

  const go = useCallback((d) => { onIndex((index + d + images.length) % images.length); setZoom(1) }, [index, images.length, onIndex])

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowRight') go(1)
      else if (e.key === 'ArrowLeft') go(-1)
      else if (e.key.toLowerCase() === 'm') setShowMask((s) => !s)
      else if (e.key === '+' || e.key === '=') setZoom((z) => Math.min(4, z + 0.5))
      else if (e.key === '-') setZoom((z) => Math.max(1, z - 0.5))
    }
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => { window.removeEventListener('keydown', onKey); document.body.style.overflow = '' }
  }, [go, onClose])

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-ink-950/95 backdrop-blur-md animate-[fadeUp_.2s_ease-out]" role="dialog" aria-label="Image viewer">
      <div className="flex items-center gap-3 border-b border-ink-700 px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium text-slate-100">{img.original_filename}</div>
          <div className="text-xs text-slate-500">{index + 1} / {images.length}</div>
        </div>
        <div className="hidden sm:flex items-center gap-1 text-[10px] text-slate-500 mr-2"><span className="kbd">←</span><span className="kbd">→</span> navigate <span className="kbd ml-2">M</span> mask <span className="kbd ml-2">Esc</span> close</div>
        <button onClick={() => setZoom((z) => Math.max(1, z - 0.5))} className="rounded-lg p-2 text-slate-400 hover:bg-ink-800" aria-label="Zoom out"><ZoomOut size={17} /></button>
        <span className="w-10 text-center font-mono text-xs text-slate-400">{zoom}×</span>
        <button onClick={() => setZoom((z) => Math.min(4, z + 0.5))} className="rounded-lg p-2 text-slate-400 hover:bg-ink-800" aria-label="Zoom in"><ZoomIn size={17} /></button>
        <button onClick={onClose} className="rounded-lg p-2 text-slate-400 hover:bg-ink-800 hover:text-slate-100" aria-label="Close"><X size={19} /></button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-auto p-4">
          <button onClick={() => go(-1)} className="absolute left-3 z-10 rounded-full border border-ink-600 bg-ink-900/80 p-2.5 text-slate-300 hover:text-white hover:border-signal/50" aria-label="Previous"><ChevronLeft size={20} /></button>
          <div className="relative transition-transform duration-200" style={{ transform: `scale(${zoom})` }}>
            <AuthImage key={img.id} path={`/images/${img.id}/file`} className="max-h-[70vh] max-w-full rounded-xl" alt={img.original_filename} />
            {hasMask && showMask && <MaskOverlay key={`m${img.id}`} imageId={img.id} opacity={maskOpacity} />}
          </div>
          <button onClick={() => go(1)} className="absolute right-3 z-10 rounded-full border border-ink-600 bg-ink-900/80 p-2.5 text-slate-300 hover:text-white hover:border-signal/50" aria-label="Next"><ChevronRight size={20} /></button>
        </div>

        <aside className="w-full shrink-0 overflow-y-auto border-t border-ink-700 p-5 lg:w-80 lg:border-l lg:border-t-0">
          <div className="flex flex-wrap gap-1.5"><ValidationBadge status={img.validation_status} /><LabelBadge label={img.ground_truth_label} /></div>

          {hasMask && (
            <div className="mt-5 rounded-xl border border-ink-700 bg-ink-900 p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-200">Defect mask overlay</span>
                <button onClick={() => setShowMask((s) => !s)} role="switch" aria-checked={showMask}
                  className={`relative h-5 w-9 rounded-full transition ${showMask ? 'bg-signal' : 'bg-ink-600'}`}>
                  <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${showMask ? 'left-[18px]' : 'left-0.5'}`} />
                </button>
              </div>
              <input type="range" min="0.1" max="1" step="0.05" value={maskOpacity} disabled={!showMask}
                onChange={(e) => setMaskOpacity(Number(e.target.value))} className="mt-3 w-full accent-cyan-400" aria-label="Mask opacity" />
              <p className="mt-1 text-[11px] text-slate-500">Ground-truth pixels from MVTec AD</p>
            </div>
          )}

          <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-4 text-sm">
            <Info k="Resolution" v={`${img.width}×${img.height}`} />
            <Info k="Channels" v={img.channels} />
            <Info k="File size" v={fmtBytes(img.file_size)} />
            <Info k="Inspection" v={<span className="capitalize">{img.inspection_status}</span>} />
          </dl>

          <div className="mt-6 text-[11px] uppercase tracking-wider text-slate-500">Image quality</div>
          <div className="mt-3 space-y-3">
            <Meter k="Sharpness" v={img.blur_score} max={300} good={(v) => v >= 8} />
            <Meter k="Brightness" v={img.brightness} max={255} good={(v) => v >= 30 && v <= 225} />
            <Meter k="Contrast" v={img.contrast} max={128} good={(v) => v >= 2} />
          </div>

          {img.validation_messages.length > 0 && (
            <ul className="mt-5 space-y-1.5 rounded-xl border border-warn/30 bg-warn/5 p-3 text-xs text-amber-200">
              {img.validation_messages.map((m) => <li key={m}>• {m}</li>)}
            </ul>
          )}
          <div className="mt-6 text-xs text-slate-500">Captured {fmtDate(img.created_at)}</div>
        </aside>
      </div>
    </div>
  )
}

/** Paints the white pixels of the ground-truth mask in signal red over the image. */
function MaskOverlay({ imageId, opacity }) {
  const [url, setUrl] = useState(null)
  useEffect(() => {
    let u
    let alive = true
    api.blob(`/images/${imageId}/mask`).then((b) => { u = URL.createObjectURL(b); if (alive) setUrl(u) }).catch(() => {})
    return () => { alive = false; if (u) URL.revokeObjectURL(u) }
  }, [imageId])
  if (!url) return null
  const mask = { maskImage: `url(${url})`, maskMode: 'luminance', maskSize: '100% 100%', WebkitMaskImage: `url(${url})`, WebkitMaskSize: '100% 100%' }
  return (
    <div className="pointer-events-none absolute inset-0 rounded-xl transition-opacity"
      style={{ ...mask, opacity, background: 'repeating-linear-gradient(135deg, #F43F5E 0 6px, #FB7185 6px 9px)' }} />
  )
}

const Info = ({ k, v }) => (
  <div><dt className="text-[11px] uppercase tracking-wider text-slate-500">{k}</dt><dd className="mt-0.5 font-mono text-slate-200">{v}</dd></div>
)

function Meter({ k, v, max, good }) {
  const ok = good(v)
  return (
    <div>
      <div className="flex justify-between text-xs"><span className="text-slate-400">{k}</span><span className={`font-mono ${ok ? 'text-slate-200' : 'text-warn'}`}>{v}</span></div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink-700">
        <div className={`h-full rounded-full ${ok ? 'bg-signal' : 'bg-warn'}`} style={{ width: `${Math.min(100, (v / max) * 100)}%` }} />
      </div>
    </div>
  )
}
