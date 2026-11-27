import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Camera, Pause, Play, Radio } from 'lucide-react'
import { api } from '../api'
import { useToast } from '../components/Toast.jsx'
import { AuthImage, ErrorNote, LabelBadge, PageHeader } from '../components/ui.jsx'

const STATIONS = ['Line-1 / Cam-A', 'Line-1 / Cam-B', 'Line-2 / Cam-A', 'Line-3 / Cam-A']

export default function CameraPage() {
  const toast = useToast()
  const [cats, setCats] = useState([])
  const [category, setCategory] = useState('')
  const [station, setStation] = useState(STATIONS[0])
  const [count, setCount] = useState(12)
  const [ratio, setRatio] = useState(0.3)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [frames, setFrames] = useState([]) // images revealed so far
  const [batch, setBatch] = useState(null)
  const [focus, setFocus] = useState(null)
  const timer = useRef(null)

  useEffect(() => {
    api.dataset().then((d) => { setCats(d.categories.map((c) => c.name)); setCategory(d.categories[0]?.name || '') }).catch((e) => setError(e.message))
    return () => clearInterval(timer.current)
  }, [])

  const capture = async () => {
    setBusy(true); setError(''); setFrames([]); setBatch(null); setFocus(null)
    clearInterval(timer.current)
    try {
      const b = await api.capture({ category, count, defect_ratio: ratio, station })
      const detail = await api.batch(b.id)
      setBatch(b)
      // Reveal frames one by one, like they're coming off the line.
      let i = 0
      timer.current = setInterval(() => {
        i += 1
        setFrames(detail.images.slice(0, i))
        setFocus(detail.images[i - 1])
        if (i >= detail.images.length) {
          clearInterval(timer.current); setBusy(false)
          const bad = detail.images.filter((x) => x.ground_truth_label && x.ground_truth_label !== 'good').length
          toast('Capture complete', { desc: `${detail.images.length} frames · ${bad} known defective`, type: bad ? 'warn' : 'success' })
        }
      }, 450)
    } catch (e) { setError(e.message); setBusy(false) }
  }

  const defective = frames.filter((f) => f.ground_truth_label && f.ground_truth_label !== 'good').length

  return (
    <>
      <PageHeader eyebrow="Acquire" title="Line camera"
        desc="Simulates a production-line camera. It captures MVTec parts with realistic lighting variation and sensor noise, mixing in defective parts at the rate you choose." />
      <div className="grid gap-5 xl:grid-cols-[1fr_340px]">
        <div className="space-y-5">
          {/* live viewport */}
          <div className="panel relative aspect-video overflow-hidden bg-ink-950">
            <div className="absolute inset-0 grid-bg" />
            {focus ? (
              <AuthImage key={focus.id} path={`/images/${focus.id}/file`} className="absolute inset-0 m-auto h-full object-contain" alt="" />
            ) : (
              <div className="absolute inset-0 grid place-items-center text-center">
                <div><Camera size={34} className="mx-auto text-slate-600" /><div className="mt-3 text-sm text-slate-500">Camera idle. Choose a product and press Capture.</div></div>
              </div>
            )}
            {busy && <div className="pointer-events-none absolute inset-x-0 h-24 bg-gradient-to-b from-transparent via-signal/25 to-transparent animate-scan" />}
            <div className="absolute inset-x-0 top-0 flex items-center justify-between px-4 py-3 font-mono text-[11px] text-slate-300">
              <span className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${busy ? 'bg-fail animate-pulse' : 'bg-slate-600'}`} />{busy ? 'LIVE' : 'STANDBY'} · {station.toUpperCase()}
              </span>
              <span>{frames.length}/{batch?.image_count || count} FRAMES</span>
            </div>
            {focus && (
              <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-lg bg-ink-950/80 px-2.5 py-1.5 font-mono text-[11px] text-slate-300 backdrop-blur">
                {focus.original_filename}<LabelBadge label={focus.ground_truth_label} />
              </div>
            )}
          </div>

          {/* filmstrip */}
          {frames.length > 0 && (
            <div className="panel-pad">
              <div className="mb-3 flex items-center justify-between text-sm">
                <span className="text-slate-300">Captured frames</span>
                <span className="text-xs text-slate-500">{defective} with known defects</span>
              </div>
              <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-8">
                {frames.map((f) => {
                  const bad = f.ground_truth_label && f.ground_truth_label !== 'good'
                  return (
                    <button key={f.id} onClick={() => setFocus(f)}
                      className={`relative overflow-hidden rounded-lg ring-2 transition animate-fadeUp ${focus?.id === f.id ? 'ring-signal' : bad ? 'ring-fuchsia-500/50' : 'ring-transparent'}`}>
                      <AuthImage path={`/images/${f.id}/thumb`} className="aspect-square w-full object-cover" alt="" />
                      {bad && <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-fuchsia-400 shadow-[0_0_6px_#E879F9]" />}
                    </button>
                  )
                })}
              </div>
              {batch && !busy && <Link to={`/app/batches/${batch.id}`} className="btn-ghost mt-4">Open batch #{batch.id}<ArrowRight size={15} /></Link>}
            </div>
          )}
        </div>

        <div className="panel-pad h-fit space-y-5 xl:sticky xl:top-24">
          <div>
            <label className="label">Product</label>
            <select className="input font-mono" value={category} onChange={(e) => setCategory(e.target.value)} disabled={busy}>
              {cats.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Station</label>
            <select className="input" value={station} onChange={(e) => setStation(e.target.value)} disabled={busy}>
              {STATIONS.map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <div className="flex justify-between"><span className="label">Frames</span><span className="font-mono text-sm text-signal">{count}</span></div>
            <input type="range" min="1" max="50" value={count} onChange={(e) => setCount(Number(e.target.value))} className="w-full accent-cyan-400" disabled={busy} />
          </div>
          <div>
            <div className="flex justify-between"><span className="label">Defect rate</span><span className="font-mono text-sm text-fuchsia-300">{Math.round(ratio * 100)}%</span></div>
            <input type="range" min="0" max="1" step="0.05" value={ratio} onChange={(e) => setRatio(Number(e.target.value))} className="w-full accent-fuchsia-400" disabled={busy} />
            <div className="mt-1 text-[11px] text-slate-500">Share of parts drawn from MVTec's defective test images</div>
          </div>
          <ErrorNote error={error} />
          <button onClick={capture} disabled={busy || !category} className="btn-primary w-full py-3">
            {busy ? <><Pause size={16} />Capturing…</> : <><Play size={16} />Capture</>}
          </button>
          <div className="flex items-start gap-2 rounded-xl bg-ink-850 p-3 text-xs text-slate-500">
            <Radio size={14} className="mt-0.5 shrink-0 text-signal" />Frames are validated like uploads and queued for defect detection. Each keeps its true label so the model can be checked against it later.
          </div>
        </div>
      </div>
    </>
  )
}
