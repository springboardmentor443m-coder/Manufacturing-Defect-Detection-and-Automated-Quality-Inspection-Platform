import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CornerDownLeft, Layers, LogOut, Search } from 'lucide-react'
import { api } from '../api'
import { useAuth } from '../auth.jsx'

export default function CommandPalette({ open, onClose, items }) {
  const navigate = useNavigate()
  const { logout } = useAuth()
  const [q, setQ] = useState('')
  const [active, setActive] = useState(0)
  const [batches, setBatches] = useState([])
  const inputRef = useRef(null)

  useEffect(() => {
    if (!open) return
    setQ(''); setActive(0)
    setTimeout(() => inputRef.current?.focus(), 10)
    api.batches({ limit: 30 }).then(setBatches).catch(() => setBatches([]))
  }, [open])

  const results = useMemo(() => {
    const all = [
      ...items.map((i) => ({ id: i.to, label: i.label, hint: 'Page', icon: i.icon, run: () => navigate(i.to) })),
      ...batches.map((b) => ({ id: `b${b.id}`, label: b.name, hint: `${b.image_count} images · ${b.product_category}`, icon: Layers, run: () => navigate(`/app/batches/${b.id}`) })),
      { id: 'logout', label: 'Sign out', hint: 'Account', icon: LogOut, run: logout },
    ]
    const s = q.trim().toLowerCase()
    return s ? all.filter((r) => `${r.label} ${r.hint}`.toLowerCase().includes(s)) : all.slice(0, 12)
  }, [q, items, batches, navigate, logout])

  useEffect(() => setActive(0), [q])
  if (!open) return null

  const onKey = (e) => {
    if (e.key === 'Escape') onClose()
    else if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(a + 1, results.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)) }
    else if (e.key === 'Enter' && results[active]) { results[active].run(); onClose() }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh]" onKeyDown={onKey}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-xl panel overflow-hidden animate-toastIn" role="dialog" aria-label="Command palette">
        <div className="flex items-center gap-3 border-b border-ink-700 px-4">
          <Search size={17} className="text-slate-500" />
          <input ref={inputRef} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search pages and batches…"
            className="h-14 flex-1 bg-transparent text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none" />
          <span className="kbd">Esc</span>
        </div>
        <ul className="max-h-[50vh] overflow-y-auto p-2">
          {results.length === 0 && <li className="px-3 py-8 text-center text-sm text-slate-500">No matches for “{q}”</li>}
          {results.map((r, i) => (
            <li key={r.id}>
              <button onMouseEnter={() => setActive(i)} onClick={() => { r.run(); onClose() }}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm ${i === active ? 'bg-signal/10 text-slate-50' : 'text-slate-300'}`}>
                <r.icon size={16} className={i === active ? 'text-signal' : 'text-slate-500'} />
                <span className="flex-1 truncate">{r.label}</span>
                <span className="text-xs text-slate-500">{r.hint}</span>
                {i === active && <CornerDownLeft size={13} className="text-slate-500" />}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
