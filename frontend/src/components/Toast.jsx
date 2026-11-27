import { createContext, useCallback, useContext, useState } from 'react'
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from 'lucide-react'

const ToastContext = createContext(() => {})
const ICONS = { success: [CheckCircle2, 'text-pass'], error: [XCircle, 'text-fail'], warn: [AlertTriangle, 'text-warn'], info: [Info, 'text-signal'] }

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])
  const dismiss = (id) => setToasts((t) => t.filter((x) => x.id !== id))
  const toast = useCallback((title, { type = 'success', desc } = {}) => {
    const id = Math.random().toString(36).slice(2)
    setToasts((t) => [...t.slice(-3), { id, title, desc, type }])
    setTimeout(() => dismiss(id), 4500)
  }, [])

  return (
    <ToastContext.Provider value={toast}>
      {children}
      <div className="fixed bottom-4 right-4 z-[100] flex w-[min(380px,calc(100vw-2rem))] flex-col gap-2" role="status" aria-live="polite">
        {toasts.map((t) => {
          const [Icon, color] = ICONS[t.type]
          return (
            <div key={t.id} className="panel flex items-start gap-3 p-3.5 pr-2 animate-toastIn">
              <Icon size={18} className={`${color} mt-0.5 shrink-0`} />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-slate-100">{t.title}</div>
                {t.desc && <div className="text-xs text-slate-400 mt-0.5">{t.desc}</div>}
              </div>
              <button onClick={() => dismiss(t.id)} className="rounded-md p-1 text-slate-500 hover:text-slate-200 hover:bg-ink-700" aria-label="Dismiss"><X size={14} /></button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}

export const useToast = () => useContext(ToastContext)
