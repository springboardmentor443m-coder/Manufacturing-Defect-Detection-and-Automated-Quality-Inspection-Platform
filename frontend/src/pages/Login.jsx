import { useState } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { ArrowLeft, ArrowRight, Eye, EyeOff, Lock, Mail } from 'lucide-react'
import { useAuth } from '../auth.jsx'
import { ErrorNote } from '../components/ui.jsx'
import Logo from '../components/Logo.jsx'
import InspectionVisual from '../components/InspectionVisual.jsx'

export default function Login() {
  const { user, login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [show, setShow] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (user) return <Navigate to={location.state?.from || '/app'} replace />

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError('')
    try { await login(email, password); navigate(location.state?.from || '/app', { replace: true }) }
    catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      <div className="relative flex flex-col px-6 py-8 sm:px-12">
        <div className="flex items-center justify-between">
          <Logo />
          <Link to="/" className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-100"><ArrowLeft size={15} />Home</Link>
        </div>
        <div className="m-auto w-full max-w-sm py-12 animate-fadeUp">
          <h1 className="text-3xl font-semibold">Welcome back</h1>
          <p className="mt-2 text-sm text-slate-400">Sign in to the quality inspection console.</p>
          <form onSubmit={submit} className="mt-8 space-y-5">
            <ErrorNote error={error} />
            <div>
              <label className="label" htmlFor="email">Work email</label>
              <div className="relative">
                <Mail size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                <input id="email" type="email" autoComplete="username" className="input pl-10" placeholder="you@factory.com"
                  value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus />
              </div>
            </div>
            <div>
              <label className="label" htmlFor="password">Password</label>
              <div className="relative">
                <Lock size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                <input id="password" type={show ? 'text' : 'password'} autoComplete="current-password" className="input pl-10 pr-10"
                  value={password} onChange={(e) => setPassword(e.target.value)} required />
                <button type="button" onClick={() => setShow((s) => !s)} className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-slate-500 hover:text-slate-200" aria-label={show ? 'Hide password' : 'Show password'}>
                  {show ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>
            <button className="btn-primary w-full py-3" disabled={busy}>
              {busy ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-ink-950/30 border-t-ink-950" /> : <>Sign in<ArrowRight size={16} /></>}
            </button>
          </form>
          <p className="mt-8 text-xs text-slate-500">Accounts are created by your platform administrator.</p>
        </div>
      </div>
      <div className="relative hidden lg:flex items-center justify-center overflow-hidden border-l border-ink-700 bg-ink-900/50 p-16">
        <div className="pointer-events-none absolute inset-0 grid-bg" />
        <div className="relative w-full max-w-md">
          <InspectionVisual compact />
          <blockquote className="mt-8">
            <p className="text-lg text-slate-200 leading-relaxed">“Every part checked, every defect explained.”</p>
            <footer className="mt-2 text-sm text-slate-500">Defect detection · severity scoring · pass/fail decisions</footer>
          </blockquote>
        </div>
      </div>
    </div>
  )
}
