import { useEffect, useState } from 'react'
import { Eye, Search, ShieldCheck, UserPlus, Wrench, X } from 'lucide-react'
import { api } from '../api'
import { ROLE_LABELS, useAuth } from '../auth.jsx'
import { useToast } from '../components/Toast.jsx'
import { ErrorNote, PageHeader, Skeleton, initials, timeAgo } from '../components/ui.jsx'

const EMPTY = { full_name: '', email: '', role: 'quality_engineer', password: '' }
const ROLE_META = {
  admin: { icon: ShieldCheck, cls: 'bg-indigo-400/10 text-indigo-300 ring-indigo-400/25', desc: 'Manage users and everything else' },
  quality_engineer: { icon: Wrench, cls: 'bg-signal/10 text-signal ring-signal/25', desc: 'Upload, import and run inspections' },
  supervisor: { icon: Eye, cls: 'bg-amber-400/10 text-amber-300 ring-amber-400/25', desc: 'View dashboards and results' },
}

export default function Users() {
  const { user: me } = useAuth()
  const toast = useToast()
  const [users, setUsers] = useState(null)
  const [form, setForm] = useState(EMPTY)
  const [showForm, setShowForm] = useState(false)
  const [error, setError] = useState('')
  const [q, setQ] = useState('')

  const load = () => api.users().then(setUsers).catch((e) => setError(e.message))
  useEffect(() => { load() }, [])

  const create = async (e) => {
    e.preventDefault(); setError('')
    try {
      await api.createUser(form)
      toast('User created', { desc: `${form.full_name} · ${ROLE_LABELS[form.role]}` })
      setForm(EMPTY); setShowForm(false); load()
    } catch (err) { setError(err.message) }
  }
  const update = async (u, patch, msg) => {
    try { await api.updateUser(u.id, patch); toast(msg, { desc: u.full_name }); load() }
    catch (err) { toast('Update failed', { type: 'error', desc: err.message }) }
  }

  const shown = (users || []).filter((u) => `${u.full_name} ${u.email}`.toLowerCase().includes(q.toLowerCase()))
  const byRole = (r) => (users || []).filter((u) => u.role === r).length

  return (
    <>
      <PageHeader eyebrow="Admin" title="Users & roles" desc="Control who can run inspections and who can only view results."
        actions={<button className="btn-primary" onClick={() => setShowForm((s) => !s)}>{showForm ? <><X size={15} />Cancel</> : <><UserPlus size={15} />Add user</>}</button>} />

      <div className="mb-5 grid gap-3 md:grid-cols-3">
        {Object.entries(ROLE_META).map(([r, { icon: Icon, cls, desc }]) => (
          <div key={r} className="panel flex items-center gap-4 px-4 py-3.5">
            <div className={`rounded-xl p-2.5 ring-1 ${cls}`}><Icon size={18} /></div>
            <div className="flex-1"><div className="text-sm font-medium text-slate-100">{ROLE_LABELS[r]}</div><div className="text-xs text-slate-500">{desc}</div></div>
            <div className="text-2xl font-semibold tabular-nums text-slate-50">{byRole(r)}</div>
          </div>
        ))}
      </div>

      {showForm && (
        <form onSubmit={create} className="panel-pad mb-5 animate-fadeUp">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div><label className="label">Full name</label><input className="input" value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} required autoFocus /></div>
            <div><label className="label">Email</label><input type="email" className="input" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} required /></div>
            <div><label className="label">Role</label>
              <select className="input" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
                {Object.entries(ROLE_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select></div>
            <div><label className="label">Temporary password</label><input type="password" minLength={8} className="input" placeholder="min. 8 characters" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} required /></div>
          </div>
          <div className="mt-4 flex items-center gap-3"><button className="btn-primary"><UserPlus size={15} />Create user</button><ErrorNote error={error} /></div>
        </form>
      )}

      <div className="panel overflow-hidden">
        <div className="flex items-center gap-3 border-b border-ink-700 p-4">
          <div className="relative w-full sm:w-72">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input className="input pl-9 py-2" placeholder="Search people…" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          <span className="ml-auto text-xs text-slate-500">{users?.length ?? 0} members</span>
        </div>
        {!users ? <div className="p-5 space-y-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-12" />)}</div> : (
          <div className="overflow-x-auto">
            <table className="table-base min-w-[720px]">
              <thead><tr><th className="pl-5">Member</th><th>Role</th><th>Status</th><th>Joined</th></tr></thead>
              <tbody>
                {shown.map((u) => {
                  const self = u.id === me.id
                  return (
                    <tr key={u.id} className="hover:bg-ink-800/40">
                      <td className="pl-5">
                        <div className="flex items-center gap-3">
                          <div className="grid h-9 w-9 place-items-center rounded-full bg-gradient-to-br from-ink-600 to-ink-700 text-xs font-semibold text-slate-200 ring-1 ring-ink-500">{initials(u.full_name)}</div>
                          <div><div className="font-medium text-slate-100">{u.full_name}{self && <span className="ml-2 chip bg-ink-700 text-slate-400 ring-ink-600">you</span>}</div><div className="text-xs text-slate-500">{u.email}</div></div>
                        </div>
                      </td>
                      <td>
                        <select className={`rounded-lg border-0 py-1 pl-2 pr-7 text-xs font-medium ring-1 ${ROLE_META[u.role].cls} bg-transparent disabled:opacity-70`} value={u.role} disabled={self}
                          onChange={(e) => update(u, { role: e.target.value }, `Role changed to ${ROLE_LABELS[e.target.value]}`)}>
                          {Object.entries(ROLE_LABELS).map(([k, v]) => <option key={k} value={k} className="bg-ink-900 text-slate-200">{v}</option>)}
                        </select>
                      </td>
                      <td>
                        <button disabled={self} onClick={() => update(u, { is_active: !u.is_active }, u.is_active ? 'Account disabled' : 'Account enabled')}
                          className="flex items-center gap-2 text-xs disabled:cursor-not-allowed" role="switch" aria-checked={u.is_active}>
                          <span className={`relative h-5 w-9 rounded-full transition ${u.is_active ? 'bg-pass/80' : 'bg-ink-600'}`}>
                            <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${u.is_active ? 'left-[18px]' : 'left-0.5'}`} />
                          </span>
                          <span className={u.is_active ? 'text-slate-300' : 'text-slate-500'}>{u.is_active ? 'Active' : 'Disabled'}</span>
                        </button>
                      </td>
                      <td className="text-xs text-slate-500">{timeAgo(u.created_at)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  )
}
