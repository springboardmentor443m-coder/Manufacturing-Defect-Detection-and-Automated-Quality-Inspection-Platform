import { useEffect, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import {
  Camera, ChevronsLeft, ChevronsRight, Database, Layers, LayoutDashboard, LogOut, Menu, Search, UploadCloud, Users, X,
} from 'lucide-react'
import { ROLE_LABELS, can, useAuth } from '../auth.jsx'
import Logo from './Logo.jsx'
import CommandPalette from './CommandPalette.jsx'
import { initials } from './ui.jsx'

export const NAV = [
  { to: '/app', label: 'Overview', icon: LayoutDashboard, show: () => true, group: 'Monitor' },
  { to: '/app/batches', label: 'Inspection Batches', icon: Layers, show: () => true, group: 'Monitor' },
  { to: '/app/upload', label: 'Upload Images', icon: UploadCloud, show: can.inspect, group: 'Acquire' },
  { to: '/app/dataset', label: 'MVTec Dataset', icon: Database, show: can.inspect, group: 'Acquire' },
  { to: '/app/camera', label: 'Line Camera', icon: Camera, show: can.inspect, group: 'Acquire' },
  { to: '/app/users', label: 'Users & Roles', icon: Users, show: can.manageUsers, group: 'Admin' },
]

const COLLAPSE_KEY = 'vi_nav_collapsed'

export default function AppShell({ children }) {
  const { user, logout } = useAuth()
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(() => { try { return localStorage.getItem(COLLAPSE_KEY) === '1' } catch { return false } })
  const [mobileOpen, setMobileOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)

  useEffect(() => { try { localStorage.setItem(COLLAPSE_KEY, collapsed ? '1' : '0') } catch { /* ignore */ } }, [collapsed])
  useEffect(() => setMobileOpen(false), [location.pathname])
  useEffect(() => {
    const onKey = (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPaletteOpen((o) => !o) } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const items = NAV.filter((n) => n.show(user))
  const groups = [...new Set(items.map((i) => i.group))]
  const current = [...items].sort((a, b) => b.to.length - a.to.length).find((n) => location.pathname.startsWith(n.to))

  const sidebar = (
    <div className="flex h-full flex-col">
      <div className={`flex h-16 items-center ${collapsed ? 'justify-center' : 'px-5'}`}>
        <Link to="/app"><Logo withText={!collapsed} /></Link>
      </div>
      <nav className="flex-1 overflow-y-auto px-3 py-2">
        {groups.map((g) => (
          <div key={g} className="mb-5">
            {!collapsed && <div className="px-3 mb-2 text-[10px] font-mono uppercase tracking-[.2em] text-slate-600">{g}</div>}
            {items.filter((i) => i.group === g).map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} end={to === '/app'} title={collapsed ? label : undefined}
                className={({ isActive }) => `group relative mb-0.5 flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors ${collapsed ? 'justify-center' : ''} ${isActive ? 'bg-signal/10 text-signal' : 'text-slate-400 hover:bg-ink-800 hover:text-slate-100'}`}>
                {({ isActive }) => (<>
                  {isActive && <span className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-signal" />}
                  <Icon size={18} strokeWidth={1.8} />
                  {!collapsed && <span>{label}</span>}
                </>)}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <div className="border-t border-ink-700 p-3">
        <div className={`flex items-center gap-3 rounded-xl p-2 ${collapsed ? 'justify-center' : ''}`}>
          <div className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-gradient-to-br from-signal/30 to-indigo-500/30 text-xs font-semibold text-slate-100 ring-1 ring-ink-600">{initials(user.full_name)}</div>
          {!collapsed && (
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-medium text-slate-100">{user.full_name}</div>
              <div className="truncate text-xs text-slate-500">{ROLE_LABELS[user.role]}</div>
            </div>
          )}
          {!collapsed && <button onClick={logout} title="Sign out" className="rounded-lg p-2 text-slate-500 hover:bg-ink-800 hover:text-fail"><LogOut size={16} /></button>}
        </div>
        <button onClick={() => setCollapsed((c) => !c)} className="hidden lg:flex w-full items-center justify-center gap-2 rounded-lg py-2 text-xs text-slate-500 hover:text-slate-200 hover:bg-ink-800">
          {collapsed ? <ChevronsRight size={15} /> : <><ChevronsLeft size={15} />Collapse</>}
        </button>
      </div>
    </div>
  )

  return (
    <div className="min-h-screen">
      {/* desktop sidebar */}
      <aside className={`fixed inset-y-0 left-0 z-30 hidden lg:block border-r border-ink-700 bg-ink-900/95 transition-[width] duration-200 ${collapsed ? 'w-[76px]' : 'w-64'}`}>{sidebar}</aside>
      {/* mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setMobileOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 border-r border-ink-700 bg-ink-900 animate-fadeUp">
            <button className="absolute right-3 top-4 p-2 text-slate-400" onClick={() => setMobileOpen(false)} aria-label="Close menu"><X size={18} /></button>
            {sidebar}
          </aside>
        </div>
      )}

      <div className={`transition-[padding] duration-200 ${collapsed ? 'lg:pl-[76px]' : 'lg:pl-64'}`}>
        <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b border-ink-700 bg-ink-950/80 px-4 backdrop-blur-xl sm:px-8">
          <button className="lg:hidden rounded-lg p-2 text-slate-400 hover:bg-ink-800" onClick={() => setMobileOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
          <div className="hidden sm:flex items-center gap-2 text-sm">
            <span className="text-slate-500">Workspace</span><span className="text-slate-700">/</span>
            <span className="text-slate-200">{current?.label || 'Batch'}</span>
          </div>
          <button onClick={() => setPaletteOpen(true)}
            className="ml-auto flex w-full max-w-xs items-center gap-2 rounded-xl border border-ink-600 bg-ink-850 px-3 py-2 text-sm text-slate-500 hover:border-ink-500 transition">
            <Search size={15} /><span className="flex-1 text-left">Jump to…</span><span className="kbd">Ctrl K</span>
          </button>
          <div className="hidden md:flex items-center gap-2 rounded-xl border border-ink-700 px-3 py-2 text-xs text-slate-400">
            <span className="h-1.5 w-1.5 rounded-full bg-pass shadow-[0_0_8px_#34D399]" />System online
          </div>
        </header>
        <main className="mx-auto max-w-[1400px] px-4 py-8 sm:px-8">{children}</main>
      </div>

      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} items={items} />
    </div>
  )
}
