import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { AlertTriangle, ArrowRight, Camera, Clock, Database, Images, Layers, RefreshCw, UploadCloud, XCircle } from 'lucide-react'
import { api } from '../api'
import { can, useAuth } from '../auth.jsx'
import { EmptyState, ErrorNote, PageHeader, Skeleton, SourceBadge, StatCard, timeAgo } from '../components/ui.jsx'

const VCOLORS = { valid: '#34D399', warning: '#FBBF24', rejected: '#F43F5E' }
const SOURCE_ICON = { upload: UploadCloud, camera: Camera, dataset: Database }

export const chartTooltip = {
  contentStyle: { background: '#0F141B', border: '1px solid #263142', borderRadius: 12, fontSize: 12, color: '#E2E8F0' },
  itemStyle: { color: '#E2E8F0' }, labelStyle: { color: '#94A3B8' }, cursor: { fill: 'rgba(34,211,238,.06)' },
}
const axis = { stroke: '#3A4658', fontSize: 11, tickLine: false, axisLine: false }

export default function Dashboard() {
  const { user } = useAuth()
  const [stats, setStats] = useState(null)
  const [error, setError] = useState('')
  const [refreshing, setRefreshing] = useState(false)

  const load = () => { setRefreshing(true); api.stats().then(setStats).catch((e) => setError(e.message)).finally(() => setRefreshing(false)) }
  useEffect(load, [])

  const greeting = (() => { const h = new Date().getHours(); return h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening' })()

  return (
    <>
      <PageHeader eyebrow="Overview" title={`${greeting}, ${user.full_name.split(' ')[0]}`}
        desc="What's happening on your inspection lines: how many images are coming in, how they passed validation, and what's waiting for the defect model."
        actions={<>
          <button onClick={load} className="btn-ghost" disabled={refreshing}><RefreshCw size={15} className={refreshing ? 'animate-spin' : ''} />Refresh</button>
          {can.inspect(user) && <Link to="/app/upload" className="btn-primary"><UploadCloud size={15} />New inspection</Link>}
        </>} />
      <ErrorNote error={error} />
      {!stats ? <DashboardSkeleton /> : stats.total_images === 0 ? <FirstRun canInspect={can.inspect(user)} /> : <Content stats={stats} />}
    </>
  )
}

function Content({ stats }) {
  const navigate = useNavigate()
  const validation = Object.entries(stats.by_validation).map(([name, value]) => ({ name, value }))
  const categories = Object.entries(stats.by_category).sort((a, b) => b[1] - a[1]).map(([name, value]) => ({ name, value }))
  const truth = Object.entries(stats.by_ground_truth).filter(([k]) => k !== 'unlabelled').sort((a, b) => b[1] - a[1])
  const truthMax = Math.max(1, ...truth.map(([, v]) => v))
  const perDay = stats.uploads_per_day.map((d) => ({ ...d, day: new Date(d.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) }))
  const spark = stats.uploads_per_day.map((d) => d.count)
  const total = stats.total_images
  const passRate = Math.round(((stats.by_validation.valid || 0) + (stats.by_validation.warning || 0)) / total * 100)

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Images captured" value={total} icon={Images} spark={spark} hint="Last 14 days trend" />
        <StatCard label="Inspection batches" value={stats.total_batches} icon={Layers} delay={60} hint={`${Object.keys(stats.by_category).length} product categories`} />
        <StatCard label="Awaiting detection" value={stats.pending_inspection} icon={Clock} tone="warn" delay={120} hint="Defect model arrives in Milestone 2" />
        <StatCard label="Rejected at intake" value={stats.by_validation.rejected || 0} icon={XCircle} tone="fail" delay={180} hint={`${passRate}% of images passed validation`} />
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="panel-pad xl:col-span-2 animate-fadeUp">
          <PanelTitle title="Image intake" sub="Images captured per day, last 14 days" />
          <div className="h-64 -ml-4">
            <ResponsiveContainer>
              <AreaChart data={perDay} margin={{ top: 8, right: 8 }}>
                <defs>
                  <linearGradient id="intake" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#22D3EE" stopOpacity={0.35} /><stop offset="100%" stopColor="#22D3EE" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#1B2430" vertical={false} />
                <XAxis dataKey="day" {...axis} interval="preserveStartEnd" />
                <YAxis allowDecimals={false} {...axis} width={40} />
                <Tooltip {...chartTooltip} />
                <Area type="monotone" dataKey="count" name="Images" stroke="#22D3EE" strokeWidth={2} fill="url(#intake)" activeDot={{ r: 5, strokeWidth: 0, fill: '#67E8F9' }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel-pad animate-fadeUp [animation-delay:80ms]">
          <PanelTitle title="Validation outcome" sub="Image quality checks at intake" />
          <div className="relative h-44">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={validation} dataKey="value" nameKey="name" innerRadius={52} outerRadius={74} paddingAngle={3} stroke="none">
                  {validation.map((v) => <Cell key={v.name} fill={VCOLORS[v.name]} />)}
                </Pie>
                <Tooltip {...chartTooltip} />
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
              <div><div className="text-2xl font-semibold text-slate-50">{passRate}%</div><div className="text-[10px] uppercase tracking-wider text-slate-500">usable</div></div>
            </div>
          </div>
          <ul className="mt-4 space-y-2 text-sm">
            {['valid', 'warning', 'rejected'].map((k) => {
              const v = stats.by_validation[k] || 0
              return (
                <li key={k} className="flex items-center gap-3">
                  <span className="h-2 w-2 rounded-full" style={{ background: VCOLORS[k] }} />
                  <span className="capitalize text-slate-300 flex-1">{k}</span>
                  <span className="tabular-nums text-slate-400">{v}</span>
                  <span className="w-10 text-right tabular-nums text-xs text-slate-500">{Math.round(v / total * 100)}%</span>
                </li>
              )
            })}
          </ul>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="panel-pad animate-fadeUp">
          <PanelTitle title="By product category" sub="Click a bar to filter batches" />
          <div style={{ height: Math.max(160, categories.length * 34) }}>
            <ResponsiveContainer>
              <BarChart data={categories} layout="vertical" margin={{ left: 10, right: 16 }}>
                <XAxis type="number" allowDecimals={false} {...axis} />
                <YAxis type="category" dataKey="name" {...axis} stroke="#94A3B8" width={90} />
                <Tooltip {...chartTooltip} />
                <Bar dataKey="value" name="Images" fill="#22D3EE" radius={[0, 6, 6, 0]} barSize={16} className="cursor-pointer"
                  onClick={(d) => navigate(`/app/batches?category=${d.name}`)} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel-pad animate-fadeUp [animation-delay:80ms]">
          <PanelTitle title="Ground-truth labels" sub="Known labels on dataset and camera images" />
          {truth.length === 0 ? (
            <EmptyState icon={Database} title="No labelled images yet" desc="Import MVTec AD images to see the mix of defect types." />
          ) : (
            <ul className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
              {truth.map(([k, v]) => (
                <li key={k}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className={k === 'good' ? 'text-sky-300' : 'text-slate-300'}>{k.replaceAll('_', ' ')}</span>
                    <span className="tabular-nums text-slate-500">{v}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-ink-700 overflow-hidden">
                    <div className={`h-full rounded-full ${k === 'good' ? 'bg-sky-400' : 'bg-gradient-to-r from-fuchsia-500 to-fail'}`} style={{ width: `${(v / truthMax) * 100}%` }} />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="panel-pad animate-fadeUp">
        <div className="flex items-center justify-between">
          <PanelTitle title="Recent activity" sub="Latest inspection batches" />
          <Link to="/app/batches" className="flex items-center gap-1 text-sm text-signal hover:text-signal-soft">View all<ArrowRight size={14} /></Link>
        </div>
        <ul className="divide-y divide-ink-700/70">
          {stats.recent_batches.map((b) => {
            const Icon = SOURCE_ICON[b.source]
            return (
              <li key={b.id}>
                <Link to={`/app/batches/${b.id}`} className="flex items-center gap-4 py-3 -mx-2 px-2 rounded-xl hover:bg-ink-800/60 transition-colors">
                  <div className="rounded-xl bg-ink-800 p-2.5 text-slate-400 ring-1 ring-ink-600"><Icon size={16} /></div>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium text-slate-100">{b.name}</div>
                    <div className="text-xs text-slate-500">{b.created_by_name} · {timeAgo(b.created_at)}</div>
                  </div>
                  <SourceBadge source={b.source} />
                  <QualityBar b={b} />
                </Link>
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}

export function QualityBar({ b, wide = false }) {
  const t = Math.max(1, b.image_count)
  return (
    <div className={`${wide ? 'w-40' : 'hidden sm:block w-32'}`} title={`${b.valid_count} valid · ${b.warning_count} warning · ${b.rejected_count} rejected`}>
      <div className="flex justify-between text-[10px] text-slate-500 mb-1"><span>{b.image_count} images</span>{b.rejected_count > 0 && <span className="text-fail flex items-center gap-0.5"><AlertTriangle size={10} />{b.rejected_count}</span>}</div>
      <div className="flex h-1.5 overflow-hidden rounded-full bg-ink-700">
        <div className="bg-pass" style={{ width: `${(b.valid_count / t) * 100}%` }} />
        <div className="bg-warn" style={{ width: `${(b.warning_count / t) * 100}%` }} />
        <div className="bg-fail" style={{ width: `${(b.rejected_count / t) * 100}%` }} />
      </div>
    </div>
  )
}

const PanelTitle = ({ title, sub }) => (
  <div className="mb-4"><h2 className="text-sm font-semibold text-slate-100">{title}</h2>{sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}</div>
)

function DashboardSkeleton() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-32 rounded-2xl" />)}</div>
      <div className="grid gap-4 xl:grid-cols-3"><Skeleton className="h-80 rounded-2xl xl:col-span-2" /><Skeleton className="h-80 rounded-2xl" /></div>
    </div>
  )
}

function FirstRun({ canInspect }) {
  const steps = [
    { to: '/app/dataset', icon: Database, title: 'Import MVTec AD images', desc: 'Load labelled images from any of the 15 product categories.' },
    { to: '/app/camera', icon: Camera, title: 'Run the line camera', desc: 'Simulate a production line capturing parts in real time.' },
    { to: '/app/upload', icon: UploadCloud, title: 'Upload your own images', desc: 'Drag and drop product photos to inspect them.' },
  ]
  return (
    <div className="panel p-8 sm:p-10 animate-fadeUp">
      <div className="max-w-xl">
        <div className="text-xs font-mono uppercase tracking-[.2em] text-signal">Get started</div>
        <h2 className="mt-2 text-xl font-semibold">No images yet. Start your first inspection.</h2>
        <p className="mt-2 text-sm text-slate-400">{canInspect ? 'Pick a way to bring images in. The dashboard fills up as soon as they arrive.' : 'A quality engineer needs to bring images in first. The dashboard fills up as soon as they arrive.'}</p>
      </div>
      {canInspect && (
        <div className="mt-8 grid gap-3 md:grid-cols-3">
          {steps.map(({ to, icon: Icon, title, desc }) => (
            <Link key={to} to={to} className="group rounded-2xl border border-ink-700 bg-ink-850 p-5 hover:border-signal/50 hover:shadow-glow transition-all">
              <Icon size={20} className="text-signal" />
              <div className="mt-4 flex items-center gap-1 font-medium text-slate-100">{title}<ArrowRight size={14} className="opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all" /></div>
              <p className="mt-1 text-sm text-slate-400">{desc}</p>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
