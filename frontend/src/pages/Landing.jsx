import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity, ArrowRight, BarChart3, Camera, CheckCircle2, Cpu, Crosshair, Database, Gauge, Layers, ScanLine, ShieldCheck, Sparkles, Tags,
} from 'lucide-react'
import { useAuth } from '../auth.jsx'
import Logo from '../components/Logo.jsx'
import InspectionVisual from '../components/InspectionVisual.jsx'

const MODULES = [
  { icon: ShieldCheck, title: 'User management', desc: 'Separate accounts for quality engineers and supervisors, with JWT login and role-based access.', status: 'live' },
  { icon: Camera, title: 'Image acquisition', desc: 'Batch upload, a simulated line camera, and automatic checks for format, resolution, blur and exposure.', status: 'live' },
  { icon: ScanLine, title: 'Image processing', desc: 'Noise removal, contrast enhancement and feature extraction before images reach the model.', status: 'next' },
  { icon: Crosshair, title: 'Defect detection', desc: 'Finds anomalies and marks where each defect is on the part.', status: 'next' },
  { icon: Tags, title: 'Classification & severity', desc: 'Names the defect type and scores it 0–100 from its size, location, type and the model\'s confidence.', status: 'planned' },
  { icon: Gauge, title: 'Quality control', desc: 'Automatic pass/fail decisions, inspection reports and recommended actions.', status: 'planned' },
  { icon: BarChart3, title: 'Manufacturing analytics', desc: 'Defect trends, quality reports and statistics for each production line.', status: 'live' },
]

const STATUS = {
  live: 'bg-pass/10 text-pass ring-pass/25',
  next: 'bg-signal/10 text-signal ring-signal/25',
  planned: 'bg-slate-500/10 text-slate-400 ring-slate-500/25',
}

const STEPS = [
  { icon: Camera, title: 'Capture', desc: 'Images come in from line cameras, batch uploads or the MVTec AD dataset.' },
  { icon: Cpu, title: 'Detect', desc: 'Computer vision models find and outline any defect.' },
  { icon: Layers, title: 'Score', desc: 'Each defect is classified and given a severity level from Low to Critical.' },
  { icon: Activity, title: 'Decide', desc: 'The part passes or fails, and the results feed into line analytics.' },
]

export default function Landing() {
  const { user } = useAuth()
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 10)
    window.addEventListener('scroll', on); return () => window.removeEventListener('scroll', on)
  }, [])
  const cta = user ? { to: '/app', label: 'Open console' } : { to: '/login', label: 'Sign in' }

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <div className="pointer-events-none absolute inset-0 grid-bg [mask-image:radial-gradient(ellipse_at_top,black_30%,transparent_70%)]" />
      <div className="pointer-events-none absolute -top-40 left-1/2 h-[500px] w-[900px] -translate-x-1/2 rounded-full bg-signal/10 blur-[120px]" />

      {/* nav */}
      <header className={`sticky top-0 z-30 transition-colors ${scrolled ? 'border-b border-ink-700 bg-ink-950/80 backdrop-blur-xl' : ''}`}>
        <div className="mx-auto flex h-16 max-w-6xl items-center gap-8 px-4 sm:px-6">
          <Logo />
          <nav className="hidden md:flex gap-6 text-sm text-slate-400">
            <a href="#platform" className="hover:text-slate-100">Platform</a>
            <a href="#workflow" className="hover:text-slate-100">Workflow</a>
            <a href="#severity" className="hover:text-slate-100">Severity model</a>
            <a href="#dataset" className="hover:text-slate-100">Dataset</a>
          </nav>
          <Link to={cta.to} className="btn-primary ml-auto">{cta.label}<ArrowRight size={15} /></Link>
        </div>
      </header>

      {/* hero */}
      <section className="relative mx-auto grid max-w-7xl items-center gap-12 px-4 pb-24 pt-14 sm:px-6 lg:grid-cols-[1.1fr_1fr] lg:pt-20">
        <div className="animate-fadeUp">
          <div className="inline-flex items-center gap-2 rounded-full border border-ink-600 bg-ink-900/70 px-3 py-1 text-xs text-slate-300">
            <Sparkles size={13} className="text-signal" />Computer vision for Industry 4.0
          </div>
          <h1 className="mt-6 text-4xl font-semibold leading-[1.08] sm:text-5xl xl:text-[3.5rem]">
            Find every defect<br className="hidden sm:block" />
            <span className="bg-gradient-to-r from-signal via-sky-300 to-indigo-300 bg-clip-text text-transparent">before it ships.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-slate-400">
            VisionInspect AI checks product images automatically. It finds and classifies defects, scores how severe they are, and makes a pass/fail call, so quality teams spend less time on manual inspection and catch more problems.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link to={cta.to} className="btn-primary px-5 py-3 text-base">{user ? 'Open console' : 'Get started'}<ArrowRight size={17} /></Link>
            <a href="#workflow" className="btn-ghost px-5 py-3 text-base">See how it works</a>
          </div>
          <dl className="mt-12 grid max-w-lg grid-cols-3 gap-6 border-t border-ink-700 pt-6">
            {[['15', 'product categories'], ['5,354', 'labelled images'], ['73', 'defect types']].map(([v, k]) => (
              <div key={k}><dt className="text-2xl font-semibold text-slate-50 tabular-nums">{v}</dt><dd className="mt-1 text-xs text-slate-500">{k}</dd></div>
            ))}
          </dl>
        </div>
        <div className="animate-fadeUp [animation-delay:150ms]"><InspectionVisual /></div>
      </section>

      {/* workflow */}
      <section id="workflow" className="relative border-y border-ink-700 bg-ink-900/40 py-20">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <SectionTitle eyebrow="Workflow" title="From the camera to a decision in milliseconds" />
          <ol className="mt-12 grid gap-4 md:grid-cols-4">
            {STEPS.map(({ icon: Icon, title, desc }, i) => (
              <li key={title} className="panel-pad relative group hover:border-signal/40 transition-colors">
                <div className="flex items-center justify-between">
                  <div className="rounded-xl bg-signal/10 p-2.5 text-signal ring-1 ring-signal/20 group-hover:shadow-glow transition-shadow"><Icon size={20} /></div>
                  <span className="font-mono text-xs text-slate-600">0{i + 1}</span>
                </div>
                <div className="mt-5 font-medium text-slate-100">{title}</div>
                <p className="mt-1.5 text-sm text-slate-400 leading-relaxed">{desc}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* modules */}
      <section id="platform" className="mx-auto max-w-6xl px-4 py-24 sm:px-6">
        <SectionTitle eyebrow="Platform" title="Seven modules, one inspection platform" desc="Each module covers one stage of quality inspection. They're being delivered across four milestones." />
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {MODULES.map(({ icon: Icon, title, desc, status }) => (
            <div key={title} className="panel-pad group hover:-translate-y-0.5 hover:border-ink-500 transition-all">
              <div className="flex items-start justify-between">
                <Icon size={22} className="text-slate-300 group-hover:text-signal transition-colors" strokeWidth={1.6} />
                <span className={`chip ${STATUS[status]}`}>{status}</span>
              </div>
              <div className="mt-5 font-medium text-slate-100">{title}</div>
              <p className="mt-1.5 text-sm text-slate-400 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* severity */}
      <section id="severity" className="border-y border-ink-700 bg-ink-900/40 py-24">
        <div className="mx-auto grid max-w-6xl items-center gap-12 px-4 sm:px-6 lg:grid-cols-2">
          <div>
            <SectionTitle eyebrow="Severity model" title="A severity score you can check" align="left"
              desc="Every defect gets a weighted score out of 100, so engineers can see exactly why a part was rejected." />
            <div className="mt-8 rounded-2xl border border-ink-700 bg-ink-950 p-5 font-mono text-sm leading-7">
              <div className="text-slate-500">// severity = weighted sum</div>
              <div><span className="text-signal">score</span> = size × <span className="text-amber-300">0.30</span></div>
              <div className="pl-[4.4rem]">+ location × <span className="text-amber-300">0.25</span></div>
              <div className="pl-[4.4rem]">+ defect_type × <span className="text-amber-300">0.25</span></div>
              <div className="pl-[4.4rem]">+ confidence × <span className="text-amber-300">0.20</span></div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {[
              ['Critical', '80–100', 'Reject the product and act immediately', 'border-fail/40 bg-fail/10 text-fail'],
              ['High', '60–79', 'Repair or rework recommended', 'border-orange-400/40 bg-orange-400/10 text-orange-300'],
              ['Medium', '40–59', 'Send for inspection review', 'border-warn/40 bg-warn/10 text-warn'],
              ['Low', '0–39', 'Cosmetic; generally acceptable', 'border-pass/40 bg-pass/10 text-pass'],
            ].map(([lvl, range, act, cls]) => (
              <div key={lvl} className={`rounded-2xl border p-5 ${cls}`}>
                <div className="font-mono text-xs opacity-80">{range}</div>
                <div className="mt-1 text-xl font-semibold">{lvl}</div>
                <div className="mt-2 text-xs text-slate-300/80 leading-relaxed">{act}</div>
              </div>
            ))}
            <div className="col-span-2 rounded-2xl border border-ink-700 bg-ink-900 p-4 text-xs text-slate-400 flex items-center gap-3">
              <CheckCircle2 size={16} className="text-signal shrink-0" />
              Any prediction below 70% confidence goes to a person for manual review.
            </div>
          </div>
        </div>
      </section>

      {/* dataset */}
      <section id="dataset" className="mx-auto max-w-6xl px-4 py-24 sm:px-6">
        <div className="panel relative overflow-hidden p-8 sm:p-12">
          <div className="pointer-events-none absolute -right-20 -top-20 h-72 w-72 rounded-full bg-indigo-500/10 blur-3xl" />
          <div className="relative grid gap-10 lg:grid-cols-[1.2fr_1fr] items-center">
            <div>
              <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-[.2em] text-signal"><Database size={14} />Benchmark data</div>
              <h2 className="mt-3 text-3xl font-semibold">Trained and tested on MVTec AD</h2>
              <p className="mt-4 text-slate-400 leading-relaxed">
                MVTec AD is the industry-standard benchmark for anomaly detection. It covers 15 kinds of objects and textures, from bottles and screws to carpet and leather, and marks every defect at the pixel level.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'].map((c) => (
                <span key={c} className="rounded-lg border border-ink-600 bg-ink-850 px-3 py-1.5 font-mono text-xs text-slate-300">{c}</span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* cta */}
      <section className="mx-auto max-w-6xl px-4 pb-24 text-center sm:px-6">
        <h2 className="text-3xl font-semibold sm:text-4xl">Ready to inspect?</h2>
        <p className="mt-3 text-slate-400">Sign in to the console to upload images, connect a line camera and watch your quality metrics live.</p>
        <Link to={cta.to} className="btn-primary mt-8 px-6 py-3 text-base">{cta.label}<ArrowRight size={17} /></Link>
      </section>

      <footer className="border-t border-ink-700 py-8">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-4 text-xs text-slate-500 sm:px-6">
          <Logo size={22} />
          <span>Infosys Springboard internship project · MVTec AD dataset © MVTec Software GmbH, CC BY-NC-SA 4.0</span>
        </div>
      </footer>
    </div>
  )
}

function SectionTitle({ eyebrow, title, desc, align = 'center' }) {
  return (
    <div className={align === 'center' ? 'mx-auto max-w-2xl text-center' : ''}>
      <div className="text-xs font-mono uppercase tracking-[.2em] text-signal">{eyebrow}</div>
      <h2 className="mt-3 text-3xl font-semibold sm:text-4xl">{title}</h2>
      {desc && <p className="mt-4 text-slate-400 leading-relaxed">{desc}</p>}
    </div>
  )
}
