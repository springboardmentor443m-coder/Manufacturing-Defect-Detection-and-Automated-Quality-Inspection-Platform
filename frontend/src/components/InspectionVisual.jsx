/**
 * Animated product-inspection mock: a machined part under a scanning beam,
 * a crack gets localised, and the severity score is computed with the
 * weighting from the project brief (size 30 · location 25 · type 25 · confidence 20).
 */
const DUR = '5s'

export default function InspectionVisual({ compact = false }) {
  return (
    <div className="relative">
      <div className="absolute -inset-8 bg-signal/10 blur-3xl rounded-full pointer-events-none" />
      <div className={`relative panel overflow-hidden ${compact ? '' : 'grid md:grid-cols-[1fr_260px]'}`}>
        {/* camera view */}
        <div className="relative aspect-square bg-ink-950 grid-bg">
          <svg viewBox="0 0 400 400" className="absolute inset-0 h-full w-full">
            <defs>
              <radialGradient id="metal" cx="40%" cy="35%" r="70%">
                <stop offset="0%" stopColor="#9AA7B8" /><stop offset="55%" stopColor="#5B6778" /><stop offset="100%" stopColor="#2A323E" />
              </radialGradient>
              <radialGradient id="bore" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#05070A" /><stop offset="85%" stopColor="#0B0F15" /><stop offset="100%" stopColor="#3A4658" />
              </radialGradient>
              <linearGradient id="beam" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#22D3EE" stopOpacity="0" /><stop offset="50%" stopColor="#22D3EE" stopOpacity=".55" /><stop offset="100%" stopColor="#22D3EE" stopOpacity="0" />
              </linearGradient>
              <filter id="soft"><feGaussianBlur stdDeviation="1.2" /></filter>
            </defs>

            {/* hex nut style part */}
            <g transform="translate(200 200)">
              <polygon points="0,-140 121,-70 121,70 0,140 -121,70 -121,-70" fill="url(#metal)" stroke="#8B98AA" strokeOpacity=".4" />
              {[...Array(28)].map((_, i) => (
                <circle key={i} r={58 + i * 2.6} fill="none" stroke="#fff" strokeOpacity={i % 2 ? 0.035 : 0.06} />
              ))}
              <circle r="56" fill="url(#bore)" />
              <circle r="56" fill="none" stroke="#A9B4C4" strokeOpacity=".5" strokeWidth="2" />
              {/* crack */}
              <path d="M58 -62 L70 -55 L66 -45 L80 -36 L76 -26 L88 -18" fill="none" stroke="#0A0D12" strokeWidth="2.6" strokeLinecap="round" />
              <path d="M58 -62 L70 -55 L66 -45 L80 -36 L76 -26 L88 -18" fill="none" stroke="#C7D0DC" strokeOpacity=".35" strokeWidth=".8" transform="translate(1 1)" />
            </g>

            {/* detection overlay: appears after the beam passes the defect */}
            <g opacity="0">
              <animate attributeName="opacity" values="0;0;1;1;0" keyTimes="0;.38;.45;.92;1" dur={DUR} repeatCount="indefinite" />
              <rect x="248" y="128" width="50" height="56" rx="3" fill="#F43F5E" fillOpacity=".12" stroke="#F43F5E" strokeWidth="1.6" strokeDasharray="5 3">
                <animate attributeName="stroke-dashoffset" from="0" to="16" dur="1s" repeatCount="indefinite" />
              </rect>
              <g transform="translate(248 108)">
                <rect width="118" height="18" rx="3" fill="#F43F5E" />
                <text x="6" y="12.5" fontSize="10" fontFamily="JetBrains Mono, monospace" fill="#fff" fontWeight="600">CRACK · 0.92 CONF</text>
              </g>
              <circle cx="273" cy="156" r="5" fill="#F43F5E" filter="url(#soft)">
                <animate attributeName="r" values="4;16" dur="1.2s" repeatCount="indefinite" />
                <animate attributeName="opacity" values=".8;0" dur="1.2s" repeatCount="indefinite" />
              </circle>
            </g>

            {/* scanning beam */}
            <g>
              <animateTransform attributeName="transform" type="translate" values="0 -60;0 400;0 400" keyTimes="0;.8;1" dur={DUR} repeatCount="indefinite" />
              <rect x="0" y="0" width="400" height="60" fill="url(#beam)" />
              <line x1="0" x2="400" y1="30" y2="30" stroke="#67E8F9" strokeWidth="1" />
            </g>
          </svg>

          {/* HUD */}
          <div className="absolute inset-x-0 top-0 flex items-center justify-between px-4 py-3 font-mono text-[10px] text-slate-400">
            <span className="flex items-center gap-1.5"><span className="relative flex h-2 w-2"><span className="absolute inset-0 rounded-full bg-fail animate-pulseRing" /><span className="relative h-2 w-2 rounded-full bg-fail" /></span>REC · CAM-A · LINE 1</span>
            <span>900×900 · 24 FPS</span>
          </div>
          {[['left-3 top-10', 'border-l border-t'], ['right-3 top-10', 'border-r border-t'], ['left-3 bottom-3', 'border-l border-b'], ['right-3 bottom-3', 'border-r border-b']].map(([pos, b]) => (
            <span key={pos} className={`absolute ${pos} h-5 w-5 ${b} border-signal/60`} />
          ))}
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 font-mono text-[10px] text-slate-500">PART #A-20419 · METAL_NUT</div>
        </div>

        {/* readout */}
        {!compact && (
          <div className="border-t md:border-t-0 md:border-l border-ink-700 p-5 font-mono text-xs flex flex-col gap-4">
            <div>
              <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Severity score</div>
              {[['Size', 85, 30], ['Location', 90, 25], ['Defect type', 95, 25], ['Confidence', 92, 20]].map(([k, v, w], i) => (
                <div key={k} className="mb-2.5">
                  <div className="flex justify-between text-slate-400"><span>{k} <span className="text-slate-600">×{w}%</span></span><span className="text-slate-200">{v}</span></div>
                  <div className="mt-1 h-1 rounded-full bg-ink-700 overflow-hidden">
                    <div className="h-full rounded-full bg-gradient-to-r from-amber-400 to-fail origin-left animate-[fadeUp_.8s_ease-out_both]" style={{ width: `${v}%`, animationDelay: `${300 + i * 120}ms` }} />
                  </div>
                </div>
              ))}
            </div>
            <div className="rounded-xl border border-fail/30 bg-fail/10 p-3">
              <div className="flex items-baseline justify-between">
                <span className="text-4xl font-semibold text-slate-50 font-sans tabular-nums">88</span>
                <span className="chip bg-fail/20 text-fail ring-fail/30">CRITICAL</span>
              </div>
              <div className="mt-2 text-[11px] leading-relaxed text-rose-200/80">Reject product · trigger quality inspection workflow</div>
            </div>
            <div className="mt-auto grid grid-cols-2 gap-2 text-[10px]">
              <div className="rounded-lg bg-ink-800 p-2"><div className="text-slate-500">LATENCY</div><div className="text-slate-200 text-sm">42 ms</div></div>
              <div className="rounded-lg bg-ink-800 p-2"><div className="text-slate-500">DECISION</div><div className="text-fail text-sm">FAIL</div></div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
