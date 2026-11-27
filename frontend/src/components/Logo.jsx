export default function Logo({ size = 28, withText = true }) {
  return (
    <span className="flex items-center gap-2.5">
      <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" rx="9" fill="#0F141B" stroke="#263142" />
        <circle cx="16" cy="16" r="8.5" fill="none" stroke="#22D3EE" strokeWidth="2.2" />
        <path d="M16 4.5v4M16 23.5v4M4.5 16h4M23.5 16h4" stroke="#22D3EE" strokeOpacity=".5" strokeWidth="1.6" strokeLinecap="round" />
        <circle cx="16" cy="16" r="2.6" fill="#22D3EE" />
      </svg>
      {withText && (
        <span className="font-semibold tracking-tight text-slate-50">
          VisionInspect<span className="text-signal"> AI</span>
        </span>
      )}
    </span>
  )
}
