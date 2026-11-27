/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      colors: {
        ink: { 950: '#07090D', 900: '#0B0F15', 850: '#0F141B', 800: '#131A22', 700: '#1B2430', 600: '#263142', 500: '#3A4658' },
        signal: { DEFAULT: '#22D3EE', soft: '#67E8F9', dim: '#0E7490' },
        pass: '#34D399',
        warn: '#FBBF24',
        fail: '#F43F5E',
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(34,211,238,.25), 0 8px 30px -8px rgba(34,211,238,.35)',
        panel: '0 1px 0 0 rgba(255,255,255,.04) inset, 0 10px 30px -12px rgba(0,0,0,.6)',
      },
      keyframes: {
        scan: { '0%': { transform: 'translateY(-10%)' }, '100%': { transform: 'translateY(110%)' } },
        fadeUp: { '0%': { opacity: 0, transform: 'translateY(8px)' }, '100%': { opacity: 1, transform: 'none' } },
        shimmer: { '100%': { transform: 'translateX(100%)' } },
        pulseRing: { '0%': { transform: 'scale(.9)', opacity: .8 }, '100%': { transform: 'scale(1.6)', opacity: 0 } },
        toastIn: { '0%': { opacity: 0, transform: 'translateY(12px) scale(.98)' }, '100%': { opacity: 1, transform: 'none' } },
      },
      animation: {
        scan: 'scan 3.2s cubic-bezier(.45,0,.55,1) infinite alternate',
        fadeUp: 'fadeUp .45s ease-out both',
        shimmer: 'shimmer 1.4s infinite',
        pulseRing: 'pulseRing 1.6s ease-out infinite',
        toastIn: 'toastIn .25s ease-out both',
      },
    },
  },
  plugins: [],
}
