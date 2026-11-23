"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import { Cpu, AlertCircle, Shield, CheckCircle2 } from "lucide-react";

export default function LoginPage() {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await login(email, password);
    } catch (err: any) {
      setError(err?.message || "Failed to authenticate. Please check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  const handleQuickFill = (demoEmail: string, demoPass: string) => {
    setEmail(demoEmail);
    setPassword(demoPass);
    setError(null);
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-slate-950 px-4 py-12 sm:px-6 lg:px-8">
      {/* Background ambient lighting */}
      <div className="pointer-events-none fixed inset-0 flex items-center justify-center">
        <div className="h-[500px] w-[500px] rounded-full bg-sky-500/10 blur-[120px]" />
      </div>

      <div className="relative z-10 w-full max-w-md space-y-8 rounded-2xl border border-slate-800 bg-slate-900/90 p-8 shadow-2xl backdrop-blur-xl">
        {/* Brand Header */}
        <div className="text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-xl bg-sky-500/10 ring-1 ring-sky-500/30">
            <Cpu className="h-8 w-8 text-sky-400" />
          </div>
          <h2 className="mt-4 text-2xl font-bold tracking-tight text-white">
            VisionInspect <span className="text-sky-400">AI</span>
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Manufacturing Defect Detection & Quality Intelligence
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-300">
              Work Email Address
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="engineer@plant-line.com"
              className="mt-1.5 block w-full rounded-lg border border-slate-700 bg-slate-800/80 px-3.5 py-2.5 text-sm text-white placeholder-slate-500 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300">
              Password
            </label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="mt-1.5 block w-full rounded-lg border border-slate-700 bg-slate-800/80 px-3.5 py-2.5 text-sm text-white placeholder-slate-500 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="flex w-full items-center justify-center rounded-lg bg-sky-500 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-sky-500/20 transition hover:bg-sky-400 focus:outline-none disabled:opacity-50"
          >
            {loading ? (
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
            ) : (
              "Sign In to Workstation"
            )}
          </button>
        </form>

        {/* Quick Fill Demo Roles Box */}
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
          <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <Shield className="h-3.5 w-3.5 text-sky-400" />
            <span>Quick-Login Demo Accounts</span>
          </div>
          <div className="mt-2.5 grid grid-cols-3 gap-2">
            <button
              type="button"
              onClick={() => handleQuickFill("admin@visioninspect.ai", "Admin@123")}
              className="rounded-md border border-purple-500/30 bg-purple-500/10 px-2 py-1.5 text-[11px] font-medium text-purple-300 transition hover:bg-purple-500/20"
            >
              Admin
            </button>
            <button
              type="button"
              onClick={() => handleQuickFill("supervisor@visioninspect.ai", "Super@123")}
              className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1.5 text-[11px] font-medium text-amber-300 transition hover:bg-amber-500/20"
            >
              Supervisor
            </button>
            <button
              type="button"
              onClick={() => handleQuickFill("qe@visioninspect.ai", "Engineer@123")}
              className="rounded-md border border-sky-500/30 bg-sky-500/10 px-2 py-1.5 text-[11px] font-medium text-sky-300 transition hover:bg-sky-500/20"
            >
              Engineer
            </button>
          </div>
        </div>

        {/* Register Link */}
        <p className="text-center text-xs text-slate-400">
          Need a new operator account?{" "}
          <Link
            href="/register"
            className="font-medium text-sky-400 hover:text-sky-300 underline underline-offset-4"
          >
            Register operator
          </Link>
        </p>
      </div>
    </div>
  );
}
