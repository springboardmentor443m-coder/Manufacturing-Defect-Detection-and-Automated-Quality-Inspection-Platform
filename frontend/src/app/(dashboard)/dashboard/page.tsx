"use client";

import React, { useEffect, useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { apiRequest } from "@/lib/api";
import {
  ShieldCheck,
  AlertTriangle,
  Layers,
  Cpu,
  ArrowRight,
  Sparkles,
  CheckCircle,
} from "lucide-react";
import Link from "next/link";

export default function DashboardOverview() {
  const { user, isAdmin, isSupervisor } = useAuth();
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchHealth() {
      try {
        const data = await apiRequest("/health");
        setHealth(data);
      } catch (e) {
        console.error("Health check error", e);
      } finally {
        setLoading(false);
      }
    }
    fetchHealth();
  }, []);

  return (
    <div className="space-y-8">
      {/* Top Banner */}
      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Manufacturing Quality Intelligence
          </h1>
          <p className="text-sm text-slate-400">
            Real-time defect segmentation, anomaly detection, and automated grading.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-mono text-slate-300">
            Active User: <span className="text-sky-400 font-semibold">{user?.email}</span>
          </span>
          <span className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-mono uppercase text-amber-400">
            {user?.role}
          </span>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              API Status
            </span>
            <div className="h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
          </div>
          <div className="mt-3 text-xl font-bold text-emerald-400">
            {loading ? "Checking..." : health?.status === "healthy" ? "OPERATIONAL" : "DEGRADED"}
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono">
            DB: {health?.database || "Connected"}
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Pipeline Stage
            </span>
            <Cpu className="h-5 w-5 text-sky-400" />
          </div>
          <div className="mt-3 text-xl font-bold text-white">Phase 1</div>
          <div className="mt-1 text-xs text-slate-500 font-mono">
            Auth, RBAC & Core Schema
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Inspection Rule
            </span>
            <Layers className="h-5 w-5 text-indigo-400" />
          </div>
          <div className="mt-3 text-xl font-bold text-white">4-Tier Scoring</div>
          <div className="mt-1 text-xs text-slate-500">
            Size • Location • Class • Conf
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Active Datasets
            </span>
            <Sparkles className="h-5 w-5 text-purple-400" />
          </div>
          <div className="mt-3 text-xl font-bold text-white">MVTec + NEU</div>
          <div className="mt-1 text-xs text-slate-500">
            Anomaly & Surface Defect
          </div>
        </div>
      </div>

      {/* Severity Rubric & Decision Matrix */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h3 className="text-base font-semibold text-white">
                Defect Severity & Automated Decision Matrix
              </h3>
              <p className="text-xs text-slate-400">
                Formula: <code className="text-sky-300">Severity = Size×0.30 + Location×0.25 + DefectType×0.25 + Confidence×0.20</code>
              </p>
            </div>
            <ShieldCheck className="h-5 w-5 text-sky-400" />
          </div>

          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-rose-300">
                  Critical (80 - 100)
                </span>
                <span className="rounded bg-rose-500 px-2 py-0.5 text-[10px] font-bold text-white">
                  REJECT
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-300">
                Immediate component rejection. Unacceptable structural or cosmetic flaw.
              </p>
            </div>

            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-amber-300">
                  High (60 - 79)
                </span>
                <span className="rounded bg-amber-500 px-2 py-0.5 text-[10px] font-bold text-slate-950">
                  REWORK
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-300">
                Reroute component to secondary refinishing, buffing, or soldering station.
              </p>
            </div>

            <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-indigo-300">
                  Medium (40 - 59)
                </span>
                <span className="rounded bg-indigo-500 px-2 py-0.5 text-[10px] font-bold text-white">
                  MANUAL REVIEW
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-300">
                Requires manual operator inspection. Also triggered if confidence &lt; 70%.
              </p>
            </div>

            <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-300">
                  Low (0 - 39)
                </span>
                <span className="rounded bg-emerald-500 px-2 py-0.5 text-[10px] font-bold text-white">
                  PASS
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-300">
                Passes all tolerance checks. Cleared for line progression or packaging.
              </p>
            </div>
          </div>
        </div>

        {/* Role Privileges Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-semibold text-white">
              Role Access Matrix
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Your assigned privileges as <span className="font-semibold text-sky-400">{user?.role}</span>
            </p>

            <div className="mt-4 space-y-3">
              <div className="flex items-start gap-2.5 text-xs">
                <CheckCircle className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-medium text-slate-200">Inspections & Defect Logs:</span>
                  <p className="text-slate-400">Available to Quality Engineers, Supervisors, Admins.</p>
                </div>
              </div>

              <div className="flex items-start gap-2.5 text-xs">
                {isSupervisor ? (
                  <CheckCircle className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-slate-600 shrink-0 mt-0.5" />
                )}
                <div>
                  <span className={`font-medium ${isSupervisor ? "text-slate-200" : "text-slate-500"}`}>
                    Product & Catalog Management:
                  </span>
                  <p className="text-slate-500">Requires Supervisor or Admin privileges.</p>
                </div>
              </div>

              <div className="flex items-start gap-2.5 text-xs">
                {isAdmin ? (
                  <CheckCircle className="h-4 w-4 text-purple-400 shrink-0 mt-0.5" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-slate-600 shrink-0 mt-0.5" />
                )}
                <div>
                  <span className={`font-medium ${isAdmin ? "text-purple-300" : "text-slate-500"}`}>
                    User Management & Role Assignment:
                  </span>
                  <p className="text-slate-500">
                    Restricted strictly to System Administrators.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-800">
            {isAdmin ? (
              <Link
                href="/users"
                className="flex items-center justify-between rounded-lg bg-purple-500/10 px-3.5 py-2 text-xs font-semibold text-purple-300 border border-purple-500/30 hover:bg-purple-500/20 transition"
              >
                <span>Open User Management</span>
                <ArrowRight className="h-4 w-4" />
              </Link>
            ) : (
              <div className="text-[11px] text-slate-500 italic">
                User Management menu is hidden and locked for your role.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
