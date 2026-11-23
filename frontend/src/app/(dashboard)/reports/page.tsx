"use client";

import React from "react";
import ProtectedRoute from "@/components/ProtectedRoute";
import { FileBarChart2, Download, Printer } from "lucide-react";

export default function ReportsPage() {
  return (
    <ProtectedRoute allowedRoles={["supervisor", "admin"]}>
      <div className="space-y-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded bg-amber-500/20 px-2 py-0.5 text-xs font-semibold text-amber-300 border border-amber-500/30">
                Supervisor / Admin Audits
              </span>
            </div>
            <h1 className="mt-2 text-2xl font-bold tracking-tight text-white">
              Inspection & Quality Audit Reports
            </h1>
            <p className="text-xs text-slate-400">
              Aggregated defect distribution, yield analysis, and compliance audit summaries.
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              disabled
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-medium text-slate-400 opacity-60"
            >
              <Printer className="h-3.5 w-3.5" />
              <span>Print Audit</span>
            </button>
            <button
              disabled
              className="flex items-center gap-1.5 rounded-lg bg-sky-500/50 px-3.5 py-2 text-xs font-semibold text-white opacity-60"
            >
              <Download className="h-4 w-4" />
              <span>Export CSV/PDF</span>
            </button>
          </div>
        </div>

        <div className="rounded-xl border border-dashed border-slate-800 bg-slate-900/30 p-12 text-center">
          <FileBarChart2 className="mx-auto h-12 w-12 text-slate-600" />
          <h3 className="mt-4 text-base font-semibold text-white">
            Quality Audit Reports Pipeline
          </h3>
          <p className="mt-2 max-w-md mx-auto text-xs text-slate-400">
            Reporting and yield charts are being wired in subsequent phases (Recharts integration in Phase 4). Reports table schema and backend models are fully provisioned and ready.
          </p>
        </div>
      </div>
    </ProtectedRoute>
  );
}
