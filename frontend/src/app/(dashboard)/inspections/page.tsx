"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import ProtectedRoute from "@/components/ProtectedRoute";
import { apiRequest, getImageUrl } from "@/lib/api";
import { Inspection, InspectionListResponse, Product } from "@/types";
import {
  ScanEye,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Wrench,
  ChevronLeft,
  ChevronRight,
  Trash2,
  Eye,
  Search,
  Filter,
  ImageOff,
} from "lucide-react";

export default function InspectionsPage() {
  const [data, setData] = useState<InspectionListResponse | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters & pagination
  const [page, setPage] = useState(1);
  const [pageSize] = useState(12);
  const [statusFilter, setStatusFilter] = useState("");
  const [decisionFilter, setDecisionFilter] = useState("");
  const [productFilter, setProductFilter] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  const fetchInspections = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      params.set("page", String(page));
      params.set("page_size", String(pageSize));
      if (statusFilter) params.set("status", statusFilter);
      if (decisionFilter) params.set("decision", decisionFilter);
      if (productFilter) params.set("product_id", productFilter);
      if (searchTerm.trim()) params.set("search", searchTerm.trim());

      const result = await apiRequest<InspectionListResponse>(
        `/inspections?${params.toString()}`
      );
      setData(result);
    } catch (err: any) {
      setError(err?.message || "Failed to load inspections");
    } finally {
      setLoading(false);
    }
  };

  const fetchProducts = async () => {
    try {
      const p = await apiRequest<Product[]>("/products");
      setProducts(p);
    } catch {}
  };

  useEffect(() => {
    fetchProducts();
  }, []);

  useEffect(() => {
    fetchInspections();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, statusFilter, decisionFilter, productFilter]);

  const handleSearch = () => {
    setPage(1);
    fetchInspections();
  };

  const handleDelete = async (id: number) => {
    if (!confirm(`Delete Inspection #${id} and its stored image?`)) return;
    try {
      await apiRequest(`/inspections/${id}`, { method: "DELETE" });
      fetchInspections();
    } catch (err: any) {
      setError(err?.message || "Delete failed");
    }
  };

  const getDecisionBadge = (decision: string | null) => {
    switch (decision) {
      case "pass":
        return (
          <span className="inline-flex items-center gap-1 rounded bg-emerald-500/10 px-2 py-0.5 text-[11px] font-semibold text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 className="h-3 w-3" /> PASS
          </span>
        );
      case "fail":
        return (
          <span className="inline-flex items-center gap-1 rounded bg-rose-500/10 px-2 py-0.5 text-[11px] font-semibold text-rose-400 border border-rose-500/20">
            <XCircle className="h-3 w-3" /> REJECT
          </span>
        );
      case "rework":
        return (
          <span className="inline-flex items-center gap-1 rounded bg-amber-500/10 px-2 py-0.5 text-[11px] font-semibold text-amber-400 border border-amber-500/20">
            <Wrench className="h-3 w-3" /> REWORK
          </span>
        );
      case "manual_review":
        return (
          <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 px-2 py-0.5 text-[11px] font-semibold text-indigo-400 border border-indigo-500/20">
            <HelpCircle className="h-3 w-3" /> REVIEW
          </span>
        );
      default:
        return (
          <span className="rounded bg-slate-800 px-2 py-0.5 text-[11px] text-slate-400">
            PENDING
          </span>
        );
    }
  };

  const getStatusDot = (status: string) => {
    const colors: Record<string, string> = {
      pending: "bg-slate-400",
      processing: "bg-amber-400 animate-pulse",
      completed: "bg-emerald-400",
      failed: "bg-rose-400",
    };
    return colors[status] || "bg-slate-500";
  };

  const getSeverityColor = (score: number) => {
    if (score >= 80) return "text-rose-400";
    if (score >= 60) return "text-amber-400";
    if (score >= 40) return "text-indigo-400";
    return "text-emerald-400";
  };

  const inspections = data?.items || [];
  const totalPages = data?.total_pages || 1;
  const total = data?.total || 0;

  return (
    <ProtectedRoute allowedRoles={["quality_engineer", "supervisor", "admin"]}>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-5">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              Inspection Registry
            </h1>
            <p className="text-xs text-slate-400">
              {total} total inspection records • Real-time thumbnails, defect scores, and triage actions.
            </p>
          </div>
          <button
            onClick={fetchInspections}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>

        {/* Filters Bar */}
        <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400">
            <Filter className="h-3.5 w-3.5" /> Filters
          </div>

          <select
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
            className="rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs text-white focus:border-sky-500 focus:outline-none"
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="processing">Processing</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>

          <select
            value={decisionFilter}
            onChange={(e) => { setDecisionFilter(e.target.value); setPage(1); }}
            className="rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs text-white focus:border-sky-500 focus:outline-none"
          >
            <option value="">All Decisions</option>
            <option value="pass">Pass</option>
            <option value="fail">Reject</option>
            <option value="rework">Rework</option>
            <option value="manual_review">Manual Review</option>
          </select>

          <select
            value={productFilter}
            onChange={(e) => { setProductFilter(e.target.value); setPage(1); }}
            className="rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs text-white focus:border-sky-500 focus:outline-none"
          >
            <option value="">All Products</option>
            {products.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>

          <div className="flex items-center gap-1.5 flex-1 min-w-[200px] max-w-sm">
            <input
              type="text"
              placeholder="Search by filename…"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              className="flex-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:border-sky-500 focus:outline-none"
            />
            <button
              onClick={handleSearch}
              className="rounded-lg bg-slate-800 p-1.5 text-slate-400 hover:text-white border border-slate-700"
            >
              <Search className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3.5 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Grid of Inspection Cards with Thumbnails */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
          </div>
        ) : inspections.length === 0 ? (
          <div className="rounded-xl border border-dashed border-slate-800 bg-slate-900/30 p-12 text-center">
            <ScanEye className="mx-auto h-12 w-12 text-slate-600" />
            <h3 className="mt-4 text-base font-semibold text-white">No inspections found</h3>
            <p className="mt-2 text-xs text-slate-400">
              Upload images from the Upload page or run the camera simulation script.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {inspections.map((insp) => {
              const imgSrc = getImageUrl(insp.image_url || insp.image_path);
              return (
                <div
                  key={insp.id}
                  className="group relative overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60 transition hover:border-slate-700"
                >
                  {/* Thumbnail */}
                  <div className="relative h-40 w-full bg-slate-800 overflow-hidden">
                    {imgSrc ? (
                      <img
                        src={imgSrc}
                        alt={insp.original_filename || `Inspection #${insp.id}`}
                        className="h-full w-full object-cover transition group-hover:scale-105"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = "none";
                          (e.target as HTMLImageElement).nextElementSibling?.classList.remove("hidden");
                        }}
                      />
                    ) : null}
                    <div className={`${imgSrc ? "hidden" : ""} absolute inset-0 flex items-center justify-center bg-slate-800`}>
                      <ImageOff className="h-8 w-8 text-slate-600" />
                    </div>
                    {/* Overlay actions */}
                    <div className="absolute inset-0 flex items-center justify-center gap-2 bg-slate-950/60 opacity-0 transition group-hover:opacity-100">
                      <Link
                        href={`/inspections/${insp.id}`}
                        className="rounded-lg bg-sky-500 px-3 py-1.5 text-xs font-semibold text-white shadow hover:bg-sky-400 transition"
                      >
                        <Eye className="inline h-3.5 w-3.5 mr-1" />
                        View
                      </Link>
                      <button
                        onClick={() => handleDelete(insp.id)}
                        className="rounded-lg bg-rose-500/80 px-3 py-1.5 text-xs font-semibold text-white shadow hover:bg-rose-500 transition"
                      >
                        <Trash2 className="inline h-3.5 w-3.5 mr-1" />
                        Delete
                      </button>
                    </div>
                    {/* ID Badge */}
                    <span className="absolute top-2 left-2 rounded bg-slate-950/80 px-2 py-0.5 text-[11px] font-bold text-white">
                      #{insp.id}
                    </span>
                    {/* Status dot */}
                    <span className={`absolute top-2 right-2 h-2.5 w-2.5 rounded-full ${getStatusDot(insp.status)} ring-2 ring-slate-900`} />
                  </div>

                  {/* Card Body */}
                  <div className="p-3 space-y-2">
                    <p className="truncate text-xs font-medium text-white">
                      {insp.original_filename || insp.image_path.split("/").pop() || "Unknown"}
                    </p>

                    <div className="flex items-center justify-between">
                      {getDecisionBadge(insp.decision)}
                      <span className={`text-xs font-bold font-mono ${getSeverityColor(insp.overall_severity)}`}>
                        {insp.overall_severity.toFixed(1)}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-slate-500">
                      <span>
                        {insp.image_width && insp.image_height
                          ? `${insp.image_width}×${insp.image_height}px`
                          : "—"}
                      </span>
                      <span>{new Date(insp.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-3 pt-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-slate-300 hover:bg-slate-700 disabled:opacity-40"
            >
              <ChevronLeft className="h-3.5 w-3.5" /> Previous
            </button>
            <span className="text-xs text-slate-400">
              Page <strong className="text-white">{page}</strong> of{" "}
              <strong className="text-white">{totalPages}</strong>
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-slate-300 hover:bg-slate-700 disabled:opacity-40"
            >
              Next <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </div>
    </ProtectedRoute>
  );
}
