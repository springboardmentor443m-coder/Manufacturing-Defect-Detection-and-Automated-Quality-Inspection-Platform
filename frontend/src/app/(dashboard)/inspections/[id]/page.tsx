"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import ProtectedRoute from "@/components/ProtectedRoute";
import { apiRequest, getImageUrl } from "@/lib/api";
import { Inspection, PreprocessingConfig, PreprocessResponse, QualityReport } from "@/types";
import {
  ArrowLeft,
  AlertCircle,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Wrench,
  FileImage,
  Ruler,
  HardDrive,
  Calendar,
  User,
  Tag,
  Maximize2,
  ImageOff,
  Trash2,
  Sliders,
  Sparkles,
  Layers,
  Activity,
  ShieldCheck,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Eye,
  Columns,
  Cpu,
} from "lucide-react";

export default function InspectionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const inspectionId = params.id as string;

  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [showFullImage, setShowFullImage] = useState(false);
  const [fullImageUrl, setFullImageUrl] = useState<string | null>(null);

  // View mode: 'side_by_side' | 'original' | 'processed' | 'edges'
  const [viewMode, setViewMode] = useState<"side_by_side" | "original" | "processed" | "edges">("side_by_side");

  // Preprocessing config state & panel toggle
  const [showPipelineControls, setShowPipelineControls] = useState(false);
  const [isPreprocessing, setIsPreprocessing] = useState(false);
  const [pipelineConfig, setPipelineConfig] = useState<PreprocessingConfig>({
    do_validate: true,
    do_resize: true,
    target_width: 512,
    target_height: 512,
    keep_aspect_ratio: true,
    do_denoise: true,
    denoise_method: "gaussian",
    denoise_kernel_size: 5,
    do_clahe: true,
    clahe_clip_limit: 2.5,
    clahe_grid_size: 8,
    do_edge_detection: false,
    edge_low_threshold: 50,
    edge_high_threshold: 150,
    do_roi_extraction: false,
    roi_mode: "center",
    roi_margin_percent: 5,
  });

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await apiRequest<Inspection>(`/inspections/${inspectionId}`);
        setInspection(data);
      } catch (err: any) {
        setError(err?.message || "Failed to load inspection");
      } finally {
        setLoading(false);
      }
    }
    if (inspectionId) load();
  }, [inspectionId]);

  const handleDelete = async () => {
    if (!confirm(`Delete Inspection #${inspectionId} permanently?`)) return;
    try {
      await apiRequest(`/inspections/${inspectionId}`, { method: "DELETE" });
      router.push("/inspections");
    } catch (err: any) {
      setError(err?.message || "Delete failed");
    }
  };

  const handleRunPreprocessing = async () => {
    if (!inspection) return;
    setIsPreprocessing(true);
    setError(null);
    setSuccessMsg(null);
    try {
      const res = await apiRequest<PreprocessResponse>(`/inspections/${inspection.id}/preprocess`, {
        method: "POST",
        body: JSON.stringify(pipelineConfig),
      });

      // Update inspection with processed details
      setInspection((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          processed_image_path: res.processed_image_path,
          processed_image_url: res.processed_image_url,
          quality_verdict: res.quality_verdict,
          quality_score: res.quality_score,
          quality_report: res.quality_report,
          preprocessing_config: res.pipeline_metadata?.config || pipelineConfig,
        };
      });

      setSuccessMsg("Preprocessing pipeline executed and quality metrics updated!");
      setTimeout(() => setSuccessMsg(null), 5000);
    } catch (err: any) {
      setError(err?.message || "Failed to run preprocessing");
    } finally {
      setIsPreprocessing(false);
    }
  };

  const getDecisionBanner = (decision: string | null) => {
    switch (decision) {
      case "pass":
        return { icon: CheckCircle2, label: "PASS — Cleared for Production", color: "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" };
      case "fail":
        return { icon: XCircle, label: "REJECT — Critical Defect Detected", color: "bg-rose-500/10 border-rose-500/30 text-rose-400" };
      case "rework":
        return { icon: Wrench, label: "REWORK — Route to Refinishing Station", color: "bg-amber-500/10 border-amber-500/30 text-amber-400" };
      case "manual_review":
        return { icon: HelpCircle, label: "MANUAL REVIEW — Operator Inspection Required", color: "bg-indigo-500/10 border-indigo-500/30 text-indigo-400" };
      default:
        return { icon: AlertCircle, label: "PENDING — Awaiting AI Analysis", color: "bg-slate-800 border-slate-700 text-slate-400" };
    }
  };

  const getQualityVerdictBadge = (verdict?: string | null) => {
    switch (verdict) {
      case "pass":
        return {
          icon: ShieldCheck,
          text: "QUALITY PASS",
          badge: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40",
          desc: "Optical metrics meet automated inspection standards.",
        };
      case "warn":
        return {
          icon: ShieldAlert,
          text: "QUALITY WARN",
          badge: "bg-amber-500/20 text-amber-400 border-amber-500/40",
          desc: "Marginal image quality; potential lighting or focus anomalies.",
        };
      case "fail":
        return {
          icon: ShieldAlert,
          text: "QUALITY FAIL",
          badge: "bg-rose-500/20 text-rose-400 border-rose-500/40",
          desc: "Severe optical defects (blur, extreme exposure, or noise) detected.",
        };
      default:
        return {
          icon: HelpCircle,
          text: "NOT ASSESSED",
          badge: "bg-slate-800 text-slate-400 border-slate-700",
          desc: "Run preprocessing pipeline to evaluate optical quality.",
        };
    }
  };

  const getSeverityColor = (score: number) => {
    if (score >= 80) return "text-rose-400";
    if (score >= 60) return "text-amber-400";
    if (score >= 40) return "text-indigo-400";
    return "text-emerald-400";
  };

  const formatBytes = (bytes: number | null | undefined) => {
    if (!bytes) return "—";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(2)} MB`;
  };

  if (loading) {
    return (
      <ProtectedRoute allowedRoles={["quality_engineer", "supervisor", "admin"]}>
        <div className="flex h-[60vh] items-center justify-center">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-sky-500 border-t-transparent" />
        </div>
      </ProtectedRoute>
    );
  }

  if (error && !inspection) {
    return (
      <ProtectedRoute allowedRoles={["quality_engineer", "supervisor", "admin"]}>
        <div className="space-y-4 p-8">
          <div className="flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-300">
            <AlertCircle className="h-5 w-5 shrink-0" />
            <span>{error || "Inspection not found"}</span>
          </div>
          <button onClick={() => router.push("/inspections")} className="text-xs text-sky-400 underline">
            ← Back to Inspections
          </button>
        </div>
      </ProtectedRoute>
    );
  }

  if (!inspection) return null;

  const originalImgSrc = getImageUrl(inspection.image_url || inspection.image_path);
  const processedImgSrc = inspection.processed_image_url
    ? getImageUrl(inspection.processed_image_url)
    : inspection.processed_image_path
    ? getImageUrl(inspection.processed_image_path)
    : null;

  const qualityReport: QualityReport | undefined = inspection.quality_report || undefined;
  const edgePreviewDataUrl = qualityReport?.features?.edge_preview_data_url || null;

  const decision = getDecisionBanner(inspection.decision);
  const DecisionIcon = decision.icon;
  const qualityBadge = getQualityVerdictBadge(inspection.quality_verdict);
  const QualityIcon = qualityBadge.icon;

  const openFullscreen = (url: string | null) => {
    if (!url) return;
    setFullImageUrl(url);
    setShowFullImage(true);
  };

  return (
    <ProtectedRoute allowedRoles={["quality_engineer", "supervisor", "admin"]}>
      <div className="space-y-6">
        {/* Alerts */}
        {error && (
          <div className="flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
        {successMsg && (
          <div className="flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4 text-xs text-emerald-300">
            <CheckCircle2 className="h-4 w-4 shrink-0" />
            <span>{successMsg}</span>
          </div>
        )}

        {/* Header Bar */}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-5">
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.push("/inspections")}
              className="rounded-lg border border-slate-700 bg-slate-800 p-2 text-slate-400 hover:text-white transition"
              title="Back to list"
            >
              <ArrowLeft className="h-4 w-4" />
            </button>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-white">
                  Inspection #{inspection.id}
                </h1>
                <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${qualityBadge.badge}`}>
                  <QualityIcon className="h-3 w-3" />
                  {qualityBadge.text}
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                {inspection.original_filename || inspection.image_path.split("/").pop()}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowPipelineControls(!showPipelineControls)}
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-medium text-slate-200 hover:bg-slate-700 transition"
            >
              <Sliders className="h-3.5 w-3.5 text-sky-400" />
              <span>Pipeline Settings</span>
              {showPipelineControls ? <ChevronUp className="h-3.5 w-3.5 ml-1" /> : <ChevronDown className="h-3.5 w-3.5 ml-1" />}
            </button>

            <button
              onClick={handleRunPreprocessing}
              disabled={isPreprocessing}
              className="flex items-center gap-1.5 rounded-lg bg-sky-500 px-3.5 py-2 text-xs font-semibold text-white shadow-lg shadow-sky-500/20 hover:bg-sky-400 disabled:opacity-50 transition"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isPreprocessing ? "animate-spin" : ""}`} />
              {isPreprocessing ? "Processing..." : "Run Preprocessing"}
            </button>

            <button
              onClick={handleDelete}
              className="flex items-center gap-1.5 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs font-medium text-rose-300 hover:bg-rose-500/20 transition"
              title="Delete inspection"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Top Status Grid: Decision Banner & Optical Quality Score */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Defect Decision Banner */}
          <div className={`flex items-center gap-3 rounded-xl border p-4 ${decision.color}`}>
            <DecisionIcon className="h-7 w-7 shrink-0" />
            <div>
              <p className="text-sm font-bold">{decision.label}</p>
              <p className="text-xs opacity-80 mt-0.5">
                Overall Severity:{" "}
                <span className={`font-bold ${getSeverityColor(inspection.overall_severity)}`}>
                  {inspection.overall_severity.toFixed(1)} / 100
                </span>
                {inspection.defects && inspection.defects.length > 0
                  ? ` • ${inspection.defects.length} defect(s) detected`
                  : " • No defects flagged"}
              </p>
            </div>
          </div>

          {/* Image Quality Score Banner */}
          <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                  Optical Quality Score
                </span>
                <span className={`rounded px-1.5 py-0.2 text-[10px] font-bold uppercase ${qualityBadge.badge}`}>
                  {inspection.quality_verdict || "UNASSESSED"}
                </span>
              </div>
              <div className="text-2xl font-black text-white">
                {inspection.quality_score !== null && inspection.quality_score !== undefined
                  ? `${inspection.quality_score.toFixed(1)} / 100`
                  : "—"}
              </div>
              <p className="text-[11px] text-slate-400">{qualityBadge.desc}</p>
            </div>
            {typeof inspection.quality_score === "number" && (
              <div className="w-24 h-3 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                <div
                  className={`h-full transition-all duration-500 ${
                    inspection.quality_score >= 70
                      ? "bg-emerald-500"
                      : inspection.quality_score >= 45
                      ? "bg-amber-500"
                      : "bg-rose-500"
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, inspection.quality_score))}%` }}
                />
              </div>
            )}
          </div>
        </div>

        {/* Preprocessing Pipeline Config Drawer (Expandable) */}
        {showPipelineControls && (
          <div className="rounded-xl border border-sky-500/30 bg-slate-900/90 p-5 shadow-2xl space-y-4 animate-in fade-in duration-200">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Sliders className="h-4 w-4 text-sky-400" />
                <h3 className="text-sm font-semibold text-white">OpenCV Preprocessing Pipeline Configuration</h3>
              </div>
              <span className="text-[11px] text-slate-400">Non-destructive (original remains untouched)</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
              {/* Step 1: Denoise */}
              <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-2">
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="font-semibold text-slate-200">Noise Reduction</span>
                  <input
                    type="checkbox"
                    checked={pipelineConfig.do_denoise}
                    onChange={(e) => setPipelineConfig({ ...pipelineConfig, do_denoise: e.target.checked })}
                    className="rounded border-slate-700 text-sky-500 focus:ring-sky-500 h-4 w-4"
                  />
                </label>
                {pipelineConfig.do_denoise && (
                  <>
                    <div>
                      <span className="text-slate-400">Method</span>
                      <select
                        value={pipelineConfig.denoise_method}
                        onChange={(e) =>
                          setPipelineConfig({
                            ...pipelineConfig,
                            denoise_method: e.target.value as any,
                          })
                        }
                        className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                      >
                        <option value="gaussian">Gaussian Filter</option>
                        <option value="median">Median Filter</option>
                        <option value="bilateral">Bilateral Filter</option>
                      </select>
                    </div>
                    <div>
                      <span className="text-slate-400">Kernel: {pipelineConfig.denoise_kernel_size}px</span>
                      <input
                        type="range"
                        min="3"
                        max="11"
                        step="2"
                        value={pipelineConfig.denoise_kernel_size}
                        onChange={(e) =>
                          setPipelineConfig({
                            ...pipelineConfig,
                            denoise_kernel_size: parseInt(e.target.value),
                          })
                        }
                        className="w-full accent-sky-400"
                      />
                    </div>
                  </>
                )}
              </div>

              {/* Step 2: CLAHE Contrast */}
              <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-2">
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="font-semibold text-slate-200">CLAHE Contrast</span>
                  <input
                    type="checkbox"
                    checked={pipelineConfig.do_clahe}
                    onChange={(e) => setPipelineConfig({ ...pipelineConfig, do_clahe: e.target.checked })}
                    className="rounded border-slate-700 text-sky-500 focus:ring-sky-500 h-4 w-4"
                  />
                </label>
                {pipelineConfig.do_clahe && (
                  <>
                    <div>
                      <span className="text-slate-400">Clip Limit: {pipelineConfig.clahe_clip_limit}</span>
                      <input
                        type="range"
                        min="1"
                        max="5"
                        step="0.5"
                        value={pipelineConfig.clahe_clip_limit}
                        onChange={(e) =>
                          setPipelineConfig({
                            ...pipelineConfig,
                            clahe_clip_limit: parseFloat(e.target.value),
                          })
                        }
                        className="w-full accent-sky-400"
                      />
                    </div>
                    <div>
                      <span className="text-slate-400">Grid Tile Size: {pipelineConfig.clahe_grid_size}x{pipelineConfig.clahe_grid_size}</span>
                      <select
                        value={pipelineConfig.clahe_grid_size}
                        onChange={(e) =>
                          setPipelineConfig({
                            ...pipelineConfig,
                            clahe_grid_size: parseInt(e.target.value),
                          })
                        }
                        className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                      >
                        <option value="4">4 × 4</option>
                        <option value="8">8 × 8 (Standard)</option>
                        <option value="16">16 × 16</option>
                      </select>
                    </div>
                  </>
                )}
              </div>

              {/* Step 3: Edge Overlay */}
              <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-2">
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="font-semibold text-slate-200">Edge Overlay</span>
                  <input
                    type="checkbox"
                    checked={pipelineConfig.do_edge_detection}
                    onChange={(e) =>
                      setPipelineConfig({
                        ...pipelineConfig,
                        do_edge_detection: e.target.checked,
                      })
                    }
                    className="rounded border-slate-700 text-sky-500 focus:ring-sky-500 h-4 w-4"
                  />
                </label>
                {pipelineConfig.do_edge_detection && (
                  <>
                    <div>
                      <span className="text-slate-400">Low Threshold: {pipelineConfig.edge_low_threshold}</span>
                      <input
                        type="range"
                        min="10"
                        max="100"
                        value={pipelineConfig.edge_low_threshold}
                        onChange={(e) =>
                          setPipelineConfig({
                            ...pipelineConfig,
                            edge_low_threshold: parseInt(e.target.value),
                          })
                        }
                        className="w-full accent-sky-400"
                      />
                    </div>
                    <div>
                      <span className="text-slate-400">High Threshold: {pipelineConfig.edge_high_threshold}</span>
                      <input
                        type="range"
                        min="100"
                        max="250"
                        value={pipelineConfig.edge_high_threshold}
                        onChange={(e) =>
                          setPipelineConfig({
                            ...pipelineConfig,
                            edge_high_threshold: parseInt(e.target.value),
                          })
                        }
                        className="w-full accent-sky-400"
                      />
                    </div>
                  </>
                )}
              </div>

              {/* Step 4: Resize & ROI */}
              <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-2">
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="font-semibold text-slate-200">Target Resize</span>
                  <input
                    type="checkbox"
                    checked={pipelineConfig.do_resize}
                    onChange={(e) => setPipelineConfig({ ...pipelineConfig, do_resize: e.target.checked })}
                    className="rounded border-slate-700 text-sky-500 focus:ring-sky-500 h-4 w-4"
                  />
                </label>
                {pipelineConfig.do_resize && (
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <span className="text-slate-400">Width</span>
                      <input
                        type="number"
                        value={pipelineConfig.target_width}
                        onChange={(e) =>
                          setPipelineConfig({
                            ...pipelineConfig,
                            target_width: parseInt(e.target.value) || 256,
                          })
                        }
                        className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                      />
                    </div>
                    <div>
                      <span className="text-slate-400">Height</span>
                      <input
                        type="number"
                        value={pipelineConfig.target_height}
                        onChange={(e) =>
                          setPipelineConfig({
                            ...pipelineConfig,
                            target_height: parseInt(e.target.value) || 256,
                          })
                        }
                        className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                      />
                    </div>
                  </div>
                )}
                <label className="flex items-center justify-between cursor-pointer pt-1">
                  <span className="text-slate-400">Auto ROI Extraction</span>
                  <input
                    type="checkbox"
                    checked={pipelineConfig.do_roi_extraction}
                    onChange={(e) =>
                      setPipelineConfig({
                        ...pipelineConfig,
                        do_roi_extraction: e.target.checked,
                      })
                    }
                    className="rounded border-slate-700 text-sky-500 focus:ring-sky-500 h-3.5 w-3.5"
                  />
                </label>
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t border-slate-800">
              <button
                onClick={() =>
                  setPipelineConfig({
                    do_validate: true,
                    do_resize: true,
                    target_width: 512,
                    target_height: 512,
                    keep_aspect_ratio: true,
                    do_denoise: true,
                    denoise_method: "gaussian",
                    denoise_kernel_size: 5,
                    do_clahe: true,
                    clahe_clip_limit: 2.5,
                    clahe_grid_size: 8,
                    do_edge_detection: false,
                    edge_low_threshold: 50,
                    edge_high_threshold: 150,
                    do_roi_extraction: false,
                    roi_mode: "center",
                    roi_margin_percent: 5,
                  })
                }
                className="px-3 py-1.5 rounded border border-slate-700 text-xs text-slate-400 hover:text-white"
              >
                Reset to Defaults
              </button>
              <button
                onClick={handleRunPreprocessing}
                disabled={isPreprocessing}
                className="flex items-center gap-1.5 rounded bg-sky-500 px-4 py-1.5 text-xs font-semibold text-white hover:bg-sky-400 disabled:opacity-50"
              >
                <Sparkles className="h-3.5 w-3.5" />
                Apply & Save Processed Version
              </button>
            </div>
          </div>
        )}

        {/* View Switcher Tabs */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <div className="flex items-center gap-1 sm:gap-2">
            <button
              onClick={() => setViewMode("side_by_side")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                viewMode === "side_by_side"
                  ? "bg-sky-500/20 text-sky-400 border border-sky-500/40"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              }`}
            >
              <Columns className="h-3.5 w-3.5" />
              Side-by-Side View
            </button>
            <button
              onClick={() => setViewMode("original")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                viewMode === "original"
                  ? "bg-sky-500/20 text-sky-400 border border-sky-500/40"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              }`}
            >
              <Eye className="h-3.5 w-3.5" />
              Original Capture
            </button>
            <button
              onClick={() => setViewMode("processed")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                viewMode === "processed"
                  ? "bg-sky-500/20 text-sky-400 border border-sky-500/40"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              }`}
            >
              <Sparkles className="h-3.5 w-3.5" />
              Processed OpenCV
            </button>
            {edgePreviewDataUrl && (
              <button
                onClick={() => setViewMode("edges")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                  viewMode === "edges"
                    ? "bg-sky-500/20 text-sky-400 border border-sky-500/40"
                    : "text-slate-400 hover:text-white hover:bg-slate-800"
                }`}
              >
                <Layers className="h-3.5 w-3.5" />
                Edge Feature Map
              </button>
            )}
          </div>
          <span className="text-[11px] text-slate-500 hidden sm:inline">
            Click image to view high-resolution fullscreen
          </span>
        </div>

        {/* Image Display Area */}
        {viewMode === "side_by_side" ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Original Box */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between border-b border-slate-800 px-4 py-2.5 bg-slate-900/80">
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-slate-400" />
                  <span className="text-xs font-semibold text-white">Original Capture</span>
                  <span className="text-[10px] text-slate-400">
                    ({inspection.image_width || "—"} × {inspection.image_height || "—"} px)
                  </span>
                </div>
                <button
                  onClick={() => openFullscreen(originalImgSrc)}
                  className="text-xs text-slate-400 hover:text-white transition flex items-center gap-1"
                >
                  <Maximize2 className="h-3 w-3" />
                </button>
              </div>
              <div className="relative bg-slate-950 flex items-center justify-center min-h-[360px] p-2 flex-1">
                {originalImgSrc ? (
                  <img
                    src={originalImgSrc}
                    alt="Original"
                    className="max-h-[480px] w-auto object-contain cursor-pointer rounded transition hover:opacity-95"
                    onClick={() => openFullscreen(originalImgSrc)}
                  />
                ) : (
                  <div className="flex flex-col items-center gap-2 text-slate-500">
                    <ImageOff className="h-10 w-10" />
                    <span className="text-xs">Original image not available</span>
                  </div>
                )}
              </div>
            </div>

            {/* Processed Box */}
            <div className="rounded-xl border border-sky-500/20 bg-slate-900/60 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between border-b border-slate-800 px-4 py-2.5 bg-slate-900/80">
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-sky-400" />
                  <span className="text-xs font-semibold text-sky-300">OpenCV Preprocessed</span>
                  {inspection.processed_image_path && (
                    <span className="rounded bg-sky-500/20 px-1.5 py-0.2 text-[10px] text-sky-300 font-mono">
                      CLAHE + Denoise
                    </span>
                  )}
                </div>
                {processedImgSrc && (
                  <button
                    onClick={() => openFullscreen(processedImgSrc)}
                    className="text-xs text-slate-400 hover:text-white transition flex items-center gap-1"
                  >
                    <Maximize2 className="h-3 w-3" />
                  </button>
                )}
              </div>
              <div className="relative bg-slate-950 flex items-center justify-center min-h-[360px] p-2 flex-1">
                {processedImgSrc ? (
                  <img
                    src={processedImgSrc}
                    alt="Processed"
                    className="max-h-[480px] w-auto object-contain cursor-pointer rounded transition hover:opacity-95"
                    onClick={() => openFullscreen(processedImgSrc)}
                  />
                ) : (
                  <div className="flex flex-col items-center gap-2 text-slate-500 p-6 text-center">
                    <Sparkles className="h-10 w-10 text-slate-600" />
                    <span className="text-xs text-slate-400">No processed image generated yet.</span>
                    <button
                      onClick={handleRunPreprocessing}
                      disabled={isPreprocessing}
                      className="mt-2 rounded bg-sky-500 px-3 py-1 text-xs font-medium text-white hover:bg-sky-400"
                    >
                      Run Pipeline Now
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : viewMode === "original" ? (
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3 bg-slate-900/80">
              <h3 className="text-sm font-semibold text-white">Original Raw Capture</h3>
              <button
                onClick={() => openFullscreen(originalImgSrc)}
                className="flex items-center gap-1 text-xs text-slate-400 hover:text-white"
              >
                <Maximize2 className="h-3.5 w-3.5" /> Fullscreen
              </button>
            </div>
            <div className="relative bg-slate-950 flex items-center justify-center min-h-[440px] p-4">
              {originalImgSrc ? (
                <img
                  src={originalImgSrc}
                  alt="Original"
                  className="max-h-[600px] w-auto object-contain cursor-pointer rounded"
                  onClick={() => openFullscreen(originalImgSrc)}
                />
              ) : (
                <div className="flex flex-col items-center gap-2 text-slate-500">
                  <ImageOff className="h-12 w-12" />
                  <span className="text-xs">Image unavailable</span>
                </div>
              )}
            </div>
          </div>
        ) : viewMode === "processed" ? (
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3 bg-slate-900/80">
              <h3 className="text-sm font-semibold text-white">OpenCV Preprocessed Version</h3>
              {processedImgSrc && (
                <button
                  onClick={() => openFullscreen(processedImgSrc)}
                  className="flex items-center gap-1 text-xs text-slate-400 hover:text-white"
                >
                  <Maximize2 className="h-3.5 w-3.5" /> Fullscreen
                </button>
              )}
            </div>
            <div className="relative bg-slate-950 flex items-center justify-center min-h-[440px] p-4">
              {processedImgSrc ? (
                <img
                  src={processedImgSrc}
                  alt="Processed"
                  className="max-h-[600px] w-auto object-contain cursor-pointer rounded"
                  onClick={() => openFullscreen(processedImgSrc)}
                />
              ) : (
                <div className="flex flex-col items-center gap-2 text-slate-500 p-8 text-center">
                  <Sparkles className="h-12 w-12 text-slate-600" />
                  <span className="text-xs text-slate-400">No processed image generated yet.</span>
                  <button
                    onClick={handleRunPreprocessing}
                    className="mt-2 rounded bg-sky-500 px-3 py-1.5 text-xs font-semibold text-white"
                  >
                    Run Pipeline
                  </button>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Edge feature map */
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3 bg-slate-900/80">
              <h3 className="text-sm font-semibold text-white">Canny Edge Feature Extraction</h3>
              {edgePreviewDataUrl && (
                <button
                  onClick={() => openFullscreen(edgePreviewDataUrl)}
                  className="flex items-center gap-1 text-xs text-slate-400 hover:text-white"
                >
                  <Maximize2 className="h-3.5 w-3.5" /> Fullscreen
                </button>
              )}
            </div>
            <div className="relative bg-black flex items-center justify-center min-h-[440px] p-4">
              {edgePreviewDataUrl ? (
                <img
                  src={edgePreviewDataUrl}
                  alt="Edge map"
                  className="max-h-[600px] w-auto object-contain cursor-pointer rounded border border-slate-800"
                  onClick={() => openFullscreen(edgePreviewDataUrl)}
                />
              ) : (
                <span className="text-xs text-slate-500">Edge preview unavailable</span>
              )}
            </div>
          </div>
        )}

        {/* Quality Report & Extracted Features Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Quality Analysis Card (2 cols) */}
          <div className="lg:col-span-2 space-y-6">
            {/* Metric Breakdown Cards */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-sky-400" />
                  <h3 className="text-sm font-semibold text-white">Image Optical Quality Report</h3>
                </div>
                <span className="text-xs text-slate-400">
                  Evaluated: {inspection.updated_at ? new Date(inspection.updated_at).toLocaleTimeString() : "Pending"}
                </span>
              </div>

              {/* Quality reasons alert if warn/fail */}
              {qualityReport?.reasons && qualityReport.reasons.length > 0 && (
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 space-y-1">
                  <p className="text-xs font-semibold text-amber-300 flex items-center gap-1.5">
                    <AlertCircle className="h-3.5 w-3.5" /> Quality Audit Observations:
                  </p>
                  <ul className="list-disc list-inside text-xs text-amber-200/80 space-y-0.5">
                    {qualityReport.reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* 4 Metric Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {/* Blur / Laplacian */}
                <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-slate-400">Sharpness / Blur</span>
                    <span
                      className={`text-[10px] font-bold uppercase rounded px-1 ${
                        qualityReport?.metrics_breakdown?.blur?.status === "pass"
                          ? "bg-emerald-500/20 text-emerald-300"
                          : qualityReport?.metrics_breakdown?.blur?.status === "warn"
                          ? "bg-amber-500/20 text-amber-300"
                          : "bg-rose-500/20 text-rose-300"
                      }`}
                    >
                      {qualityReport?.metrics_breakdown?.blur?.status || "—"}
                    </span>
                  </div>
                  <div className="text-base font-bold text-white">
                    {qualityReport?.blur_score !== undefined ? qualityReport.blur_score.toFixed(1) : "—"}
                  </div>
                  <p className="text-[10px] text-slate-500">Laplacian variance</p>
                </div>

                {/* Brightness */}
                <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-slate-400">Brightness</span>
                    <span
                      className={`text-[10px] font-bold uppercase rounded px-1 ${
                        qualityReport?.metrics_breakdown?.brightness?.status === "pass"
                          ? "bg-emerald-500/20 text-emerald-300"
                          : qualityReport?.metrics_breakdown?.brightness?.status === "warn"
                          ? "bg-amber-500/20 text-amber-300"
                          : "bg-rose-500/20 text-rose-300"
                      }`}
                    >
                      {qualityReport?.metrics_breakdown?.brightness?.status || "—"}
                    </span>
                  </div>
                  <div className="text-base font-bold text-white">
                    {qualityReport?.brightness !== undefined ? qualityReport.brightness.toFixed(1) : "—"}
                  </div>
                  <p className="text-[10px] text-slate-500">Mean luminance (0-255)</p>
                </div>

                {/* Contrast */}
                <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-slate-400">Contrast</span>
                    <span
                      className={`text-[10px] font-bold uppercase rounded px-1 ${
                        qualityReport?.metrics_breakdown?.contrast?.status === "pass"
                          ? "bg-emerald-500/20 text-emerald-300"
                          : qualityReport?.metrics_breakdown?.contrast?.status === "warn"
                          ? "bg-amber-500/20 text-amber-300"
                          : "bg-rose-500/20 text-rose-300"
                      }`}
                    >
                      {qualityReport?.metrics_breakdown?.contrast?.status || "—"}
                    </span>
                  </div>
                  <div className="text-base font-bold text-white">
                    {qualityReport?.contrast !== undefined ? qualityReport.contrast.toFixed(1) : "—"}
                  </div>
                  <p className="text-[10px] text-slate-500">RMS Std. Dev</p>
                </div>

                {/* Noise */}
                <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-slate-400">Noise Estimate</span>
                    <span
                      className={`text-[10px] font-bold uppercase rounded px-1 ${
                        qualityReport?.metrics_breakdown?.noise?.status === "pass"
                          ? "bg-emerald-500/20 text-emerald-300"
                          : qualityReport?.metrics_breakdown?.noise?.status === "warn"
                          ? "bg-amber-500/20 text-amber-300"
                          : "bg-rose-500/20 text-rose-300"
                      }`}
                    >
                      {qualityReport?.metrics_breakdown?.noise?.status || "—"}
                    </span>
                  </div>
                  <div className="text-base font-bold text-white">
                    {qualityReport?.noise_estimate !== undefined ? qualityReport.noise_estimate.toFixed(2) : "—"}
                  </div>
                  <p className="text-[10px] text-slate-500">MAD High-freq σ</p>
                </div>
              </div>
            </div>

            {/* Feature Extraction Card */}
            {qualityReport?.features && (
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <Cpu className="h-4 w-4 text-sky-400" />
                    <h3 className="text-sm font-semibold text-white">Extracted Visual Features</h3>
                  </div>
                  <span className="text-[11px] text-slate-400">Texture & Edge Descriptors</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
                  {/* Edges */}
                  <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-2">
                    <h4 className="font-semibold text-sky-300">Edge Statistics</h4>
                    <div className="space-y-1.5 text-slate-300">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Edge Density:</span>
                        <span className="font-mono">{qualityReport.features.edges?.edge_density_pct.toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Gradient Mag:</span>
                        <span className="font-mono">{qualityReport.features.edges?.mean_gradient_magnitude.toFixed(1)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Edge Pixels:</span>
                        <span className="font-mono">{qualityReport.features.edges?.edge_pixel_count.toLocaleString()}</span>
                      </div>
                    </div>
                  </div>

                  {/* Texture */}
                  <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-2">
                    <h4 className="font-semibold text-sky-300">Texture Moments</h4>
                    <div className="space-y-1.5 text-slate-300">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Entropy:</span>
                        <span className="font-mono">{qualityReport.features.texture?.entropy.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Uniformity:</span>
                        <span className="font-mono">{qualityReport.features.texture?.uniformity_energy.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Variance:</span>
                        <span className="font-mono">{qualityReport.features.texture?.variance.toFixed(1)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Skewness:</span>
                        <span className="font-mono">{qualityReport.features.texture?.skewness.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Shape */}
                  <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-2">
                    <h4 className="font-semibold text-sky-300">Shape & Geometry</h4>
                    <div className="space-y-1.5 text-slate-300">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Circularity:</span>
                        <span className="font-mono">{qualityReport.features.shape?.circularity.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Solidity:</span>
                        <span className="font-mono">{qualityReport.features.shape?.solidity.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Aspect Ratio:</span>
                        <span className="font-mono">{qualityReport.features.shape?.aspect_ratio.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Contour Count:</span>
                        <span className="font-mono">{qualityReport.features.shape?.contour_count}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: File Metadata & Inspection Details */}
          <div className="space-y-4">
            {/* Image Metadata */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
              <h3 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">
                Image Metadata
              </h3>
              <div className="space-y-2.5 text-xs">
                <div className="flex items-center gap-2.5">
                  <FileImage className="h-4 w-4 text-slate-500 shrink-0" />
                  <div>
                    <span className="text-slate-400">Filename</span>
                    <p className="font-medium text-white break-all">
                      {inspection.original_filename || "—"}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2.5">
                  <Ruler className="h-4 w-4 text-slate-500 shrink-0" />
                  <div>
                    <span className="text-slate-400">Resolution</span>
                    <p className="font-medium text-white">
                      {inspection.image_width && inspection.image_height
                        ? `${inspection.image_width} × ${inspection.image_height} px`
                        : "—"}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2.5">
                  <HardDrive className="h-4 w-4 text-slate-500 shrink-0" />
                  <div>
                    <span className="text-slate-400">File Size</span>
                    <p className="font-medium text-white">
                      {formatBytes(inspection.file_size_bytes)}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2.5">
                  <Tag className="h-4 w-4 text-slate-500 shrink-0" />
                  <div>
                    <span className="text-slate-400">MIME Type</span>
                    <p className="font-medium text-white font-mono">
                      {inspection.mime_type || "—"}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Inspection Record */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
              <h3 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">
                Inspection Record
              </h3>
              <div className="space-y-2.5 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Status</span>
                  <span className="rounded bg-slate-800 px-2 py-0.5 text-[11px] uppercase text-slate-300">
                    {inspection.status}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Product ID</span>
                  <span className="font-medium text-white">#{inspection.product_id}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Created By</span>
                  <span className="font-medium text-white flex items-center gap-1">
                    <User className="h-3 w-3" /> User #{inspection.created_by}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Created At</span>
                  <span className="font-medium text-white flex items-center gap-1">
                    <Calendar className="h-3 w-3" />
                    {new Date(inspection.created_at).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>

            {/* Defects List */}
            {inspection.defects && inspection.defects.length > 0 && (
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
                <h3 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">
                  Detected Defects ({inspection.defects.length})
                </h3>
                {inspection.defects.map((d) => (
                  <div key={d.id} className="rounded-lg border border-slate-800 bg-slate-800/50 p-3 text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-white uppercase">{d.defect_type}</span>
                      <span className={`font-bold ${getSeverityColor(d.severity_score)}`}>
                        {d.severity_score.toFixed(1)}
                      </span>
                    </div>
                    <div className="text-slate-400">
                      Confidence: {(d.confidence * 100).toFixed(1)}% •
                      Size: {d.size_score.toFixed(0)} •
                      Location: {d.location_score.toFixed(0)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Full-screen Image Overlay */}
        {showFullImage && fullImageUrl && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/95 p-4 cursor-pointer"
            onClick={() => setShowFullImage(false)}
          >
            <img
              src={fullImageUrl}
              alt={`Inspection #${inspection.id} — full resolution`}
              className="max-h-[90vh] max-w-[90vw] object-contain rounded-lg shadow-2xl"
            />
            <button
              onClick={() => setShowFullImage(false)}
              className="absolute top-6 right-6 rounded-full bg-slate-800/80 p-2 text-white hover:bg-slate-700 transition"
            >
              ✕
            </button>
          </div>
        )}
      </div>
    </ProtectedRoute>
  );
}
