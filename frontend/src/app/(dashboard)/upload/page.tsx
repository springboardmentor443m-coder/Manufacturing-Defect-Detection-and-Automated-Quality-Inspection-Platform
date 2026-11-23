"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import ProtectedRoute from "@/components/ProtectedRoute";
import { apiRequest, apiUpload } from "@/lib/api";
import { BatchUploadResponse, Inspection, Product } from "@/types";
import {
  Upload,
  CloudUpload,
  X,
  FileImage,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Trash2,
} from "lucide-react";

type FileUploadState = {
  file: File;
  preview: string;
  progress: number;
  status: "queued" | "uploading" | "success" | "error";
  error?: string;
  inspection?: Inspection;
};

export default function UploadPage() {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<number>(0);
  const [files, setFiles] = useState<FileUploadState[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    apiRequest<Product[]>("/products")
      .then((data) => {
        setProducts(data);
        if (data.length > 0) setSelectedProductId(data[0].id);
      })
      .catch(() => {});
  }, []);

  const addFiles = useCallback((newFiles: FileList | File[]) => {
    const incoming = Array.from(newFiles);
    const states: FileUploadState[] = incoming.map((f) => ({
      file: f,
      preview: URL.createObjectURL(f),
      progress: 0,
      status: "queued" as const,
    }));
    setFiles((prev) => [...prev, ...states]);
  }, []);

  const removeFile = (index: number) => {
    setFiles((prev) => {
      URL.revokeObjectURL(prev[index].preview);
      return prev.filter((_, i) => i !== index);
    });
  };

  const clearAll = () => {
    files.forEach((f) => URL.revokeObjectURL(f.preview));
    setFiles([]);
  };

  // --- Drag & Drop handlers ---
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  };

  // --- Upload logic ---
  const uploadAll = async () => {
    if (files.length === 0 || selectedProductId === 0) return;
    setIsUploading(true);

    // Use batch endpoint for multiple files
    const queuedFiles = files.filter((f) => f.status === "queued" || f.status === "error");

    if (queuedFiles.length === 0) {
      setIsUploading(false);
      return;
    }

    // Upload one-by-one to show individual progress
    for (let i = 0; i < files.length; i++) {
      const fs = files[i];
      if (fs.status !== "queued" && fs.status !== "error") continue;

      // Mark as uploading
      setFiles((prev) =>
        prev.map((f, idx) =>
          idx === i ? { ...f, status: "uploading" as const, progress: 0, error: undefined } : f
        )
      );

      const formData = new FormData();
      formData.append("file", fs.file);
      formData.append("product_id", String(selectedProductId));

      try {
        const result = await apiUpload<Inspection>("/inspections/upload", formData, {
          onProgress: (pct) => {
            setFiles((prev) =>
              prev.map((f, idx) =>
                idx === i ? { ...f, progress: pct } : f
              )
            );
          },
        });

        setFiles((prev) =>
          prev.map((f, idx) =>
            idx === i
              ? { ...f, status: "success" as const, progress: 100, inspection: result }
              : f
          )
        );
      } catch (err: any) {
        setFiles((prev) =>
          prev.map((f, idx) =>
            idx === i
              ? { ...f, status: "error" as const, error: err?.message || "Upload failed" }
              : f
          )
        );
      }
    }
    setIsUploading(false);
  };

  const successCount = files.filter((f) => f.status === "success").length;
  const errorCount = files.filter((f) => f.status === "error").length;
  const queuedCount = files.filter((f) => f.status === "queued").length;

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(2)} MB`;
  };

  return (
    <ProtectedRoute allowedRoles={["quality_engineer", "supervisor", "admin"]}>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-5">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              Image Acquisition & Upload
            </h1>
            <p className="text-xs text-slate-400">
              Upload manufacturing inspection images for AI-powered defect analysis.
              Accepted: JPG, PNG, BMP, TIFF, WebP (max 20 MB each).
            </p>
          </div>
          <div className="flex items-center gap-2.5">
            {files.length > 0 && (
              <button
                onClick={clearAll}
                className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-medium text-slate-300 hover:bg-slate-700"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Clear All
              </button>
            )}
          </div>
        </div>

        {/* Product Selector */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
            Target Product / Component
          </label>
          <select
            value={selectedProductId}
            onChange={(e) => setSelectedProductId(Number(e.target.value))}
            className="mt-2 block w-full max-w-md rounded-lg border border-slate-700 bg-slate-800 px-3.5 py-2.5 text-sm text-white focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
          >
            {products.length === 0 && <option value={0}>Loading products…</option>}
            {products.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.sku})
              </option>
            ))}
          </select>
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 text-center transition-all ${
            isDragging
              ? "border-sky-400 bg-sky-500/10 scale-[1.01]"
              : "border-slate-700 bg-slate-900/40 hover:border-slate-500 hover:bg-slate-900/60"
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            multiple
            accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif,.webp"
            onChange={(e) => {
              if (e.target.files) addFiles(e.target.files);
              e.target.value = "";
            }}
            className="hidden"
          />

          <div
            className={`rounded-full p-4 transition ${
              isDragging ? "bg-sky-500/20" : "bg-slate-800"
            }`}
          >
            <CloudUpload
              className={`h-10 w-10 transition ${
                isDragging ? "text-sky-400" : "text-slate-500"
              }`}
            />
          </div>
          <h3 className="mt-4 text-base font-semibold text-white">
            {isDragging ? "Release to upload images" : "Drag & drop inspection images here"}
          </h3>
          <p className="mt-1 text-xs text-slate-400">
            or click to browse files • Multi-select supported
          </p>
          <p className="mt-2 inline-flex items-center gap-1 rounded-full border border-slate-700 bg-slate-800 px-3 py-1 text-[11px] font-mono text-slate-400">
            <FileImage className="h-3 w-3" /> JPG · PNG · BMP · TIFF · WebP
          </p>
        </div>

        {/* File Queue */}
        {files.length > 0 && (
          <div className="space-y-4">
            {/* Summary Bar */}
            <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/60 px-5 py-3">
              <div className="flex items-center gap-4 text-xs font-medium text-slate-300">
                <span>Total: <strong className="text-white">{files.length}</strong></span>
                {queuedCount > 0 && (
                  <span className="text-slate-400">Queued: {queuedCount}</span>
                )}
                {successCount > 0 && (
                  <span className="text-emerald-400">✓ Uploaded: {successCount}</span>
                )}
                {errorCount > 0 && (
                  <span className="text-rose-400">✕ Failed: {errorCount}</span>
                )}
              </div>
              <button
                onClick={uploadAll}
                disabled={isUploading || queuedCount === 0 || selectedProductId === 0}
                className="flex items-center gap-1.5 rounded-lg bg-sky-500 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-sky-500/20 transition hover:bg-sky-400 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isUploading ? (
                  <>
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    Uploading…
                  </>
                ) : (
                  <>
                    <Upload className="h-3.5 w-3.5" />
                    Upload {queuedCount > 0 ? `${queuedCount} file${queuedCount > 1 ? "s" : ""}` : "All"}
                  </>
                )}
              </button>
            </div>

            {/* File Cards Grid */}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {files.map((fs, idx) => (
                <div
                  key={`${fs.file.name}-${idx}`}
                  className={`relative overflow-hidden rounded-xl border bg-slate-900/60 transition ${
                    fs.status === "success"
                      ? "border-emerald-500/30"
                      : fs.status === "error"
                      ? "border-rose-500/30"
                      : fs.status === "uploading"
                      ? "border-sky-500/30"
                      : "border-slate-800"
                  }`}
                >
                  {/* Thumbnail */}
                  <div className="relative h-36 w-full bg-slate-800">
                    <img
                      src={fs.preview}
                      alt={fs.file.name}
                      className="h-full w-full object-cover"
                    />
                    {fs.status === "uploading" && (
                      <div className="absolute inset-0 flex items-center justify-center bg-slate-950/60">
                        <div className="text-center">
                          <div className="h-8 w-8 mx-auto animate-spin rounded-full border-2 border-sky-400 border-t-transparent" />
                          <span className="mt-1 block text-xs font-bold text-sky-300">
                            {fs.progress}%
                          </span>
                        </div>
                      </div>
                    )}
                    {fs.status === "success" && (
                      <div className="absolute top-2 right-2 rounded-full bg-emerald-500 p-1">
                        <CheckCircle2 className="h-4 w-4 text-white" />
                      </div>
                    )}
                    {fs.status === "error" && (
                      <div className="absolute top-2 right-2 rounded-full bg-rose-500 p-1">
                        <AlertCircle className="h-4 w-4 text-white" />
                      </div>
                    )}
                    {/* Remove button */}
                    {fs.status !== "uploading" && (
                      <button
                        onClick={() => removeFile(idx)}
                        className="absolute top-2 left-2 rounded-full bg-slate-950/70 p-1 text-slate-300 hover:bg-slate-800 hover:text-white transition"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>

                  {/* Progress bar */}
                  {fs.status === "uploading" && (
                    <div className="h-1 w-full bg-slate-800">
                      <div
                        className="h-full bg-sky-500 transition-all duration-300"
                        style={{ width: `${fs.progress}%` }}
                      />
                    </div>
                  )}

                  {/* Info */}
                  <div className="p-3">
                    <p className="truncate text-xs font-medium text-white">
                      {fs.file.name}
                    </p>
                    <p className="mt-0.5 text-[11px] text-slate-500 font-mono">
                      {formatSize(fs.file.size)}
                    </p>
                    {fs.status === "error" && fs.error && (
                      <p className="mt-1.5 text-[11px] text-rose-400 leading-tight">
                        {fs.error}
                      </p>
                    )}
                    {fs.status === "success" && fs.inspection && (
                      <p className="mt-1.5 text-[11px] text-emerald-400">
                        Inspection #{fs.inspection.id} •{" "}
                        {fs.inspection.image_width}×{fs.inspection.image_height}px
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </ProtectedRoute>
  );
}
