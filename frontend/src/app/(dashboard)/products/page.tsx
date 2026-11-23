"use client";

import React, { useEffect, useState } from "react";
import ProtectedRoute from "@/components/ProtectedRoute";
import { apiRequest } from "@/lib/api";
import { Product } from "@/types";
import { Boxes, Plus, RefreshCw, AlertCircle } from "lucide-react";

export default function ProductsPage() {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modal
  const [showModal, setShowModal] = useState(false);
  const [name, setName] = useState("");
  const [sku, setSku] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);

  const fetchProducts = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiRequest<Product[]>("/products");
      setProducts(data);
    } catch (err: any) {
      setError(err?.message || "Failed to load products");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProducts();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await apiRequest<Product>("/products", {
        method: "POST",
        body: JSON.stringify({ name, sku, description, is_active: true }),
      });
      setShowModal(false);
      setName("");
      setSku("");
      setDescription("");
      fetchProducts();
    } catch (err: any) {
      setError(err?.message || "Failed to register product");
    } finally {
      setSaving(false);
    }
  };

  return (
    <ProtectedRoute allowedRoles={["supervisor", "admin"]}>
      <div className="space-y-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded bg-amber-500/20 px-2 py-0.5 text-xs font-semibold text-amber-300 border border-amber-500/30">
                Supervisor / Admin Management
              </span>
            </div>
            <h1 className="mt-2 text-2xl font-bold tracking-tight text-white">
              Component & Product Catalog
            </h1>
            <p className="text-xs text-slate-400">
              Registered manufactured parts, SKU definitions, and defect profile associations.
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={fetchProducts}
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              <span>Refresh</span>
            </button>
            <button
              onClick={() => setShowModal(true)}
              className="flex items-center gap-1.5 rounded-lg bg-sky-500 px-3.5 py-2 text-xs font-semibold text-white shadow hover:bg-sky-400 transition"
            >
              <Plus className="h-4 w-4" />
              <span>New Product</span>
            </button>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3.5 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60 shadow">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-300">
              <thead className="border-b border-slate-800 bg-slate-900/90 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="px-6 py-3.5">ID</th>
                  <th className="px-6 py-3.5">Component Name</th>
                  <th className="px-6 py-3.5">SKU</th>
                  <th className="px-6 py-3.5">Description</th>
                  <th className="px-6 py-3.5">Status</th>
                  <th className="px-6 py-3.5">Added</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 font-mono text-xs">
                {loading ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                      Loading product catalog...
                    </td>
                  </tr>
                ) : products.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                      No products registered yet.
                    </td>
                  </tr>
                ) : (
                  products.map((p) => (
                    <tr key={p.id} className="hover:bg-slate-850/50 transition">
                      <td className="px-6 py-4 font-bold text-white">#{p.id}</td>
                      <td className="px-6 py-4 font-sans font-medium text-white">{p.name}</td>
                      <td className="px-6 py-4 text-sky-400">{p.sku}</td>
                      <td className="px-6 py-4 font-sans text-slate-400 max-w-sm truncate">
                        {p.description || "—"}
                      </td>
                      <td className="px-6 py-4">
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-400">
                          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                          Active
                        </span>
                      </td>
                      <td className="px-6 py-4 text-slate-500">
                        {new Date(p.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Modal: New Product */}
        {showModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm">
            <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-2xl">
              <h3 className="text-lg font-bold text-white">Register Component</h3>
              <p className="mt-1 text-xs text-slate-400">
                Add manufactured part specification for line inspection.
              </p>

              <form onSubmit={handleCreate} className="mt-4 space-y-3.5">
                <div>
                  <label className="block text-xs font-medium text-slate-300">
                    Product Name
                  </label>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Brake Disc Cast Iron v4"
                    className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-white focus:border-sky-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300">
                    SKU Code
                  </label>
                  <input
                    type="text"
                    required
                    value={sku}
                    onChange={(e) => setSku(e.target.value)}
                    placeholder="BRK-CAST-04"
                    className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-white focus:border-sky-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300">
                    Description
                  </label>
                  <textarea
                    rows={3}
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Tolerance specifications, critical surfaces, and defect thresholds..."
                    className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-white focus:border-sky-500 focus:outline-none"
                  />
                </div>

                <div className="mt-6 flex items-center justify-end gap-2.5 pt-2">
                  <button
                    type="button"
                    onClick={() => setShowModal(false)}
                    className="rounded-lg border border-slate-700 px-3.5 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={saving}
                    className="rounded-lg bg-sky-500 px-4 py-2 text-xs font-semibold text-white shadow hover:bg-sky-400 disabled:opacity-50"
                  >
                    {saving ? "Saving..." : "Save Product"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </ProtectedRoute>
  );
}
