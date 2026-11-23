"use client";

import React, { useEffect, useState } from "react";
import ProtectedRoute from "@/components/ProtectedRoute";
import { apiRequest } from "@/lib/api";
import { User, UserRole } from "@/types";
import {
  Users,
  ShieldAlert,
  UserPlus,
  RefreshCw,
  Check,
  AlertCircle,
} from "lucide-react";

export default function UserManagementPage() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // New user modal state
  const [showAddModal, setShowAddModal] = useState(false);
  const [newEmail, setNewEmail] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newName, setNewName] = useState("");
  const [newRole, setNewRole] = useState<UserRole>("quality_engineer");
  const [submitting, setSubmitting] = useState(false);

  const fetchUsers = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiRequest<User[]>("/users");
      setUsers(data);
    } catch (err: any) {
      setError(err?.message || "Failed to load users list");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleRoleChange = async (userId: number, role: UserRole) => {
    setError(null);
    setSuccess(null);
    try {
      await apiRequest<User>(`/users/${userId}/role`, {
        method: "PATCH",
        body: JSON.stringify({ role }),
      });
      setSuccess(`Role updated successfully for user #${userId}`);
      fetchUsers();
    } catch (err: any) {
      setError(err?.message || "Failed to update role");
    }
  };

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setSuccess(null);
    try {
      await apiRequest<User>("/users", {
        method: "POST",
        body: JSON.stringify({
          email: newEmail,
          password: newPassword,
          full_name: newName,
          role: newRole,
        }),
      });
      setSuccess(`User ${newEmail} created successfully!`);
      setShowAddModal(false);
      setNewEmail("");
      setNewPassword("");
      setNewName("");
      fetchUsers();
    } catch (err: any) {
      setError(err?.message || "Failed to create user");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <ProtectedRoute allowedRoles={["admin"]}>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded bg-purple-500/20 px-2 py-0.5 text-xs font-semibold text-purple-300 border border-purple-500/30">
                Administrator Restricted Area
              </span>
            </div>
            <h1 className="mt-2 text-2xl font-bold tracking-tight text-white">
              Operator & User Management
            </h1>
            <p className="text-xs text-slate-400">
              Manage inspector access credentials, plant security levels, and role-based permissions.
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={fetchUsers}
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              <span>Refresh</span>
            </button>
            <button
              onClick={() => setShowAddModal(true)}
              className="flex items-center gap-1.5 rounded-lg bg-sky-500 px-3.5 py-2 text-xs font-semibold text-white shadow hover:bg-sky-400 transition"
            >
              <UserPlus className="h-4 w-4" />
              <span>Add User</span>
            </button>
          </div>
        </div>

        {/* Notifications */}
        {error && (
          <div className="flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3.5 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3.5 text-xs text-emerald-300">
            <Check className="h-4 w-4 shrink-0" />
            <span>{success}</span>
          </div>
        )}

        {/* Users Table */}
        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60 shadow">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-300">
              <thead className="border-b border-slate-800 bg-slate-900/90 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="px-6 py-3.5">User</th>
                  <th className="px-6 py-3.5">Email</th>
                  <th className="px-6 py-3.5">Current Role</th>
                  <th className="px-6 py-3.5">Change Role</th>
                  <th className="px-6 py-3.5">Status</th>
                  <th className="px-6 py-3.5">Registered</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 font-mono text-xs">
                {loading ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                      Loading users list...
                    </td>
                  </tr>
                ) : users.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                      No users found.
                    </td>
                  </tr>
                ) : (
                  users.map((u) => (
                    <tr key={u.id} className="hover:bg-slate-850/50 transition">
                      <td className="px-6 py-4 font-sans font-medium text-white">
                        {u.full_name || "—"}
                      </td>
                      <td className="px-6 py-4 text-slate-300">{u.email}</td>
                      <td className="px-6 py-4">
                        <span
                          className={`inline-flex rounded border px-2 py-0.5 text-[11px] font-semibold uppercase ${
                            u.role === "admin"
                              ? "border-purple-500/30 bg-purple-500/10 text-purple-400"
                              : u.role === "supervisor"
                              ? "border-amber-500/30 bg-amber-500/10 text-amber-400"
                              : "border-sky-500/30 bg-sky-500/10 text-sky-400"
                          }`}
                        >
                          {u.role.replace(/_/g, " ")}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <select
                          value={u.role}
                          onChange={(e) =>
                            handleRoleChange(u.id, e.target.value as UserRole)
                          }
                          className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200 focus:border-sky-500 focus:outline-none"
                        >
                          <option value="quality_engineer">quality_engineer</option>
                          <option value="supervisor">supervisor</option>
                          <option value="admin">admin</option>
                        </select>
                      </td>
                      <td className="px-6 py-4">
                        <span
                          className={`inline-flex items-center gap-1 text-[11px] font-medium ${
                            u.is_active ? "text-emerald-400" : "text-rose-400"
                          }`}
                        >
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${
                              u.is_active ? "bg-emerald-400" : "bg-rose-400"
                            }`}
                          />
                          {u.is_active ? "Active" : "Disabled"}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-slate-500">
                        {new Date(u.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Modal: Add User */}
        {showAddModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm">
            <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-2xl">
              <h3 className="text-lg font-bold text-white">Create New User</h3>
              <p className="mt-1 text-xs text-slate-400">
                Add an inspector, line supervisor, or administrator.
              </p>

              <form onSubmit={handleCreateUser} className="mt-4 space-y-3.5">
                <div>
                  <label className="block text-xs font-medium text-slate-300">
                    Full Name
                  </label>
                  <input
                    type="text"
                    required
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    placeholder="Marcus Reid"
                    className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-white focus:border-sky-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300">
                    Email
                  </label>
                  <input
                    type="email"
                    required
                    value={newEmail}
                    onChange={(e) => setNewEmail(e.target.value)}
                    placeholder="m.reid@plant.com"
                    className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-white focus:border-sky-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300">
                    Password
                  </label>
                  <input
                    type="password"
                    required
                    minLength={6}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Minimum 6 characters"
                    className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-white focus:border-sky-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300">
                    Role
                  </label>
                  <select
                    value={newRole}
                    onChange={(e) => setNewRole(e.target.value as UserRole)}
                    className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-white focus:border-sky-500 focus:outline-none"
                  >
                    <option value="quality_engineer">quality_engineer</option>
                    <option value="supervisor">supervisor</option>
                    <option value="admin">admin</option>
                  </select>
                </div>

                <div className="mt-6 flex items-center justify-end gap-2.5 pt-2">
                  <button
                    type="button"
                    onClick={() => setShowAddModal(false)}
                    className="rounded-lg border border-slate-700 px-3.5 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={submitting}
                    className="rounded-lg bg-sky-500 px-4 py-2 text-xs font-semibold text-white shadow hover:bg-sky-400 disabled:opacity-50"
                  >
                    {submitting ? "Creating..." : "Save User"}
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
