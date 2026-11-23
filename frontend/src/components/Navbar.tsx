"use client";

import React from "react";
import { useAuth } from "@/context/AuthContext";
import { LogOut, UserCircle, Activity } from "lucide-react";

export default function Navbar() {
  const { user, logout } = useAuth();

  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-slate-800 bg-slate-950/80 px-6 backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-xs font-medium text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
          <Activity className="h-3.5 w-3.5 animate-pulse" />
          <span>Pipeline Ready</span>
        </div>
      </div>

      {user && (
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2.5 text-right">
            <UserCircle className="h-8 w-8 text-slate-400" />
            <div className="hidden sm:block text-left">
              <div className="text-xs font-semibold text-slate-200">
                {user.full_name || user.email}
              </div>
              <div className="text-[11px] text-slate-500">
                {user.email}
              </div>
            </div>
          </div>

          <button
            onClick={logout}
            className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:bg-rose-500/10 hover:text-rose-400 hover:border-rose-500/30"
            title="Sign out"
          >
            <LogOut className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Logout</span>
          </button>
        </div>
      )}
    </header>
  );
}
