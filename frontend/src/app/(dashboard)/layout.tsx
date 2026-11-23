"use client";

import React from "react";
import ProtectedRoute from "@/components/ProtectedRoute";
import Sidebar from "@/components/Sidebar";
import Navbar from "@/components/Navbar";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ProtectedRoute>
      <div className="flex min-h-screen bg-slate-950 text-slate-100">
        <Sidebar />
        <div className="flex flex-1 flex-col pl-64">
          <Navbar />
          <main className="flex-1 p-8">{children}</main>
        </div>
      </div>
    </ProtectedRoute>
  );
}
