"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { UserRole } from "@/types";
import {
  LayoutDashboard,
  ScanEye,
  Upload,
  Boxes,
  Users,
  FileBarChart2,
  Cpu,
} from "lucide-react";

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  roles: UserRole[];
}

const navigation: NavItem[] = [
  {
    name: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
    roles: ["quality_engineer", "supervisor", "admin"],
  },
  {
    name: "Inspections",
    href: "/inspections",
    icon: ScanEye,
    roles: ["quality_engineer", "supervisor", "admin"],
  },
  {
    name: "Upload Images",
    href: "/upload",
    icon: Upload,
    roles: ["quality_engineer", "supervisor", "admin"],
  },
  {
    name: "Products",
    href: "/products",
    icon: Boxes,
    roles: ["supervisor", "admin"],
  },
  {
    name: "User Management",
    href: "/users",
    icon: Users,
    roles: ["admin"], // ADMIN ONLY!
  },
  {
    name: "Reports",
    href: "/reports",
    icon: FileBarChart2,
    roles: ["supervisor", "admin"],
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { user } = useAuth();

  const getRoleBadge = (role: string) => {
    switch (role) {
      case "admin":
        return "bg-purple-500/10 text-purple-400 border-purple-500/30";
      case "supervisor":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      case "quality_engineer":
      default:
        return "bg-sky-500/10 text-sky-400 border-sky-500/30";
    }
  };

  const formatRoleName = (role: string) => {
    return role.replace(/_/g, " ").toUpperCase();
  };

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-64 flex-col border-r border-slate-800 bg-slate-950 px-4 py-6 text-slate-300">
      {/* Brand Header */}
      <div className="flex items-center gap-3 px-2 pb-6 border-b border-slate-800">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-sky-500/10 ring-1 ring-sky-500/30">
          <Cpu className="h-6 w-6 text-sky-400" />
        </div>
        <div>
          <h1 className="text-base font-bold tracking-tight text-white">
            VisionInspect <span className="text-sky-400">AI</span>
          </h1>
          <p className="text-xs text-slate-500 font-mono">Defect Detection v1.0</p>
        </div>
      </div>

      {/* Current User Role Pill */}
      {user && (
        <div className="my-4 rounded-lg bg-slate-900 p-3 border border-slate-800">
          <div className="text-[10px] font-semibold tracking-wider text-slate-500 uppercase">
            Active Role
          </div>
          <div className="mt-1 flex items-center justify-between">
            <span
              className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold ${getRoleBadge(
                user.role
              )}`}
            >
              {formatRoleName(user.role)}
            </span>
            <span className="text-xs text-slate-400 truncate max-w-[100px]">
              {user.email.split("@")[0]}
            </span>
          </div>
        </div>
      )}

      {/* Dynamic Role-Aware Navigation Menu */}
      <nav className="flex-1 space-y-1.5 pt-2">
        <div className="px-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
          Inspection Workspace
        </div>
        {navigation.map((item) => {
          // Check role visibility
          const isVisible = user && item.roles.includes(user.role);
          if (!isVisible) return null;

          const isActive = pathname === item.href;
          const Icon = item.icon;

          return (
            <Link
              key={item.name}
              href={item.href}
              className={`group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition ${
                isActive
                  ? "bg-sky-500/15 text-sky-400 border border-sky-500/30"
                  : "text-slate-400 hover:bg-slate-900 hover:text-slate-100"
              }`}
            >
              <Icon
                className={`h-5 w-5 transition ${
                  isActive ? "text-sky-400" : "text-slate-500 group-hover:text-slate-300"
                }`}
              />
              {item.name}
              {item.name === "User Management" && (
                <span className="ml-auto rounded bg-purple-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-purple-300">
                  Admin
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="border-t border-slate-800 pt-4 text-center">
        <p className="text-[11px] text-slate-600">
          Phase 2: Image Acquisition
        </p>
      </div>
    </aside>
  );
}
