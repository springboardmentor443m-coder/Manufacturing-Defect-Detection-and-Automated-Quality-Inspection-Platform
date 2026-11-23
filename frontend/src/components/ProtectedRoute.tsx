"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { UserRole } from "@/types";
import { ShieldAlert } from "lucide-react";

interface ProtectedRouteProps {
  children: React.ReactNode;
  allowedRoles?: UserRole[];
}

export default function ProtectedRoute({
  children,
  allowedRoles,
}: ProtectedRouteProps) {
  const { user, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !user) {
      router.push("/login");
    }
  }, [user, isLoading, router]);

  if (isLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-slate-950 text-slate-200">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-sky-500 border-t-transparent" />
          <p className="text-sm font-medium tracking-wide text-slate-400">
            Authenticating VisionInspect AI session...
          </p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  if (allowedRoles && !allowedRoles.includes(user.role)) {
    return (
      <div className="flex min-h-[70vh] flex-col items-center justify-center p-6 text-center">
        <div className="rounded-full bg-rose-500/10 p-5 ring-1 ring-rose-500/30">
          <ShieldAlert className="h-12 w-12 text-rose-400" />
        </div>
        <h2 className="mt-4 text-2xl font-bold tracking-tight text-white">
          Access Restricted
        </h2>
        <p className="mt-2 max-w-md text-sm text-slate-400">
          You do not have the required permissions to view this resource.
          <br />
          Required role: <span className="font-semibold text-rose-300">{allowedRoles.join(" or ")}</span>.
          <br />
          Your active role: <span className="font-semibold text-amber-300">{user.role}</span>.
        </p>
        <button
          onClick={() => router.push("/dashboard")}
          className="mt-6 rounded-lg bg-slate-800 px-5 py-2.5 text-sm font-medium text-slate-200 ring-1 ring-slate-700 transition hover:bg-slate-700 hover:text-white"
        >
          Return to Dashboard
        </button>
      </div>
    );
  }

  return <>{children}</>;
}
