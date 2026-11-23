"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";

export default function HomePage() {
  const router = useRouter();
  const { user, isLoading } = useAuth();

  useEffect(() => {
    if (!isLoading) {
      if (user) {
        router.push("/dashboard");
      } else {
        router.push("/login");
      }
    }
  }, [user, isLoading, router]);

  return (
    <div className="flex h-screen w-full items-center justify-center bg-slate-950">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-sky-500 border-t-transparent" />
    </div>
  );
}
