"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { apiRequest } from "@/lib/api";
import { AuthResponse, User, UserRole } from "@/types";

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string, role?: string) => Promise<void>;
  logout: () => void;
  hasRole: (...roles: UserRole[]) => boolean;
  isAdmin: boolean;
  isSupervisor: boolean;
  isQualityEngineer: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  // Load user from stored token on initial mount
  useEffect(() => {
    async function loadUser() {
      const storedToken = localStorage.getItem("token");
      if (!storedToken) {
        setIsLoading(false);
        return;
      }

      setToken(storedToken);
      try {
        const userData = await apiRequest<User>("/auth/me", { token: storedToken });
        setUser(userData);
      } catch (err) {
        console.warn("Failed to validate active session token, clearing credentials.");
        localStorage.removeItem("token");
        setToken(null);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    }

    loadUser();
  }, []);

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    try {
      const res = await apiRequest<AuthResponse>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });

      localStorage.setItem("token", res.access_token);
      setToken(res.access_token);

      // Fetch full user profile
      const userData = await apiRequest<User>("/auth/me", { token: res.access_token });
      setUser(userData);
      router.push("/dashboard");
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (email: string, password: string, fullName?: string, role?: string) => {
    setIsLoading(true);
    try {
      await apiRequest<User>("/auth/register", {
        method: "POST",
        body: JSON.stringify({
          email,
          password,
          full_name: fullName,
          role: role || "quality_engineer",
        }),
      });

      // Auto-login after registration
      await login(email, password);
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("token");
    setToken(null);
    setUser(null);
    router.push("/login");
  };

  const hasRole = (...roles: UserRole[]): boolean => {
    if (!user) return false;
    return roles.includes(user.role);
  };

  const isAdmin = user?.role === "admin";
  const isSupervisor = user?.role === "supervisor" || isAdmin;
  const isQualityEngineer = user?.role === "quality_engineer" || isSupervisor;

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isLoading,
        login,
        register,
        logout,
        hasRole,
        isAdmin,
        isSupervisor,
        isQualityEngineer,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
