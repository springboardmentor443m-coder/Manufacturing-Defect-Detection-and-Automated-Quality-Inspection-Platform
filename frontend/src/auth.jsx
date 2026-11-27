import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api, getToken, setToken } from './api'

const AuthContext = createContext(null)

export const ROLE_LABELS = {
  admin: 'Admin',
  quality_engineer: 'Quality Engineer',
  supervisor: 'Factory Supervisor',
}

// Mirrors backend/app/core/deps.py
export const can = {
  manageUsers: (u) => u?.role === 'admin',
  inspect: (u) => u?.role === 'admin' || u?.role === 'quality_engineer',
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(!!getToken())

  const logout = useCallback(() => { setToken(null); setUser(null) }, [])

  useEffect(() => {
    if (getToken()) api.me().then(setUser).catch(logout).finally(() => setLoading(false))
    window.addEventListener('vi:logout', logout)
    return () => window.removeEventListener('vi:logout', logout)
  }, [logout])

  const login = async (email, password) => {
    const { access_token } = await api.login(email, password)
    setToken(access_token)
    setUser(await api.me())
  }

  return <AuthContext.Provider value={{ user, loading, login, logout }}>{children}</AuthContext.Provider>
}

export const useAuth = () => useContext(AuthContext)
