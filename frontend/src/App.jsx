import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { can, useAuth } from './auth.jsx'
import AppShell from './components/AppShell.jsx'
import Logo from './components/Logo.jsx'
import Landing from './pages/Landing.jsx'
import Login from './pages/Login.jsx'
import Dashboard from './pages/Dashboard.jsx'
import Upload from './pages/Upload.jsx'
import Batches from './pages/Batches.jsx'
import BatchDetail from './pages/BatchDetail.jsx'
import Dataset from './pages/Dataset.jsx'
import CameraPage from './pages/Camera.jsx'
import Users from './pages/Users.jsx'

function Protected({ allow = () => true, children }) {
  const { user } = useAuth()
  const location = useLocation()
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />
  if (!allow(user)) return <Navigate to="/app" replace />
  return <AppShell>{children}</AppShell>
}

export default function App() {
  const { loading } = useAuth()
  if (loading) {
    return <div className="grid min-h-screen place-items-center"><div className="animate-pulse"><Logo size={40} /></div></div>
  }
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />
      <Route path="/app" element={<Protected><Dashboard /></Protected>} />
      <Route path="/app/batches" element={<Protected><Batches /></Protected>} />
      <Route path="/app/batches/:id" element={<Protected><BatchDetail /></Protected>} />
      <Route path="/app/upload" element={<Protected allow={can.inspect}><Upload /></Protected>} />
      <Route path="/app/dataset" element={<Protected allow={can.inspect}><Dataset /></Protected>} />
      <Route path="/app/camera" element={<Protected allow={can.inspect}><CameraPage /></Protected>} />
      <Route path="/app/users" element={<Protected allow={can.manageUsers}><Users /></Protected>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
