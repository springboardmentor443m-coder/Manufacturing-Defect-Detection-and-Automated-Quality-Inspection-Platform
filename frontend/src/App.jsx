import { useEffect, useState } from 'react'

const API_URL = 'http://localhost:8000'

function App() {
  const [isLogin, setIsLogin] = useState(true)
  const [form, setForm] = useState({
    full_name: '',
    email: 'admin@example.com',
    password: 'admin123',
    role: 'qa_engineer',
  })
  const [token, setToken] = useState(localStorage.getItem('token') || '')
  const [dashboard, setDashboard] = useState(null)
  const [inspections, setInspections] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [result, setResult] = useState(null)
  const [message, setMessage] = useState('')

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  const handleAuth = async (e) => {
    e.preventDefault()
    const endpoint = isLogin ? '/login' : '/register'
    const payload = isLogin
      ? { email: form.email, password: form.password }
      : {
          full_name: form.full_name,
          email: form.email,
          password: form.password,
          role: form.role,
        }

    const response = await fetch(`${API_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    const data = await response.json()
    if (!response.ok) {
      setMessage(data.detail || 'Authentication failed')
      return
    }

    setToken(data.access_token)
    localStorage.setItem('token', data.access_token)
    setMessage('Authentication successful')
    loadDashboard(data.access_token)
  }

  const loadDashboard = async (authToken = token) => {
    if (!authToken) return

    const [dashRes, inspectionsRes] = await Promise.all([
      fetch(`${API_URL}/dashboard`, {
        headers: { Authorization: `Bearer ${authToken}` },
      }),
      fetch(`${API_URL}/inspections`, {
        headers: { Authorization: `Bearer ${authToken}` },
      }),
    ])

    const dashData = await dashRes.json()
    const inspectionData = await inspectionsRes.json()
    setDashboard(dashData)
    setInspections(Array.isArray(inspectionData) ? inspectionData : [])
  }

  useEffect(() => {
    if (token) {
      loadDashboard(token)
    }
  }, [token])

  const handleUpload = async (e) => {
    e.preventDefault()
    if (!selectedFile || !token) {
      setMessage('Please select an image and log in first')
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)

    const uploadResponse = await fetch(`${API_URL}/upload-image`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    })

    const uploadData = await uploadResponse.json()
    if (!uploadResponse.ok) {
      setMessage(uploadData.detail || 'Upload failed')
      return
    }

    const predictResponse = await fetch(`${API_URL}/predict/${uploadData.inspection_id}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
    })

    const predictData = await predictResponse.json()
    if (!predictResponse.ok) {
      setMessage(predictData.detail || 'Prediction failed')
      return
    }

    setResult(predictData)
    setMessage('Inspection complete')
    loadDashboard(token)
  }

  const logout = () => {
    setToken('')
    localStorage.removeItem('token')
    setDashboard(null)
    setInspections([])
    setResult(null)
    setMessage('Logged out')
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>VisionInspect AI</h1>
          <p>Manufacturing Defect Detection Platform</p>
        </div>
        {token && (
          <button className="secondary" onClick={logout}>Logout</button>
        )}
      </header>

      {!token ? (
        <section className="card auth-card">
          <div className="toggle-row">
            <button className={isLogin ? 'active' : ''} onClick={() => setIsLogin(true)}>Login</button>
            <button className={!isLogin ? 'active' : ''} onClick={() => setIsLogin(false)}>Register</button>
          </div>

          <form onSubmit={handleAuth} className="form-grid">
            {!isLogin && (
              <input
                name="full_name"
                placeholder="Full name"
                value={form.full_name}
                onChange={handleChange}
              />
            )}
            <input
              name="email"
              type="email"
              placeholder="Email"
              value={form.email}
              onChange={handleChange}
            />
            <input
              name="password"
              type="password"
              placeholder="Password"
              value={form.password}
              onChange={handleChange}
            />
            {!isLogin && (
              <select name="role" value={form.role} onChange={handleChange}>
                <option value="qa_engineer">QA Engineer</option>
                <option value="admin">Admin</option>
                <option value="manager">Manager</option>
              </select>
            )}
            <button type="submit" className="primary-btn">
              {isLogin ? 'Login' : 'Create Account'}
            </button>
          </form>
        </section>
      ) : (
        <>
          <section className="stats-grid">
            <div className="stat-card">
              <span>Total Inspections</span>
              <strong>{dashboard?.total_inspections ?? 0}</strong>
            </div>
            <div className="stat-card warn">
              <span>Defective</span>
              <strong>{dashboard?.defective_count ?? 0}</strong>
            </div>
            <div className="stat-card ok">
              <span>Normal</span>
              <strong>{dashboard?.normal_count ?? 0}</strong>
            </div>
          </section>

          <section className="card upload-card">
            <h2>Upload Product Image</h2>
            <form onSubmit={handleUpload} className="upload-form">
              <input type="file" accept="image/*" onChange={(e) => setSelectedFile(e.target.files[0])} />
              <button type="submit" className="primary-btn">Run Inspection</button>
            </form>
          </section>

          {result && (
            <section className="card result-card">
              <h2>Inspection Result</h2>
              <div className="result-row">
                <span>Status</span>
                <strong>{result.status}</strong>
              </div>
              <div className="result-row">
                <span>Defect Type</span>
                <strong>{result.defect_type}</strong>
              </div>
              <div className="result-row">
                <span>Severity</span>
                <strong>{result.severity}</strong>
              </div>
              <div className="result-row">
                <span>Confidence</span>
                <strong>{result.confidence * 100}%</strong>
              </div>
              <p>{result.notes}</p>
            </section>
          )}

          <section className="card table-card">
            <h2>Inspection History</h2>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>File</th>
                  <th>Status</th>
                  <th>Defect</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {inspections.map((item) => (
                  <tr key={item.id}>
                    <td>{item.id}</td>
                    <td>{item.file_name}</td>
                    <td>{item.status}</td>
                    <td>{item.defect_type || '—'}</td>
                    <td>{item.confidence ? `${(item.confidence * 100).toFixed(0)}%` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}

      {message && <div className="message">{message}</div>}
    </div>
  )
}

export default App
