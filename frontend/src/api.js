const TOKEN_KEY = 'vi_token'

export const getToken = () => {
  try { return localStorage.getItem(TOKEN_KEY) } catch { return null }
}
export const setToken = (t) => {
  try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY) } catch { /* storage blocked */ }
}

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status }
}

async function request(path, { method = 'GET', body, form, raw } = {}) {
  const headers = {}
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  let payload
  if (form) payload = form
  else if (body !== undefined) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body) }

  const res = await fetch(`/api${path}`, { method, headers, body: payload })
  if (res.status === 401 && path !== '/auth/login') {
    setToken(null)
    window.dispatchEvent(new Event('vi:logout'))
  }
  if (!res.ok) {
    let msg = res.statusText
    try {
      const data = await res.json()
      msg = typeof data.detail === 'string' ? data.detail : data.detail?.map((d) => d.msg).join(', ') || msg
    } catch { /* non-JSON error */ }
    throw new ApiError(res.status, msg)
  }
  if (raw) return res
  return res.status === 204 ? null : res.json()
}

export const api = {
  login: (email, password) => {
    const form = new URLSearchParams({ username: email, password })
    return fetch('/api/auth/login', { method: 'POST', body: form }).then(async (r) => {
      const data = await r.json()
      if (!r.ok) throw new ApiError(r.status, data.detail || 'Login failed')
      return data
    })
  },
  me: () => request('/auth/me'),

  users: () => request('/users'),
  createUser: (u) => request('/users', { method: 'POST', body: u }),
  updateUser: (id, u) => request(`/users/${id}`, { method: 'PATCH', body: u }),

  stats: () => request('/dashboard/stats'),
  batches: (params = {}) => request(`/batches?${new URLSearchParams(params)}`),
  batch: (id) => request(`/batches/${id}`),
  deleteBatch: (id) => request(`/batches/${id}`, { method: 'DELETE' }),
  upload: (form) => request('/batches/upload', { method: 'POST', form }),

  dataset: () => request('/dataset'),
  importDataset: (b) => request('/dataset/import', { method: 'POST', body: b }),
  capture: (b) => request('/camera/capture', { method: 'POST', body: b }),

  // Image endpoints need the bearer token, so they are fetched as blobs.
  blob: (path) => request(path, { raw: true }).then((r) => r.blob()),
}
