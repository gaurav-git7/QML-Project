async function j(r) {
  if (!r.ok) {
    let msg = r.statusText
    try { msg = (await r.json()).detail || msg } catch { /* keep the status text */ }
    throw new Error(msg)
  }
  return r.json()
}

const post = (url, body) =>
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(j)

export const api = {
  health: () => fetch('/api/health').then(j),
  models: (module) => fetch(`/api/models/${module}`).then(j),
  primary: (module) => fetch(`/api/primary/${module}`).then(j),
  worklist: () => fetch('/api/worklist').then(j),
  heartSchema: () => fetch('/api/heart/schema').then(j),
  circuit: (module) => fetch(`/api/circuit/${module}`).then(j),
  studyImageUrl: (id) => `/api/study/${id}/image`,
  analyse: ({ file, studyId, runAll }) => {
    const fd = new FormData()
    if (file) fd.append('file', file)
    if (studyId != null) fd.append('study_id', String(studyId))
    fd.append('run_all', runAll ? 'true' : 'false')
    return fetch('/api/analyse', { method: 'POST', body: fd }).then(j)
  },
  heartPredict: (values) => post('/api/heart/predict', values),
  logDecision: (body) => post('/api/decision', body),
}
