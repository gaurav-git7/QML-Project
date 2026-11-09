import { useCallback, useEffect, useState } from 'react'
import { api } from './api.js'
import { Icon } from './components/Icons.jsx'
import NavRail, { NAV } from './components/NavRail.jsx'
import Cardiac from './pages/Cardiac.jsx'
import Methods from './pages/Methods.jsx'
import Performance from './pages/Performance.jsx'
import ReadingRoom from './pages/ReadingRoom.jsx'
import Worklist from './pages/Worklist.jsx'

const EMPTY = { models: [], roles: [] }
const DISCLAIMER = 'Research prototype for comparing quantum and classical machine-learning models. Not a medical device - not validated for clinical use or diagnosis.'

export default function App() {
  const [page, setPage] = useState('read')
  const [health, setHealth] = useState(null)
  const [data, setData] = useState({ xray: EMPTY, heart: EMPTY })
  const [worklist, setWorklist] = useState([])
  const [request, setRequest] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [theme, setTheme] = useState(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light')

  const load = useCallback(async () => {
    setLoadError('')
    try {
      setHealth(await api.health())
      const [xm, xp, hm, hp, wl] = await Promise.all([api.models('xray'), api.primary('xray'), api.models('heart'), api.primary('heart'), api.worklist()])
      setData({ xray: { models: xm, roles: xp.roles }, heart: { models: hm, roles: hp.roles } })
      setWorklist(wl)
    } catch (e) { setLoadError(e.message) }
  }, [])
  useEffect(() => { load() }, [load])

  // poll until the server has warmed up its models
  useEffect(() => {
    if (health?.ready) return undefined
    const t = setInterval(() => api.health().then(setHealth).catch(() => {}), 2000)
    return () => clearInterval(t)
  }, [health?.ready])

  // never let a dropped file navigate the browser away from the app
  useEffect(() => {
    const stop = (e) => e.preventDefault()
    window.addEventListener('dragover', stop); window.addEventListener('drop', stop)
    return () => { window.removeEventListener('dragover', stop); window.removeEventListener('drop', stop) }
  }, [])

  // page changes: reset scroll and update the tab title
  useEffect(() => {
    window.scrollTo(0, 0)
    document.title = `${NAV.find((n) => n.id === page)?.label} · QuantumML Diagnostic Workstation`
  }, [page])

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    document.documentElement.dataset.theme = next
    try { localStorage.setItem('qml-theme', next) } catch { /* private mode */ }
  }
  const openStudy = (id) => { setRequest({ id, ts: Date.now() }); setPage('read') }

  return (
    <div className="app">
      <a className="skip" href="#main">Skip to content</a>
      <NavRail page={page} setPage={setPage} ready={!!health?.ready} theme={theme} toggleTheme={toggleTheme} />
      <div className="content">
        <div className="disclaimer-bar" role="note"><Icon name="alert" size={18} /><span>{health?.disclaimer || DISCLAIMER}</span></div>
        {loadError && (
          <div className="msg err" role="alert" style={{ marginBottom: 16 }}>
            Could not reach the server ({loadError}). Start it with <code>python server/run.py</code>, then <button className="btn sm" onClick={load}>Try again</button>
          </div>
        )}
        <main id="main">
          <ReadingRoom active={page === 'read'} models={data.xray.models} worklist={worklist} request={request} />
          <Worklist active={page === 'worklist'} rows={worklist} onOpen={openStudy} />
          <Performance active={page === 'perf'} data={data} />
          <Cardiac active={page === 'heart'} models={data.heart.models} />
          <Methods active={page === 'methods'} />
        </main>
      </div>
    </div>
  )
}
