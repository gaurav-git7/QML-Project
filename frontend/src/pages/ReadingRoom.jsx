import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { Findings, PageHeader } from '../components/common.jsx'
import DataTable from '../components/DataTable.jsx'
import { Icon } from '../components/Icons.jsx'
import Viewer from '../components/Viewer.jsx'
import { caseId, shortName } from '../lib.js'

const POS = 'Pneumonia suspected'
const NEG = 'Normal'
const SUBJECT = 'probability of pneumonia'
const DECISIONS = ['Agree with AI consensus', 'Normal', 'Pneumonia suspected', 'Indeterminate']
const isTyping = (t) => ['INPUT', 'TEXTAREA', 'SELECT'].includes(t?.tagName) && t.type !== 'range' && t.type !== 'checkbox'

export default function ReadingRoom({ active, models, worklist, request }) {
  const [source, setSource] = useState(null)       // { kind: 'file' | 'study', url, name, file?, studyId?, reference? }
  const [size, setSize] = useState(null)
  const [auto, setAuto] = useState(true)
  const [runAll, setRunAll] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [decision, setDecision] = useState('')
  const [notes, setNotes] = useState('')
  const [signed, setSigned] = useState(false)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef(null)
  const reqId = useRef(0)

  const reset = () => { setAnalysis(null); setError(''); setSigned(false); setDecision(''); setNotes(''); setSize(null) }

  const loadFile = useCallback((file) => {
    if (!file) return
    if (!file.type.startsWith('image/')) { setError('That file is not an image. Please choose a PNG or JPEG chest radiograph (DICOM files are not supported).'); return }
    setSource((prev) => { if (prev?.kind === 'file') URL.revokeObjectURL(prev.url); return { kind: 'file', file, url: URL.createObjectURL(file), name: file.name || 'pasted image' } })
    reset()
  }, [])

  const openStudy = useCallback((id) => {
    const row = worklist.find((r) => r.id === id)
    if (!row) return
    setSource({ kind: 'study', studyId: id, url: api.studyImageUrl(id), name: row.study, reference: row.reference })
    reset()
  }, [worklist])

  const randomStudy = useCallback(() => {
    if (!worklist.length) { setError('No held-out studies available - train the X-ray models first.'); return }
    openStudy(worklist[Math.floor(Math.random() * worklist.length)].id)
  }, [worklist, openStudy])

  const analyse = useCallback(async () => {
    if (!source) { setError('Load a chest radiograph first.'); return }
    const mine = ++reqId.current
    setLoading(true); setError(''); setSigned(false)
    try {
      const res = await api.analyse({ file: source.kind === 'file' ? source.file : null, studyId: source.kind === 'study' ? source.studyId : null, runAll })
      if (mine === reqId.current) setAnalysis(res)
    } catch (e) { if (mine === reqId.current) setError(e.message) }
    if (mine === reqId.current) setLoading(false)
  }, [source, runAll])

  useEffect(() => { if (request) openStudy(request.id) }, [request, openStudy])
  // analyse automatically as soon as a study is loaded
  useEffect(() => { if (source && auto) analyse() }, [source]) // eslint-disable-line react-hooks/exhaustive-deps

  // paste, drag-and-drop anywhere, keyboard shortcuts - only while this page is showing
  useEffect(() => {
    if (!active) return undefined
    let depth = 0
    const onPaste = (e) => {
      const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith('image/'))
      if (f) { e.preventDefault(); loadFile(f) }
    }
    const onEnter = (e) => { if (e.dataTransfer?.types?.includes('Files')) { depth++; setDragging(true) } }
    const onLeave = () => { depth = Math.max(0, depth - 1); if (!depth) setDragging(false) }
    const onDrop = (e) => { depth = 0; setDragging(false); if (e.dataTransfer?.files?.length) loadFile(e.dataTransfer.files[0]) }
    const onKey = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || isTyping(e.target)) return
      const k = e.key.toLowerCase()
      if (k === 'o') fileInput.current?.click()
      else if (k === 'n') randomStudy()
      else if (k === 'a') analyse()
      else if (k === 'i') window.dispatchEvent(new Event('viewer-invert'))
      else if (k === 'r') window.dispatchEvent(new Event('viewer-reset'))
      else if (k === '+' || k === '=') window.dispatchEvent(new Event('viewer-zoom-in'))
      else if (k === '-') window.dispatchEvent(new Event('viewer-zoom-out'))
      else return
      e.preventDefault()
    }
    window.addEventListener('paste', onPaste); window.addEventListener('dragenter', onEnter); window.addEventListener('dragleave', onLeave)
    window.addEventListener('drop', onDrop); window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('paste', onPaste); window.removeEventListener('dragenter', onEnter); window.removeEventListener('dragleave', onLeave)
      window.removeEventListener('drop', onDrop); window.removeEventListener('keydown', onKey); setDragging(false)
    }
  }, [active, loadFile, randomStudy, analyse])

  const sign = async () => {
    if (!analysis) { setError('Run the analysis before signing off.'); return }
    if (!decision) { setError("Choose the reader's impression first."); return }
    try {
      const q = analysis.roles[0]?.prob, c = analysis.roles.length > 1 ? analysis.roles[analysis.roles.length - 1].prob : null
      await api.logDecision({ reference: analysis.reference, quantum: q, classical: c, decision, notes })
      setSigned(true); setError('')
    } catch (e) { setError(e.message) }
  }

  const byId = (id) => models.find((m) => m.id === id)
  const items = analysis ? analysis.roles.map((r) => ({ role: r.role, rec: byId(r.id), prob: r.prob })) : []
  const agree = items.length > 0 && new Set(items.map((i) => i.prob >= 0.5)).size === 1
  const allRows = analysis ? analysis.predictions.map((p) => ({ id: p.id, rec: byId(p.id), prob: p.prob })) : []
  const revealed = !!analysis                                // the ground truth stays hidden until the models have read the study
  const idle = !source ? "Load a study to see the models' findings." : auto ? 'Analysing…' : 'Press Analyse to run the models.'

  return (
    <div className={active ? '' : 'hidden'}>
      <PageHeader active={active} title="Reading Room" subtitle="Load a chest radiograph and compare the quantum and classical models on the same study." />
      <div className="workspace">
        {/* ------------------------------------------------ viewer */}
        <section className="card viewer-card" aria-label="Image viewer">
          <Viewer src={source?.url} onSize={setSize}
            overlay={source ? (source.kind === 'study' ? `Held-out case ${caseId(source.studyId)}` : source.name) : null}
            leftControls={<>
              <button className="btn dark" onClick={() => fileInput.current?.click()} title="Open an image (O)"><Icon name="upload" size={16} />Open image…</button>
              <button className="btn dark" onClick={randomStudy} title="Load a random held-out test study (N)"><Icon name="shuffle" size={16} />Random test study</button>
            </>}
            empty={
              <button className="drop" onClick={() => fileInput.current?.click()}>
                <Icon name="upload" size={34} />
                <b>Drop a chest X-ray here</b>
                <span>or click to browse your files</span>
                <small>You can also paste an image (Ctrl+V) or load a random held-out test study from the toolbar.</small>
              </button>
            } />
          <input ref={fileInput} type="file" accept="image/*" hidden onChange={(e) => { loadFile(e.target.files?.[0]); e.target.value = '' }} />
        </section>

        {/* ------------------------------------------------ side panel */}
        <div className="side">
          <section className="card" aria-label="Study">
            <div className="card-h"><h2>Study</h2></div>
            <dl className="kv">
              <dt>Source</dt><dd>{source ? (source.kind === 'study' ? `Held-out test case ${caseId(source.studyId)}` : 'Your upload') : '-'}</dd>
              <dt>File</dt><dd>{!source ? '-' : source.kind === 'file' ? source.name : revealed ? source.name : <span className="blind"><Icon name="eyeOff" size={15} />Hidden until analysed</span>}</dd>
              <dt>Image size</dt><dd>{size ? `${size[0]} × ${size[1]} px` : '-'}</dd>
              {source?.kind === 'study' && <><dt>Reference label</dt><dd>{revealed ? <b>{source.reference}</b> : <span className="blind"><Icon name="eyeOff" size={15} />Hidden until analysed</span>}</dd></>}
            </dl>
            <div style={{ marginTop: 14, display: 'grid', gap: 10 }}>
              <label className="switch"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />Analyse automatically when a study loads</label>
              <label className="switch"><input type="checkbox" checked={runAll} onChange={(e) => setRunAll(e.target.checked)} />Also run every trained model (slower)</label>
              <button className="btn primary full" onClick={analyse} disabled={!source || loading}>
                {loading ? <><span className="spinner" />Analysing…</> : <><Icon name="perf" size={16} />{analysis ? 'Analyse again' : 'Analyse study'}</>}
              </button>
            </div>
            {error && <div className="msg err" role="alert">{error}</div>}
          </section>

          <section className="card" aria-label="AI findings">
            <div className="card-h"><h2>AI findings</h2></div>
            {loading && !analysis && <><div className="skeleton" /><div className="skeleton" /><div className="skeleton" /></>}
            {(!loading || analysis) && <Findings items={items} posName={POS} negName={NEG} subject={SUBJECT} reference={revealed ? source?.reference : undefined} idle={idle} />}
            {analysis && runAll && (
              <details className="fold" open>
                <summary>All model outputs ({allRows.length})</summary>
                <DataTable rows={allRows} initialSort={{ key: 'prob', dir: 'desc' }} columns={[
                  { key: 'm', label: 'Model', render: (r) => (r.rec ? `${shortName(r.rec)} (${r.rec.regime === 'sub' ? 'subset' : 'full'})` : r.id), sortValue: (r) => r.rec?.label },
                  { key: 'f', label: 'Type', render: (r) => <span className={'tag ' + (r.rec?.family === 'quantum' ? 'q' : 'c')}>{r.rec?.family}</span> },
                  { key: 'prob', label: 'P(pneumonia)', render: (r) => `${(r.prob * 100).toFixed(1)}%`, sortValue: (r) => r.prob },
                ]} />
              </details>
            )}
          </section>

          <section className="card" aria-label="Reader's decision">
            <div className="card-h"><h2>Reader&apos;s decision</h2></div>
            <div className="seg" role="group" aria-label="Impression">
              {DECISIONS.map((d) => {
                const blocked = d.startsWith('Agree') && analysis && !agree
                return <button key={d} aria-pressed={decision === d} disabled={blocked} title={blocked ? 'The models disagree, so there is no consensus to agree with' : undefined}
                  onClick={() => { setDecision(d); setSigned(false) }}>{d}</button>
              })}
            </div>
            <label className="field" style={{ marginTop: 14 }}>
              <span>Notes <span className="hint">(optional)</span></span>
              <textarea className="textarea" value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Findings, follow-up, remarks…" />
            </label>
            {!signed
              ? <button className="btn full" onClick={sign} disabled={!analysis}><Icon name="check" size={16} />Sign off &amp; log</button>
              : <>
                <div className="msg ok" role="status"><b>Signed off</b> - saved to the reader log.</div>
                <button className="btn primary full" style={{ marginTop: 10 }} onClick={randomStudy}>Next case <Icon name="next" size={16} /></button>
              </>}
            <div className="kbd" aria-label="Keyboard shortcuts">
              <span><kbd>O</kbd> open</span><span><kbd>N</kbd> next case</span><span><kbd>A</kbd> analyse</span><span><kbd>I</kbd> invert</span><span><kbd>R</kbd> reset view</span><span><kbd>+</kbd><kbd>-</kbd> zoom</span>
            </div>
          </section>
        </div>
      </div>
      {dragging && <div className="dragging-overlay" aria-hidden="true">Drop the image to open it</div>}
    </div>
  )
}
