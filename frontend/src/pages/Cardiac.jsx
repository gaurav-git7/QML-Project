import { useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'
import { Findings, PageHeader } from '../components/common.jsx'
import DataTable from '../components/DataTable.jsx'
import { shortName } from '../lib.js'

const POS = 'Heart disease likely'
const NEG = 'Lower risk'
const SUBJECT = 'estimated probability of heart disease'
const GROUPS = [
  { title: 'About the patient', fields: ['Sex', 'AgeCategory'] },
  { title: 'Body and sleep', fields: ['BMI', 'SleepTime'] },
  { title: 'Lifestyle', fields: ['Smoking', 'AlcoholDrinking', 'PhysicalActivity'] },
  { title: 'Medical history', fields: ['GenHealth', 'Diabetic', 'Stroke', 'Asthma', 'KidneyDisease', 'SkinCancer', 'DiffWalking'] },
  { title: 'Last 30 days', fields: ['PhysicalHealth', 'MentalHealth'] },
]
const UNITS = { BMI: 'kg/m²', SleepTime: 'hours per night', PhysicalHealth: 'days, 0-30', MentalHealth: 'days, 0-30' }

function check(f, v) {
  if (f.kind !== 'num') return ''
  if (v === '' || v == null || Number.isNaN(Number(v))) return 'Enter a number.'
  const n = Number(v)
  if (n < f.min || n > f.max) return `Must be between ${f.min} and ${f.max}.`
  return ''
}

function Field({ f, value, onChange, error }) {
  const id = `f-${f.name}`
  return (
    <div className="field">
      <label className="lab" htmlFor={f.kind === 'num' || !['yn', 'sex'].includes(f.kind) ? id : undefined} id={`${id}-l`}>
        {f.label}{UNITS[f.name] && <span className="hint"> ({UNITS[f.name]})</span>}
      </label>
      {f.kind === 'num' && <input id={id} className="input" type="number" min={f.min} max={f.max} step="any" value={value} aria-invalid={!!error} aria-describedby={error ? `${id}-e` : undefined} onChange={(e) => onChange(e.target.value)} />}
      {(f.kind === 'yn' || f.kind === 'sex') && (
        <div className="seg" role="group" aria-labelledby={`${id}-l`}>{f.options.map((o) => <button type="button" key={o} aria-pressed={value === o} onClick={() => onChange(o)}>{o}</button>)}</div>
      )}
      {!['num', 'yn', 'sex'].includes(f.kind) && <select id={id} className="select" value={value} onChange={(e) => onChange(e.target.value)}>{f.options.map((o) => <option key={o}>{o}</option>)}</select>}
      {error && <div className="bad" id={`${id}-e`} role="alert">{error}</div>}
    </div>
  )
}

export default function Cardiac({ active, models }) {
  const [schema, setSchema] = useState([])
  const [values, setValues] = useState({})
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const defaults = (s) => Object.fromEntries(s.map((f) => [f.name, f.default]))
  useEffect(() => { api.heartSchema().then((s) => { setSchema(s); setValues(defaults(s)) }).catch((e) => setError(e.message)) }, [])

  const byName = useMemo(() => Object.fromEntries(schema.map((f) => [f.name, f])), [schema])
  const errors = useMemo(() => Object.fromEntries(schema.map((f) => [f.name, check(f, values[f.name])])), [schema, values])
  const valid = schema.length > 0 && Object.values(errors).every((e) => !e)
  const byId = (id) => models.find((m) => m.id === id)

  const submit = async (e) => {
    e.preventDefault()
    if (!valid) return
    setLoading(true); setError('')
    try { setResult(await api.heartPredict(values)) } catch (err) { setError(err.message) }
    setLoading(false)
  }
  const reset = () => { setValues(defaults(schema)); setResult(null); setError('') }
  const items = result ? result.roles.map((r) => ({ role: r.role, rec: byId(r.id), prob: r.prob })) : []
  const rows = result ? result.predictions.map((p) => ({ id: p.id, rec: byId(p.id), prob: p.prob })) : []

  return (
    <div className={active ? '' : 'hidden'}>
      <PageHeader active={active} title="Cardiac Risk" subtitle="Enter a patient's survey history. Quantum and classical models estimate the probability of heart disease side by side." />
      <div className="grid2" style={{ gridTemplateColumns: 'minmax(0, 1.25fr) minmax(0, 1fr)' }}>
        <form className="card" onSubmit={submit} noValidate>
          <div className="card-h"><h2>Patient history</h2><button type="button" className="btn sm" onClick={reset}>Reset to defaults</button></div>
          {GROUPS.map((g) => (
            <fieldset className="fieldset" key={g.title}>
              <legend>{g.title}</legend>
              <div className="formgrid">
                {g.fields.map((n) => byName[n] && <Field key={n} f={byName[n]} value={values[n] ?? ''} error={errors[n]} onChange={(v) => setValues((s) => ({ ...s, [n]: v }))} />)}
              </div>
            </fieldset>
          ))}
          <button className="btn primary full" type="submit" disabled={!valid || loading}>{loading ? <><span className="spinner" />Estimating…</> : 'Estimate risk'}</button>
          {!valid && schema.length > 0 && <p className="note" style={{ marginTop: 8 }}>Fix the highlighted fields to continue.</p>}
          {error && <div className="msg err" role="alert">{error}</div>}
        </form>
        <div className="card sticky-card" aria-label="AI findings">
          <div className="card-h"><h2>AI findings</h2></div>
          <Findings items={items} posName={POS} negName={NEG} subject={SUBJECT} idle="Fill in the form and press Estimate risk." />
          {result && (
            <details className="fold">
              <summary>All model outputs ({rows.length})</summary>
              <DataTable rows={rows} initialSort={{ key: 'prob', dir: 'desc' }} columns={[
                { key: 'm', label: 'Model', render: (r) => (r.rec ? `${shortName(r.rec)} (${r.rec.regime === 'sub' ? 'subset' : 'full'})` : r.id), sortValue: (r) => r.rec?.label },
                { key: 'f', label: 'Type', render: (r) => <span className={'tag ' + (r.rec?.family === 'quantum' ? 'q' : 'c')}>{r.rec?.family}</span> },
                { key: 'prob', label: 'P(heart disease)', render: (r) => `${(r.prob * 100).toFixed(1)}%`, sortValue: (r) => r.prob },
              ]} />
            </details>
          )}
          <p className="note" style={{ marginTop: 10 }}>This is a screening-style estimate from self-reported survey answers. It is not a diagnosis.</p>
        </div>
      </div>
    </div>
  )
}
