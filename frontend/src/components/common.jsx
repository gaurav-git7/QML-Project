import { useEffect, useRef } from 'react'
import { Icon } from './Icons.jsx'
import { shortName } from '../lib.js'

/** Page header: the h1 receives focus when the page changes (screen-reader friendly). */
export function PageHeader({ title, subtitle, active, children }) {
  const ref = useRef(null)
  useEffect(() => { if (active) ref.current?.focus({ preventScroll: true }) }, [active])
  return (
    <header className="pagehead">
      <div>
        <h1 tabIndex={-1} ref={ref}>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {children && <div className="actions">{children}</div>}
    </header>
  )
}

/** items: [{ label, value, sub, pct (0-1) | undefined, tone: 'q' | 'c' | undefined }] */
export function StatCards({ items }) {
  return (
    <div className="statrow">
      {items.map((it, i) => (
        <div key={i} className={'stat ' + (it.tone || '')}>
          <div className="lab">{it.label}</div>
          <div className="val">{it.value}</div>
          <div className="sub" title={it.sub}>{it.sub}</div>
          {it.pct != null && <div className="meter"><div style={{ width: `${Math.max(0, Math.min(1, it.pct)) * 100}%` }} /></div>}
        </div>
      ))}
    </div>
  )
}

/** One model's result: finding, probability and a gauge with the 50% decision threshold marked. */
export function ModelRow({ role, rec, prob, posName, negName, subject, reference }) {
  const positive = prob >= 0.5
  const kind = rec?.family === 'quantum' ? 'q' : 'c'
  let refNote = null
  if (reference) {
    const ok = (reference === 'PNEUMONIA') === positive
    refNote = <span className={ok ? 'ref-ok' : 'ref-bad'}>{ok ? '✓ matches the reference label' : '✗ differs from the reference label'}</span>
  }
  const t = rec?.test
  return (
    <div className={'mrow ' + kind}>
      <div className="mrow-head">
        <span className={'badge ' + kind}>{role}</span>
        <span className="mrow-name" title={rec?.label}>{rec && shortName(rec)}</span>
      </div>
      <div className="mrow-body">
        <div className={'finding ' + (positive ? 'pos' : 'neg')}><Icon name={positive ? 'alert' : 'checkCircle'} size={18} />{positive ? posName : negName}</div>
        <div className="prob"><b>{(prob * 100).toFixed(1)}%</b><small>{subject}</small></div>
      </div>
      <div className={'gauge ' + (positive ? 'pos' : '')} role="meter" aria-label={subject} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(prob * 100)}>
        <div style={{ width: `${prob * 100}%` }} /><span className="tick" title="50% decision threshold" />
      </div>
      <div className="gauge-scale" aria-hidden="true"><span>0%</span><span>50% threshold</span><span>100%</span></div>
      <div className="mrow-foot">
        {refNote && <div style={{ marginBottom: 4 }}>{refNote}</div>}
        {t && <>Held-out test: balanced accuracy {t.balanced_accuracy.toFixed(3)} · sensitivity {t.sensitivity.toFixed(3)} · specificity {t.specificity.toFixed(3)}</>}
      </div>
    </div>
  )
}

/** items: [{ role, rec, prob }] */
export function Findings({ items, posName, negName, subject, reference, idle }) {
  if (!items || !items.length) return <div className="banner idle"><Icon name="read" size={18} />{idle || 'No analysis yet.'}</div>
  const reads = new Set(items.map((i) => i.prob >= 0.5))
  const agree = reads.size === 1
  const positive = agree && [...reads][0]
  return (
    <div aria-live="polite">
      {agree
        ? <div className={'banner ' + (positive ? 'pos' : 'neg')}><Icon name={positive ? 'alert' : 'checkCircle'} size={20} />Models agree: {positive ? posName : negName}</div>
        : <div className="banner warn"><Icon name="alert" size={20} />Models disagree - clinician review needed</div>}
      {items.map((it, i) => <ModelRow key={i} {...it} posName={posName} negName={negName} subject={subject} reference={reference} />)}
      <p className="note">
        Each percentage is that model&apos;s estimated probability of the finding, from a model checked on a held-out test set (see Model Performance).
        These are research outputs: they can be confidently wrong, and they support - never replace - a clinician&apos;s read.
      </p>
    </div>
  )
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>
}
