import { useEffect, useMemo, useState } from 'react'
import { Confusion, HBars, RocChart } from '../components/Charts.jsx'
import { Empty, PageHeader, StatCards } from '../components/common.jsx'
import DataTable from '../components/DataTable.jsx'
import { f3, modelName, pct, shortName } from '../lib.js'

const cmp = (a, b) => a.val.balanced_accuracy - b.val.balanced_accuracy || a.val.auc - b.val.auc
const best = (list) => (list.length ? list.reduce((a, b) => (cmp(b, a) > 0 ? b : a)) : null)
const bal = (r) => r.test.balanced_accuracy

export default function Performance({ active, data }) {
  const [mod, setMod] = useState('xray')
  const models = data[mod].models
  const roles = data[mod].roles
  const byId = (id) => models.find((m) => m.id === id)
  const primaryIds = useMemo(() => [...new Set(roles.map((r) => r.id))], [roles])
  const [rocSel, setRocSel] = useState(primaryIds)
  const [cmId, setCmId] = useState(primaryIds[0])
  useEffect(() => { setRocSel(primaryIds); setCmId(primaryIds[0]) }, [primaryIds])

  const q = byId(roles.find((r) => r.key === 'quantum')?.id)
  const c = byId((roles.find((r) => r.key === 'classical_same') || roles.find((r) => r.key === 'classical_best'))?.id)
  const nQ = models.filter((m) => m.family === 'quantum').length
  const pos = mod === 'xray' ? 'pneumonia' : 'heart disease'
  const neg = mod === 'xray' ? 'normal' : 'lower risk'
  const ordered = useMemo(() => [...models].sort((a, b) => bal(b) - bal(a)), [models])

  const sideBySide = useMemo(() => {
    const reps = [...new Set(models.filter((m) => m.family === 'quantum').map((m) => m.representation))]
    const rows = reps.map((rep) => {
      const qq = best(models.filter((m) => m.family === 'quantum' && m.representation === rep))
      const cc = best(models.filter((m) => m.family === 'classical' && m.representation === rep && m.regime === 'sub'))
      return { id: rep, input: qq.rep_label, qModel: shortName(qq), qBal: bal(qq), qAuc: qq.test.auc, cModel: cc ? shortName(cc) : '-', cBal: cc ? bal(cc) : null, cAuc: cc?.test.auc ?? null, gap: cc ? bal(qq) - bal(cc) : null }
    })
    const ceil = best(models.filter((m) => m.family === 'classical' && m.representation === 'DENSE'))
    if (ceil) rows.push({ id: 'ceiling', input: 'All 1,024 CNN features (upper reference)', qModel: '-', qBal: null, qAuc: null, cModel: shortName(ceil), cBal: bal(ceil), cAuc: ceil.test.auc, gap: null })
    return rows
  }, [models])

  const bars = useMemo(() => ordered.map((m) => ({ name: modelName(m), value: bal(m), family: m.family, hint: `Trained on ${m.regime === 'sub' ? 'the balanced subset' : 'the full training set'} (n=${m.n_train.toLocaleString()})` })), [ordered])
  const curves = ordered.filter((m) => rocSel.includes(m.id)).map((m) => ({ name: modelName(m), family: m.family, fpr: m.roc.fpr, tpr: m.roc.tpr, auc: m.test.auc }))
  const cmModel = byId(cmId)
  const toggleRoc = (id) => setRocSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]))
  const gap = q && c ? (bal(q) - bal(c)) * 100 : null

  return (
    <div className={active ? '' : 'hidden'}>
      <PageHeader active={active} title="Model Performance" subtitle="Quantum and classical models compared on the same inputs, scored once on a held-out test set.">
        <div className="seg" role="group" aria-label="Dataset">
          {[['xray', 'Chest X-ray'], ['heart', 'Heart disease']].map(([k, l]) => <button key={k} aria-pressed={mod === k} onClick={() => setMod(k)}>{l}</button>)}
        </div>
      </PageHeader>

      {!models.length ? <div className="card"><Empty>No {mod} results yet - run the training script first.</Empty></div> : (
        <div className="stack">
          <StatCards items={[
            q ? { label: 'Best quantum model (chosen on validation)', value: pct(bal(q)), sub: shortName(q), pct: bal(q), tone: 'q' } : { label: 'Best quantum model', value: '-', sub: 'not trained yet' },
            c ? { label: 'Classical model on the same input', value: pct(bal(c)), sub: shortName(c), pct: bal(c), tone: 'c' } : { label: 'Classical, same input', value: '-', sub: 'not trained yet' },
            gap != null ? { label: 'Quantum minus classical', value: `${gap >= 0 ? '+' : ''}${gap.toFixed(1)} pts`, sub: 'test balanced accuracy' } : { label: 'Quantum minus classical', value: '-', sub: 'needs both models' },
            { label: 'Models evaluated', value: String(models.length), sub: `${nQ} quantum · ${models.length - nQ} classical` },
          ]} />

          <details className="card glossary">
            <summary style={{ cursor: 'pointer', fontWeight: 650 }}>What do these metrics mean?</summary>
            <dl style={{ marginTop: 14 }}>
              <dt>Balanced accuracy</dt><dd>The average of sensitivity and specificity. It is fair when one class is much more common, which is the case here. 0.5 means random guessing.</dd>
              <dt>Sensitivity</dt><dd>Of the patients who truly have {pos}, the share the model catches. Low sensitivity means missed cases.</dd>
              <dt>Specificity</dt><dd>Of the patients who are {neg}, the share the model correctly clears. Low specificity means false alarms.</dd>
              <dt>AUC</dt><dd>How well the model ranks sick patients above healthy ones across all thresholds: 1.0 is perfect, 0.5 is chance.</dd>
              <dt>F1</dt><dd>A single score that balances precision and sensitivity.</dd>
              <dt>Validation vs test</dt><dd>Settings were chosen on the validation set. The test set was used once, so test numbers are the honest ones.</dd>
            </dl>
          </details>

          <div className="card">
            <div className="card-h"><h2>Quantum vs classical on the same input</h2></div>
            <DataTable rows={sideBySide} columns={[
              { key: 'input', label: 'Same input' }, { key: 'qModel', label: 'Quantum model' },
              { key: 'qBal', label: 'Quantum bal. acc.', render: (r) => f3(r.qBal) }, { key: 'qAuc', label: 'Quantum AUC', render: (r) => f3(r.qAuc) },
              { key: 'cModel', label: 'Classical model' },
              { key: 'cBal', label: 'Classical bal. acc.', render: (r) => f3(r.cBal) }, { key: 'cAuc', label: 'Classical AUC', render: (r) => f3(r.cAuc) },
              { key: 'gap', label: 'Quantum − classical', render: (r) => (r.gap == null ? '-' : <b style={{ color: r.gap >= 0 ? 'var(--neg)' : 'var(--pos)' }}>{r.gap >= 0 ? '+' : ''}{r.gap.toFixed(3)}</b>) },
            ]} />
          </div>

          <div className="card">
            <div className="card-h"><h2>All models - balanced accuracy on the test set</h2></div>
            <HBars items={bars} />
          </div>

          <div className="grid2">
            <div className="card">
              <div className="card-h"><h2>ROC curves</h2></div>
              <RocChart curves={curves} />
              <details className="fold"><summary>Choose models ({rocSel.length} selected)</summary>
                <div style={{ maxHeight: 220, overflowY: 'auto' }}>
                  {ordered.map((m) => <label key={m.id} className="switch" style={{ margin: '8px 0' }}><input type="checkbox" checked={rocSel.includes(m.id)} onChange={() => toggleRoc(m.id)} />{modelName(m)}</label>)}
                </div>
              </details>
            </div>
            <div className="card">
              <div className="card-h"><h2>Confusion matrix</h2></div>
              <label className="field"><span className="sr-only">Model</span>
                <select className="select" value={cmId} onChange={(e) => setCmId(e.target.value)}>
                  {ordered.map((m) => <option key={m.id} value={m.id}>{modelName(m)}</option>)}
                </select>
              </label>
              {cmModel && <Confusion t={cmModel.test} posName={pos} negName={neg} />}
              {cmModel && <p className="note" style={{ marginTop: 14 }}>Balanced accuracy {f3(cmModel.test.balanced_accuracy)} · sensitivity {f3(cmModel.test.sensitivity)} · specificity {f3(cmModel.test.specificity)} · AUC {f3(cmModel.test.auc)} · F1 {f3(cmModel.test.f1)}</p>}
            </div>
          </div>

          <div className="card">
            <div className="card-h"><h2>Full results</h2></div>
            <DataTable rows={models} initialSort={{ key: 'test', dir: 'desc' }} columns={[
              { key: 'label', label: 'Model', render: (r) => shortName(r), sortValue: (r) => shortName(r) },
              { key: 'family', label: 'Type', render: (r) => <span className={'tag ' + (r.family === 'quantum' ? 'q' : 'c')}>{r.family}</span> },
              { key: 'trained', label: 'Trained on', render: (r) => `${r.regime === 'sub' ? 'balanced subset' : 'full set'} (n=${r.n_train.toLocaleString()})`, sortValue: (r) => r.n_train },
              { key: 'val', label: 'Validation bal. acc.', render: (r) => f3(r.val.balanced_accuracy), sortValue: (r) => r.val.balanced_accuracy },
              { key: 'test', label: 'Test bal. acc.', render: (r) => <b>{f3(r.test.balanced_accuracy)}</b>, sortValue: (r) => r.test.balanced_accuracy },
              { key: 'sens', label: 'Sensitivity', render: (r) => f3(r.test.sensitivity), sortValue: (r) => r.test.sensitivity },
              { key: 'spec', label: 'Specificity', render: (r) => f3(r.test.specificity), sortValue: (r) => r.test.specificity },
              { key: 'auc', label: 'AUC', render: (r) => f3(r.test.auc), sortValue: (r) => r.test.auc },
              { key: 'f1', label: 'F1', render: (r) => f3(r.test.f1), sortValue: (r) => r.test.f1 },
              { key: 'time', label: 'Training time', render: (r) => (r.train_time_s ? `${r.train_time_s.toFixed(0)} s` : '-'), sortValue: (r) => r.train_time_s },
            ]} />
          </div>
        </div>
      )}
    </div>
  )
}
