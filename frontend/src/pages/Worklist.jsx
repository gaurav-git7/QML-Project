import { useMemo, useState } from 'react'
import { Empty, PageHeader, StatCards } from '../components/common.jsx'
import DataTable from '../components/DataTable.jsx'
import { Icon } from '../components/Icons.jsx'
import { caseId, pct } from '../lib.js'

const YesNo = ({ v }) => <span className={'tag ' + (v ? 'yes' : 'no')}>{v ? 'Yes' : 'No'}</span>
const Ref = ({ v }) => <b style={{ color: v === 'PNEUMONIA' ? 'var(--pos)' : 'var(--neg)' }}>{v === 'PNEUMONIA' ? 'Pneumonia' : 'Normal'}</b>
const read = (v) => (v === 'PNEUMONIA' ? 'Pneumonia' : 'Normal')

function downloadCsv(rows) {
  const head = ['case', 'file', 'reference', 'quantum_p_pneumonia', 'classical_p_pneumonia', 'quantum_read', 'classical_read', 'models_agree', 'quantum_correct', 'classical_correct']
  const lines = [head.join(',')].concat(rows.map((r) => [caseId(r.id), r.study, r.reference, r.qProb, r.cProb, r.qRead, r.cRead, r.agree, r.qCorrect, r.cCorrect].join(',')))
  const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }))
  const a = document.createElement('a'); a.href = url; a.download = 'worklist.csv'; a.click(); URL.revokeObjectURL(url)
}

export default function Worklist({ active, rows, onOpen }) {
  const [label, setLabel] = useState('All')
  const [onlyDisagree, setOnlyDisagree] = useState(false)
  const [query, setQuery] = useState('')

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    return rows.filter((r) => (label === 'All' || r.reference === label) && (!onlyDisagree || !r.agree)
      && (!q || caseId(r.id).toLowerCase().includes(q) || r.study.toLowerCase().includes(q)))
  }, [rows, label, onlyDisagree, query])
  const share = (k) => (shown.length ? shown.filter((r) => r[k]).length / shown.length : 0)

  return (
    <div className={active ? '' : 'hidden'}>
      <PageHeader active={active} title="Worklist & Validation" subtitle="Every held-out test study with both models' outputs. Open a case to read it in the Reading Room.">
        <button className="btn" onClick={() => downloadCsv(shown)} disabled={!shown.length}><Icon name="download" size={16} />Export CSV</button>
      </PageHeader>
      {!rows.length ? <div className="card"><Empty>No trained X-ray models yet - run <code>scripts/train_xray.py</code>.</Empty></div> : (
        <>
          <StatCards items={[
            { label: 'Cases shown', value: shown.length.toLocaleString(), sub: `of ${rows.length.toLocaleString()} held-out test cases` },
            { label: 'Quantum model correct', value: pct(share('qCorrect')), sub: 'primary quantum model', pct: share('qCorrect'), tone: 'q' },
            { label: 'Classical model correct', value: pct(share('cCorrect')), sub: 'same-input classical model', pct: share('cCorrect'), tone: 'c' },
            { label: 'Models agree', value: pct(share('agree')), sub: 'quantum and classical give the same read', pct: share('agree') },
          ]} />
          <div className="card">
            <div className="toolbar">
              <div className="seg" role="group" aria-label="Reference label">
                {[['All', 'All'], ['NORMAL', 'Normal'], ['PNEUMONIA', 'Pneumonia']].map(([k, l]) => <button key={k} aria-pressed={label === k} onClick={() => setLabel(k)}>{l}</button>)}
              </div>
              <label className="switch"><input type="checkbox" checked={onlyDisagree} onChange={(e) => setOnlyDisagree(e.target.checked)} />Only cases where the models disagree</label>
              <label className="grow" style={{ position: 'relative' }}>
                <span className="sr-only">Search cases</span>
                <input className="input" style={{ paddingLeft: 36 }} type="search" placeholder="Search case number or file name…" value={query} onChange={(e) => setQuery(e.target.value)} />
                <span style={{ position: 'absolute', left: 11, top: 10, color: 'var(--muted)' }}><Icon name="search" size={16} /></span>
              </label>
            </div>
            <DataTable rows={shown} pageSize={25} pageSizes={[10, 25, 50, 100]} onRowClick={(r) => onOpen(r.id)} rowLabel={(r) => `Open case ${caseId(r.id)} in the Reading Room`}
              emptyText="No cases match these filters." columns={[
                { key: 'id', label: 'Case', render: (r) => <b>{caseId(r.id)}</b> },
                { key: 'study', label: 'File' },
                { key: 'reference', label: 'Reference', render: (r) => <Ref v={r.reference} /> },
                { key: 'qProb', label: 'Quantum P(pneumonia)', render: (r) => `${(r.qProb * 100).toFixed(1)}%` },
                { key: 'cProb', label: 'Classical P(pneumonia)', render: (r) => `${(r.cProb * 100).toFixed(1)}%` },
                { key: 'qRead', label: 'Quantum read', render: (r) => read(r.qRead) },
                { key: 'cRead', label: 'Classical read', render: (r) => read(r.cRead) },
                { key: 'agree', label: 'Agree', render: (r) => <YesNo v={r.agree} />, sortValue: (r) => (r.agree ? 1 : 0) },
                { key: 'qCorrect', label: 'Quantum correct', render: (r) => <YesNo v={r.qCorrect} />, sortValue: (r) => (r.qCorrect ? 1 : 0) },
                { key: 'cCorrect', label: 'Classical correct', render: (r) => <YesNo v={r.cCorrect} />, sortValue: (r) => (r.cCorrect ? 1 : 0) },
              ]} />
          </div>
        </>
      )}
    </div>
  )
}
