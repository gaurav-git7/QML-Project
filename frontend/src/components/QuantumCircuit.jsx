import { useMemo, useState } from 'react'

/* ---------------------------------------------------------------- the circuit as a list of columns
   Every column holds gates that act on different qubits at the same time. The ZZ terms of the feature map are diagonal, so
   they commute and are drawn as an even-pairs layer then an odd-pairs layer (verified equal to the model's simulator). */
export function buildColumns(n, fmReps, anReps) {
  const cols = []
  const add = (band, sub, gates) => cols.push({ band, sub, gates })
  const all = (t, extra = {}) => Array.from({ length: n }, (_, q) => ({ t, q, ...extra }))
  for (let r = 1; r <= fmReps; r++) {
    const band = `fm${r}`
    add(band, 'H', all('H'))
    add(band, 'P(2θ)', all('P'))
    for (const parity of [0, 1]) {
      const starts = []
      for (let i = parity; i < n - 1; i += 2) starts.push(i)
      add(band, 'CX', starts.map((i) => ({ t: 'CX', c: i, tg: i + 1 })))
      add(band, 'P(φ)', starts.map((i) => ({ t: 'P', q: i + 1 })))
      add(band, 'CX', starts.map((i) => ({ t: 'CX', c: i, tg: i + 1 })))
    }
  }
  for (let l = 0; l <= anReps; l++) {
    add(`rot${l}`, 'Rz', all('R', { name: 'Rz' }))
    add(`rot${l}`, 'Ry', all('R', { name: 'Ry' }))
    add(`rot${l}`, 'Rz', all('R', { name: 'Rz' }))
    if (l < anReps) {
      for (let q = 0; q < n - 1; q++) add(`ring${l}`, `CX ${q}→${q + 1}`, [{ t: 'CX', c: q, tg: q + 1 }])
      add(`ring${l}`, `CX ${n - 1}→0`, [{ t: 'CX', c: n - 1, tg: 0 }])
    }
  }
  add('meas', 'measure', [{ t: 'M', q: 0 }])
  return cols
}

export function countGates(cols) {
  const c = { H: 0, P: 0, CX: 0, Rz: 0, Ry: 0, M: 0 }
  cols.forEach((col) => col.gates.forEach((g) => { if (g.t === 'R') c[g.name]++; else c[g.t]++ }))
  return c
}

const BAND_LABEL = (b, anReps) => {
  if (b.startsWith('fm')) return `ZZ feature map · repetition ${b.slice(2)}`
  if (b.startsWith('rot')) return `Rotations · layer ${+b.slice(3) + 1}`
  if (b.startsWith('ring')) return `CNOT ring ${+b.slice(4) + 1}`
  return b === 'meas' ? 'Readout' : b
}
const bandClass = (b) => (b.startsWith('fm') ? 'fm' : b.startsWith('rot') ? 'rot' : b.startsWith('ring') ? 'ring' : 'meas')

/* ---------------------------------------------------------------- full gate-level circuit */
function FullCircuit({ info }) {
  const n = info.n_qubits
  const cols = useMemo(() => buildColumns(n, info.fm_reps, info.ansatz_reps), [n, info.fm_reps, info.ansatz_reps])
  const CW = 36, RH = 30, X0 = 176, Y0 = 118
  const cx = (c) => X0 + c * CW + CW / 2, qy = (q) => Y0 + q * RH
  const W = X0 + cols.length * CW + 30, H = Y0 + (n - 1) * RH + 40

  const runs = []                                      // contiguous groups of columns sharing a band, for the header labels
  cols.forEach((col, i) => { const last = runs[runs.length - 1]; if (last && last.band === col.band) last.end = i; else runs.push({ band: col.band, start: i, end: i }) })

  return (
    <div className="diagram-scroll tall">
      <svg className="circ" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Complete ${n}-qubit circuit, gate by gate`}>
        {runs.map((r, i) => (
          <g key={i}>
            <rect className={'cband ' + bandClass(r.band)} x={X0 + r.start * CW + 1} y={44} width={(r.end - r.start + 1) * CW - 2} height={H - 54} rx="8" />
            <text className="ctitle" x={X0 + r.start * CW + 8} y={36}>{(r.end - r.start + 1) * CW > 110 ? BAND_LABEL(r.band) : ''}</text>
          </g>
        ))}
        {Array.from({ length: n }, (_, q) => (
          <g key={q}>
            <text className="cq" x={10} y={qy(q) + 4}>q{q}</text>
            <text className="cin" x={X0 - 10} y={qy(q) + 4} textAnchor="end">{info.inputs[q]}</text>
            <line className="cwire" x1={X0 - 4} x2={q === 0 ? W - 14 : X0 + (cols.length - 1) * CW} y1={qy(q)} y2={qy(q)} />
          </g>
        ))}
        {cols.map((col, i) => (
          <g key={i}>
            <text className="csub" transform={`translate(${cx(i) + 4} ${Y0 - 14}) rotate(-90)`}>{col.sub}</text>
            {col.gates.map((g, k) => {
              if (g.t === 'CX') {
                const y1 = qy(g.c), y2 = qy(g.tg)
                return (
                  <g key={k}>
                    <line className="cline" x1={cx(i)} x2={cx(i)} y1={y1} y2={y2} />
                    <circle className="cdot" cx={cx(i)} cy={y1} r="4.5" />
                    <circle className="ctgt" cx={cx(i)} cy={y2} r="9" />
                    <line className="cline" x1={cx(i) - 9} x2={cx(i) + 9} y1={y2} y2={y2} /><line className="cline" x1={cx(i)} x2={cx(i)} y1={y2 - 9} y2={y2 + 9} />
                  </g>
                )
              }
              if (g.t === 'M') {
                return (
                  <g key={k}>
                    <rect className="cgate meas" x={cx(i) - 12} y={qy(g.q) - 12} width="24" height="24" rx="4" />
                    <path className="cmeter" d={`M${cx(i) - 8} ${qy(g.q) + 5} A8 8 0 0 1 ${cx(i) + 8} ${qy(g.q) + 5}`} /><line className="cmeter" x1={cx(i)} y1={qy(g.q) + 5} x2={cx(i) + 5} y2={qy(g.q) - 6} />
                  </g>
                )
              }
              const cls = g.t === 'R' ? 'rot' : 'fm'
              const label = g.t === 'R' ? g.name : g.t
              return (
                <g key={k}>
                  <rect className={'cgate ' + cls} x={cx(i) - 12} y={qy(g.q) - 12} width="24" height="24" rx="5" />
                  <text className="cgl" x={cx(i)} y={qy(g.q) + 4.5} textAnchor="middle">{label}</text>
                </g>
              )
            })}
          </g>
        ))}
      </svg>
    </div>
  )
}

/* ---------------------------------------------------------------- compact block view */
function BlockCircuit({ info }) {
  const n = info.n_qubits
  const X0 = 190, Y0 = 84, RH = 24, GAP = 8
  const qy = (q) => Y0 + q * RH
  const bottom = qy(n - 1)
  const fm = [{ w: 40, t: 'H on every qubit' }, { w: 74, t: 'P(2θᵢ)  on qubit i' }, { w: 116, t: 'ZZ chain  CX · P · CX' }]
  const blocks = []
  for (let r = 1; r <= info.fm_reps; r++) fm.forEach((b) => blocks.push({ ...b, kind: 'fm', grp: `ZZ feature map · repetition ${r}`, g: `fm${r}` }))
  for (let l = 0; l <= info.ansatz_reps; l++) {
    blocks.push({ w: 84, t: `Rz · Ry · Rz  (layer ${l + 1})`, kind: 'rot', grp: 'Trainable ansatz', g: 'an' })
    if (l < info.ansatz_reps) blocks.push({ w: 84, t: `CNOT ring ${l + 1}`, kind: 'ring', grp: 'Trainable ansatz', g: 'an' })
  }
  blocks.push({ w: 44, t: 'measure q0', kind: 'meas', grp: 'Readout', g: 'meas' })
  let x = X0
  const placed = blocks.map((b) => { const p = { ...b, x }; x += b.w + GAP; return p })
  const end = x - GAP
  const groups = []
  placed.forEach((b) => { const last = groups[groups.length - 1]; if (last && last.g === b.g) last.x2 = b.x + b.w; else groups.push({ g: b.g, grp: b.grp, x1: b.x, x2: b.x + b.w }) })
  const fmEnd = placed.filter((b) => b.kind === 'fm').slice(-1)[0]
  const W = end + 30, H = bottom + 80

  return (
    <div className="diagram-scroll">
      <svg className="circ" viewBox={`0 0 ${W} ${H}`} style={{ minWidth: 900 }} role="img" aria-label={`${n}-qubit circuit, block view`}>
        {groups.map((g, i) => (
          <g key={i}>
            <line className="cgrp" x1={g.x1} x2={g.x2} y1={50} y2={50} />
            <text className="ctitle" x={(g.x1 + g.x2) / 2} y={40} textAnchor="middle">{g.grp}</text>
          </g>
        ))}
        {Array.from({ length: n }, (_, q) => (
          <g key={q}>
            <text className="cq" x={8} y={qy(q) + 4}>q{q}</text>
            <text className="cin" x={X0 - 12} y={qy(q) + 4} textAnchor="end">{info.inputs[q]}</text>
            <line className="cwire" x1={X0 - 6} x2={q === 0 ? end + 14 : placed.filter((b) => b.kind !== 'meas').slice(-1)[0].x + 84} y1={qy(q)} y2={qy(q)} />
          </g>
        ))}
        {placed.map((b, i) => (
          <g key={i}>
            <rect x={b.x} y={Y0 - 14} width={b.w} height={bottom - Y0 + 28} rx="8" fill="var(--surface)" />
            <rect className={'cblk ' + b.kind} x={b.x} y={Y0 - 14} width={b.w} height={bottom - Y0 + 28} rx="8" />
            <text className="cblt" transform={`translate(${b.x + b.w / 2 + 4.5} ${(Y0 + bottom) / 2}) rotate(-90)`} textAnchor="middle">{b.t}</text>
          </g>
        ))}
        {fmEnd && (
          <g>
            <line className="cdash" x1={fmEnd.x + fmEnd.w + GAP / 2} x2={fmEnd.x + fmEnd.w + GAP / 2} y1={Y0 - 22} y2={bottom + 40} />
            <text className="ctitle" x={fmEnd.x + fmEnd.w + GAP / 2 - 6} y={bottom + 58} textAnchor="end">QSVM compares two images / patients here (fidelity kernel)</text>
            <text className="ctitle" x={fmEnd.x + fmEnd.w + GAP / 2 + 6} y={bottom + 58}>VQC continues through the trainable ansatz →</text>
          </g>
        )}
      </svg>
    </div>
  )
}

/* ---------------------------------------------------------------- panel with pipeline, equations, view switch and facts */
const PIPE = {
  xray: ['Chest X-ray image', 'grayscale · 128×128 · contrast stretch', 'frozen DenseNet-121 → 1,024 features', 'PCA → 16 numbers', 'scale to [0, 1]'],
  heart: ['16 survey answers', 'encode each answer as a number', 'scale to [0, 1] (1st-99th percentile)'],
}

export default function QuantumCircuit({ module, info }) {
  const [view, setView] = useState('block')
  const counts = useMemo(() => (info ? countGates(buildColumns(info.n_qubits, info.fm_reps, info.ansatz_reps)) : null), [info])
  if (!info) return <div className="empty"><span className="spinner" /> Loading the circuit of the trained model…</div>
  const tc = info.counts.total, fc = info.counts.feature_map, ac = info.counts.ansatz
  const drawnOk = counts && counts.H === (fc.h || 0) && counts.P === (fc.p || 0) && counts.CX === (fc.cx || 0) + (ac.cx || 0) && counts.Rz === (ac.rz || 0) && counts.Ry === (ac.ry || 0)
  const pct = (v) => (v == null ? '-' : v.toFixed(3))

  return (
    <div>
      <div className="flow" aria-label="How the 16 inputs are produced">
        {PIPE[module].map((s, i) => <span key={i} style={{ display: 'contents' }}>{i > 0 && <i>→</i>}<span>{s}</span></span>)}
        <i>→</i><span><b>16-qubit circuit</b></span><i>→</i><span>P(qubit 0 = 1)</span><i>→</i><span>scale &amp; bias → probability</span>
      </div>
      <p className="eq">
        Encoding: <b>θᵢ = s · (π/2) · xᵢ</b> with <b>s = {info.fm_scale}</b> (chosen on the validation set) ·
        feature map: <b>H</b>, then <b>P(2θᵢ)</b> on every qubit, then for each neighbouring pair <b>CX · P(2(π−θᵢ)(π−θᵢ₊₁)) · CX</b>, repeated {info.fm_reps}× ·
        ansatz: <b>Rz · Ry · Rz</b> on every qubit and a <b>CNOT ring</b>, {info.ansatz_reps}× , then a final rotation layer.
      </p>

      <div className="toolbar" style={{ marginTop: 14 }}>
        <div className="seg" role="group" aria-label="Circuit view">
          <button aria-pressed={view === 'block'} onClick={() => setView('block')}>Block view</button>
          <button aria-pressed={view === 'full'} onClick={() => setView('full')}>Full gate-level circuit</button>
        </div>
        <span className={'tag ' + (drawnOk ? 'yes' : 'no')}>{drawnOk ? '✓ drawing matches the counted Qiskit circuit' : '⚠ drawing differs from the Qiskit gate count'}</span>
      </div>
      {view === 'block' ? <BlockCircuit info={info} /> : <FullCircuit info={info} />}
      {view === 'full' && <p className="note">Scroll sideways to follow the circuit. Each column is one time step; the ZZ terms are drawn in two parallel layers because they commute. The ring&apos;s CNOTs form a staircase: each one needs the previous, so they cannot run in parallel.</p>}

      <div className="legend" style={{ marginTop: 12 }}>
        <span><i style={{ background: 'var(--quantum)' }} />feature-map gates (H, P, ZZ terms)</span>
        <span><i style={{ background: 'var(--blue)' }} />trainable rotations (Rz, Ry)</span>
        <span><i style={{ background: 'var(--text-2)' }} />CNOT (dot = control, ⊕ = target)</span>
        <span><i style={{ background: 'var(--text)' }} />measurement of qubit 0</span>
      </div>

      <div className="tablewrap" style={{ marginTop: 16 }}>
        <table className="data">
          <tbody>
            <tr><td><b>Qubits</b></td><td>{info.n_qubits} - one per input number ({module === 'xray' ? 'PCA components PC1-PC16' : 'the 16 survey features'})</td></tr>
            <tr><td><b>Feature map</b></td><td>ZZ, {info.fm_reps} repetitions, linear entanglement · {fc.h} H · {fc.p} P · {fc.cx} CNOT · depth {info.depth.feature_map}</td></tr>
            <tr><td><b>Trainable ansatz</b></td><td>3 rotation layers (Rz·Ry·Rz) and 2 CNOT rings · {ac.rz} Rz · {ac.ry} Ry · {ac.cx} CNOT · depth {info.depth.ansatz} · <b>{info.trainable_angles} trainable angles</b> + {info.head_params} classical head parameters (scale w, bias b)</td></tr>
            <tr><td><b>Whole circuit</b></td><td>{tc.cx} CNOT · {tc.h} H · {tc.p} P · {tc.rz} Rz · {tc.ry} Ry · 1 measurement · total depth {info.depth.total}</td></tr>
            <tr><td><b>Readout</b></td><td>p₁ = P(qubit 0 = 1), then probability = σ( w · (2p₁ − 1) + b ). The other 15 qubits are not measured.</td></tr>
            <tr><td><b>QSVM</b></td><td>uses only the feature-map part: kernel = fidelity |⟨ψ(x)|ψ(y)⟩|² of two encoded states (rescaled with γ = {info.qsvm.gamma}), SVM C = {info.qsvm.C}, {info.qsvm.support_vectors} support vectors, trained on {info.qsvm.n_train?.toLocaleString()} cases · test balanced accuracy <b>{pct(info.qsvm.test_balanced_accuracy)}</b></td></tr>
            <tr><td><b>VQC</b></td><td>the whole circuit; Adam (learning rate {info.vqc.lr}, batch {info.vqc.batch}), {info.vqc.epochs_run} epochs on {info.vqc.n_train?.toLocaleString()} cases · test balanced accuracy <b>{pct(info.vqc.test_balanced_accuracy)}</b></td></tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
