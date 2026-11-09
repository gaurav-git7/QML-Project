export const COLORS = { quantum: 'var(--quantum)', classical: 'var(--classical)' }

// quantum = teal family, classical = amber/brown family (the same colours as the bars and tags everywhere)
const Q_SHADES = ['#0f766e', '#14b8a6', '#0891b2']
const C_SHADES = ['#b45309', '#d97706', '#92400e', '#ea8a1f', '#a16207']
export const curveColor = (family, i) => (family === 'quantum' ? Q_SHADES : C_SHADES)[i % (family === 'quantum' ? Q_SHADES.length : C_SHADES.length)]

/** Horizontal bar list on a 0.5-1.0 axis (0.5 = chance). items: [{ name, value, family, hint }] */
export function HBars({ items, min = 0.5, max = 1 }) {
  const pos = (v) => `${Math.max(0, Math.min(1, (v - min) / (max - min))) * 100}%`
  const ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  return (
    <div>
      <div className="hbars">
        {items.map((it, i) => (
          <div className="hbar-row" key={i} title={it.hint}>
            <div className="nm">{it.name}</div>
            <div className="hbar"><div style={{ width: pos(it.value), background: it.family === 'quantum' ? COLORS.quantum : COLORS.classical }} /></div>
            <div className="v">{it.value.toFixed(3)}</div>
          </div>
        ))}
      </div>
      <div className="axis"><span /><div>{ticks.map((t) => <span key={t}>{t.toFixed(1)}</span>)}</div><span /></div>
      <div className="legend"><span><i style={{ background: COLORS.quantum }} />Quantum</span><span><i style={{ background: COLORS.classical }} />Classical</span><span>Axis starts at 0.5 = random guessing</span></div>
    </div>
  )
}

/** ROC chart: curves [{ name, family, fpr, tpr, auc }] */
export function RocChart({ curves }) {
  const W = 440, H = 420, P = 48
  const x = (v) => P + v * (W - P - 14), y = (v) => H - P - v * (H - P - 14)
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  const counts = { quantum: 0, classical: 0 }
  const styled = curves.map((c) => ({ ...c, color: curveColor(c.family, counts[c.family]++) }))
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxHeight: 440 }} role="img" aria-label="ROC curves of the selected models">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} stroke="var(--border)" />
            <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} stroke="var(--border)" />
            <text x={P - 8} y={y(t) + 4} fontSize="12" textAnchor="end" fill="var(--muted)">{t}</text>
            <text x={x(t)} y={H - P + 17} fontSize="12" textAnchor="middle" fill="var(--muted)">{t}</text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--border-strong)" strokeDasharray="4 4" />
        {styled.map((c) => (
          <polyline key={c.name} fill="none" strokeWidth="2.6" stroke={c.color} strokeDasharray={c.family === 'quantum' ? undefined : '7 4'}
            points={c.fpr.map((f, k) => `${x(f)},${y(c.tpr[k])}`).join(' ')} />
        ))}
        <text x={(x(0) + x(1)) / 2} y={H - 6} fontSize="12.5" textAnchor="middle" fill="var(--muted)">False-positive rate (1 − specificity)</text>
        <text transform={`translate(13 ${(y(0) + y(1)) / 2}) rotate(-90)`} fontSize="12.5" textAnchor="middle" fill="var(--muted)">Sensitivity (true-positive rate)</text>
      </svg>
      <div className="legend">
        {styled.map((c) => <span key={c.name}><i style={{ background: c.color }} />{c.name} · AUC {c.auc.toFixed(3)}{c.family === 'quantum' ? ' (solid)' : ' (dashed)'}</span>)}
      </div>
    </div>
  )
}

/** Confusion matrix from { tn, fp, fn, tp } */
export function Confusion({ t, posName, negName }) {
  return (
    <div className="cm" role="table" aria-label="Confusion matrix">
      <div /><div className="hd">Predicted {negName}</div><div className="hd">Predicted {posName}</div>
      <div className="hd">Actual {negName}</div>
      <div className="cell good">{t.tn}<small>true negatives</small></div>
      <div className="cell badc">{t.fp}<small>false alarms</small></div>
      <div className="hd">Actual {posName}</div>
      <div className="cell badc">{t.fn}<small>missed cases</small></div>
      <div className="cell good">{t.tp}<small>true positives</small></div>
    </div>
  )
}
