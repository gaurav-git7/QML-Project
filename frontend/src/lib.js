// small shared helpers
export const pct = (v, d = 1) => (v == null ? '-' : `${(v * 100).toFixed(d)}%`)
export const f3 = (v) => (v == null || Number.isNaN(v) ? '-' : Number(v).toFixed(3))
export const caseId = (id) => `#${String(id - 5231).padStart(4, '0')}`   // anonymised case number (test studies start at dataset index 5232)

const METHOD_SHORT = { 'Quantum kernel SVM': 'QSVM', 'Variational quantum classifier': 'VQC', 'Logistic regression': 'Logistic regression', 'SVM (RBF)': 'SVM (RBF)', 'Neural network (MLP)': 'Neural network', 'Gradient boosting': 'Gradient boosting' }
const REP_SHORT = { '16 CNN-PCA features': '16 PCA features', '1,024 CNN features': '1,024 CNN features', '16 clinical features': '16 features' }

/** short human name of a model, e.g. "QSVM · 16 qubits" or "SVM (RBF) · 16 PCA features" */
export function shortName(r) {
  const m = METHOD_SHORT[r.method] || r.method
  return r.family === 'quantum' ? `${m} · ${r.n_qubits} qubits` : `${m} · ${REP_SHORT[r.rep_label] || r.rep_label}`
}

/** unique name including how it was trained, for tables, charts and menus */
export const modelName = (r) => `${shortName(r)} (${r.regime === 'sub' ? 'subset' : 'full data'})`
