/** System architecture: offline training lane, artifacts on disk, online serving lane. Pure SVG, theme-aware. */
function Box({ x, y, w, h, kind = 'plain', title, lines = [] }) {
  return (
    <g>
      <rect className={'abox ' + kind} x={x} y={y} width={w} height={h} rx="12" />
      <text className="at1" x={x + 14} y={y + 26}>{title}</text>
      {lines.map((l, i) => <text key={i} className="at2" x={x + 14} y={y + 50 + i * 19}>{l}</text>)}
    </g>
  )
}

function Arrow({ d, both = false, label, lx, ly, anchor = 'middle' }) {
  return (
    <g>
      <path className="aline" d={d} markerEnd="url(#arrow)" markerStart={both ? 'url(#arrow)' : undefined} />
      {label && <text className="at3" x={lx} y={ly} textAnchor={anchor}>{label}</text>}
    </g>
  )
}

export default function Architecture() {
  return (
    <div className="diagram-scroll">
      <svg className="arch" viewBox="0 0 1200 660" role="img" aria-label="System architecture: offline training, artifacts on disk, and online serving with a React front end, a FastAPI server and an inference engine"
        style={{ minWidth: 980 }}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="var(--muted)" />
          </marker>
        </defs>

        {/* ---------------- offline lane ---------------- */}
        <rect className="alane" x="10" y="12" width="1180" height="268" rx="16" />
        <text className="at4" x="28" y="36">OFFLINE · TRAINING  (python scripts)</text>

        <Box x={30} y={50} w={200} h={104} kind="data" title="Chest X-ray images" lines={['5,856 pediatric', 'radiographs (normal /', 'pneumonia)']} />
        <Box x={30} y={166} w={200} h={104} kind="data" title="Heart-disease survey" lines={['319,795 CDC 2020', 'records, 8.6 % with', 'heart disease']} />
        <Box x={270} y={50} w={290} h={104} title="Feature extraction (X-ray)" lines={['grayscale · 128×128 · contrast stretch', 'frozen DenseNet-121 → 1,024 features', 'PCA → 16 numbers in [0, 1]']} />
        <Box x={270} y={166} w={290} h={104} title="Feature preparation (heart)" lines={['encode every answer as a number', 'scale each to [0, 1] (1st-99th pct)', '16 numbers in [0, 1]']} />
        <Box x={610} y={50} w={290} h={124} kind="quantum" title="Quantum models · 16 qubits" lines={['angle encoding → ZZ feature map', 'Quantum kernel SVM (QSVM)', 'Variational circuit classifier (VQC)']} />
        <Box x={610} y={188} w={290} h={82} kind="classical" title="Classical baselines" lines={['same 16 numbers: LR · SVM · MLP · GB', '+ 1,024-feature ceiling (X-ray)']} />
        <Box x={930} y={50} w={250} h={220} title="Training & evaluation" lines={['balanced training subset', 'balanced validation set picks', 'settings and early stopping', 'held-out test set, scored once', 'balanced accuracy · sensitivity', 'specificity · AUC · F1']} />

        <Arrow d="M230 102 H270" />
        <Arrow d="M230 218 H270" />
        <path className="aline" d="M560 102 H585 V218 H560" />
        <Arrow d="M585 112 H610" />
        <Arrow d="M585 222 H610" />
        <Arrow d="M900 112 H930" />
        <Arrow d="M900 229 H930" />
        <text className="at3" x="585" y="156" textAnchor="middle" transform="rotate(-90 585 156)">16 numbers</text>

        {/* ---------------- artifacts ---------------- */}
        <Arrow d="M1055 270 V326" label="saves" lx={1075} ly={304} anchor="start" />
        <Box x={30} y={326} w={1140} h={62} kind="store" title="Artifacts (files on disk)"
          lines={['cache: images + DenseNet features  ·  models: .joblib  ·  results: .json + .npz (metrics, ROC, test predictions)  ·  reader_log.csv']} />

        {/* ---------------- online lane ---------------- */}
        <rect className="alane" x="10" y="410" width="1180" height="236" rx="16" />
        <text className="at4" x="28" y="434">ONLINE · SERVING  (python server/run.py)</text>

        <Box x={30} y={452} w={290} h={160} kind="ui" title="Browser · React app" lines={['Reading Room: viewer, blind read,', 'side-by-side findings, sign-off', 'Worklist · Model Performance', 'Cardiac Risk · Methods & Limits']} />
        <Box x={420} y={452} w={290} h={160} title="FastAPI server" lines={['POST /api/analyse  (image)', 'POST /api/heart/predict (16 answers)', 'GET /api/models · /api/worklist', 'POST /api/decision · serves the UI']} />
        <Box x={810} y={452} w={360} h={160} kind="quantum" title="Inference engine" lines={['loads models on first use', 'image → DenseNet-121 → PCA → 16 numbers', 'QSVM / VQC: exact 16-qubit GPU simulation', 'classical models: scikit-learn']} />

        <Arrow d="M320 532 H420" both label="HTTP · JSON" lx={370} ly={520} />
        <Arrow d="M710 532 H810" both label="predict" lx={760} ly={520} />
        <Arrow d="M990 388 V452" label="loads models + results" lx={1008} ly={424} anchor="start" />
        <Arrow d="M565 452 V388" label="writes reader log" lx={583} ly={424} anchor="start" />
        <text className="at3" x="30" y="632">No quantum hardware is used: the 16-qubit circuits are simulated exactly on a classical GPU.</text>
      </svg>
    </div>
  )
}
