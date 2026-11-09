import { useEffect, useState } from 'react'
import { api } from '../api.js'
import Architecture from '../components/Architecture.jsx'
import QuantumCircuit from '../components/QuantumCircuit.jsx'
import { PageHeader } from '../components/common.jsx'

export default function Methods({ active }) {
  const [mod, setMod] = useState('xray')
  const [info, setInfo] = useState({})
  const [error, setError] = useState('')

  useEffect(() => {
    if (!active || info[mod]) return
    api.circuit(mod).then((d) => setInfo((s) => ({ ...s, [mod]: d }))).catch((e) => setError(e.message))
  }, [active, mod, info])

  return (
    <div className={active ? '' : 'hidden'}>
      <PageHeader active={active} title="Methods & Limits" subtitle="The system architecture, the quantum circuits of the trained models, how the numbers were obtained, and what they do not mean." />
      <div className="stack">
        <section className="card" aria-label="System architecture">
          <div className="card-h"><h2>System architecture</h2></div>
          <p className="note" style={{ marginBottom: 12 }}>Models are trained offline by Python scripts and saved to disk. The server loads them to answer requests from the browser.</p>
          <Architecture />
        </section>

        <section className="card" aria-label="Quantum circuits">
          <div className="card-h">
            <h2>Complete quantum circuit of the trained models</h2>
            <div className="seg" role="group" aria-label="Dataset">
              {[['xray', 'Chest X-ray'], ['heart', 'Heart disease']].map(([k, l]) => <button key={k} aria-pressed={mod === k} onClick={() => setMod(k)}>{l}</button>)}
            </div>
          </div>
          <p className="note" style={{ marginBottom: 6 }}>
            Both modules use the same 16-qubit circuit structure; what differs is where the 16 input numbers come from and the input scale.
            Gate counts, depth and settings below are read from the trained models, not typed in.
          </p>
          {error && <div className="msg err" role="alert">{error}</div>}
          <QuantumCircuit key={mod} module={mod} info={info[mod]} />
        </section>

        <section className="card prose" aria-label="Pipeline, protocol and limits">
          <h2>How each part works</h2>
          <ul>
            <li><b>Encoding:</b> each of the 16 numbers sets the rotation of one qubit (angle encoding); the <b>ZZ feature map</b> then adds Hadamards, phase rotations and entangling ZZ terms between neighbouring qubits (2 repetitions) to produce a richer quantum state.</li>
            <li><b>Quantum kernel SVM (QSVM):</b> two patients or images are compared by the overlap (fidelity) of their quantum states; an SVM uses that as its kernel.</li>
            <li><b>Variational quantum classifier (VQC):</b> a trainable circuit (Rz·Ry·Rz rotations and a CNOT ring, 2 repetitions) follows the feature map; qubit 0 is measured and a 2-parameter classical head turns it into a probability.</li>
            <li><b>Classical models</b> (logistic regression, RBF-SVM, neural network, and gradient boosting for heart) see exactly the same 16 numbers, plus a &quot;ceiling&quot; using all 1,024 CNN features for the X-rays.</li>
          </ul>
          <h2>How the numbers were obtained</h2>
          <ul>
            <li>Hyper-parameters and early stopping use only a <b>balanced validation set</b>; the <b>held-out test set</b> is scored once.</li>
            <li>Quantum models train on a class-balanced subset (the simulation is expensive); classical models are reported on the same subset and on the full training set.</li>
            <li>Quantum circuits are simulated <b>exactly on a classical computer</b> (16 qubits). There is no hardware noise and no speed claim.</li>
            <li>Balanced accuracy, sensitivity (recall for disease) and specificity are reported because both datasets are imbalanced.</li>
            <li>In the Reading Room the dataset label and file name stay hidden until the models have read the study, so a test case can be read blind.</li>
          </ul>
          <h2>Limits you should keep in mind</h2>
          <ul>
            <li><b>Chest X-ray data:</b> a public pediatric pneumonia-vs-normal collection reported to come from a single medical centre. The pneumonia label mixes bacterial and viral cases, and the official test set is known to be harder than the training set (validation scores are much higher than test scores). Results will not transfer to adult patients or other scanners without new validation.</li>
            <li><b>Heart-disease data:</b> self-reported answers from the 2020 CDC telephone health survey, not clinical measurements; it estimates risk, it does not diagnose.</li>
            <li>In the X-ray pipeline the CNN does most of the image work; the quantum model only sees 16 numbers.</li>
            <li>Probabilities are not calibrated to any hospital&apos;s disease prevalence, and a model can be confidently wrong.</li>
            <li>This is a research prototype and <b>not a medical device</b>.</li>
          </ul>
        </section>
      </div>
    </div>
  )
}
