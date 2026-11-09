# QuantumML Diagnostic Workstation

Quantum vs classical machine learning for two real datasets in `../data/`, with a React dashboard.

| Module | Data | Task |
|---|---|---|
| Chest X-ray | `data/chest_xray` - 5,856 images | normal vs pneumonia from an image |
| Heart disease | `data/heart_disease/2020` - 319,795 survey rows | heart disease yes/no from patient history |

No synthetic data is used anywhere.

## After cloning this repository

The repository contains the code, the built front end, and the trained models and results (`artifacts/models`, `artifacts/results`).
It does **not** contain the datasets or the image cache (`artifacts/cache`, 89 MB, derived from the datasets):

1. `pip install -r requirements.txt`
2. Download the two Kaggle datasets and place them next to this folder as
   `../data/chest_xray/chest_xray/{train,val,test}/...` (`paultimothymooney/chest-xray-pneumonia`) and
   `../data/heart_disease/2020/heart_2020_cleaned.csv` (`kamilpytlak/personal-key-indicators-of-heart-disease`).
3. `python scripts/prepare_xray.py` rebuilds the image cache (needed by the worklist and by re-training).
4. `python server/run.py`

## Start the dashboard

```bash
cd C:\College_projects\Sem-7\Quantum\Project\qml_health
python server/run.py              # then open http://127.0.0.1:8000
```

The trained models are already in `artifacts/`. The React app is already built (`frontend/dist`), so no Node is needed to use it.

To change the front end:

```bash
cd frontend
npm install          # first time only
npm run build        # rebuild frontend/dist (the Python server serves it)
# or, for live editing:  python server/run.py   (in one terminal)   +   npm run dev   (in another, http://localhost:5173)
```

## The final design: one quantum encoding per module, 16 qubits

```
X-ray : image -> frozen DenseNet-121 (1,024 features) -> PCA -> 16 numbers in [0,1] --\
                                                                                        >-> ZZ feature map (16 qubits) -> QSVM / VQC
Heart : 16 survey features scaled to [0,1] --------------------------------------------/
```

* **Encoding / feature map:** each number rotates one qubit (angle encoding); the ZZ feature map (2 repetitions, linear
  entanglement) adds Hadamards, phase rotations and entangling ZZ terms. This is the winner of experiments 1-4.
* **Quantum kernel SVM (QSVM):** the fidelity between two encoded states is the kernel of an SVM. Encoder bandwidth,
  kernel scale and C are chosen on the validation set. The learning is classical; the quantum part is a fixed feature map.
* **Variational quantum classifier (VQC):** feature map + trainable RZ-RY-RZ / CNOT-ring circuit (2 repetitions, 144 angles),
  one-qubit readout, 2-parameter classical scale/bias head, trained with Adam on an exact GPU simulation. It reuses the
  bandwidth chosen by the QSVM search.
* **Classical models** see exactly the same 16 numbers (logistic regression, RBF-SVM, MLP, + gradient boosting for heart),
  each trained on the balanced subset the quantum models use ('sub') and on the full training set ('full').
  For the X-rays there is also a "ceiling" row using all 1,024 CNN features.

Protocol: model selection only on a balanced validation set; the held-out test set (the official 624 test X-rays; a balanced
4,000-patient sample of the heart test split) is scored once per model.

## Re-training (optional - the models are already trained)

```bash
python scripts/prepare_xray.py        # reads all images, DenseNet-121 features on the GPU
python scripts/verify_encoders.py     # seconds: torch encoders / ansatz == Qiskit circuits
python scripts/train_xray.py          # X-ray models
python scripts/train_heart.py         # heart models
```

On an RTX 4060 laptop the whole thing took only a few minutes. Add `--quick` to any script for a tiny smoke test that writes
to `artifacts_quick/` (and `python server/run.py --quick` to view it). Useful flags: `--only <id substring>`,
`--skip-quantum`, `--skip-classical`, `--epochs N`, `--batch N`, `--force`. Finished models are skipped on re-run.

## Dashboard pages

Light and dark themes (the switch is at the bottom of the left navigation; it follows your system setting by default),
a labelled navigation rail on desktop and a bottom tab bar on phones.

* **Reading Room** - a one-screen workstation. Open an image (button, drag-and-drop anywhere, or paste with Ctrl+V) or load a random held-out
  test case. The viewer has window level / width, invert, zoom (wheel) and pan (drag). Models analyse automatically; findings show each
  model's probability against the 50% threshold, an agreement banner ("clinician review needed" when they disagree), and the
  reader signs off with a note and moves on with "Next case". **Blind read:** a test case's file name and reference label stay hidden until
  the models have read it. Keyboard shortcuts: `O` open, `N` next case, `A` analyse, `I` invert, `R` reset view, `+` / `-` zoom.
* **Worklist & Validation** - every held-out test case (anonymised case numbers) with both models' outputs; filter, search, sort,
  export CSV, open a case with a click or Enter.
* **Model Performance** - quantum vs classical on the same input, a ranked bar chart, ROC curves, confusion matrices, a glossary of the
  metrics and the full table.
* **Cardiac Risk** - a grouped form for the 16 survey features with units and validation; same side-by-side output.
* **Methods & Limits** - a system architecture diagram (offline training, artifacts, online serving) and the complete 16-qubit circuit of the
  trained X-ray and heart models, as a compact block view or a full gate-by-gate view. The settings, gate counts and depth are read from the trained
  models through `GET /api/circuit/{xray|heart}` (counted from a real Qiskit circuit), then the protocol and dataset caveats.

## Layout

```
qml_health/
  config.py                paths (artifacts/ or artifacts_quick/), constants
  common/                  data_xray, data_heart, features_cnn, encoders, qsim, quantum_models,
                           classical_models, registry, results, runner, metrics
  scripts/                 prepare_xray, verify_encoders, train_xray, train_heart
  server/                  FastAPI backend: main.py (API), engine.py (model loading/prediction), run.py
  frontend/                React (Vite) app: src/pages, src/components, dist/ (built)
  artifacts/               generated: cache/, models/, results/, reader_log.csv
```

## Honest notes

* The quantum circuits are simulated exactly on a classical computer: no hardware noise, no speed claims.
* In the X-ray pipeline the CNN does most of the image work; the quantum models only see 16 numbers.
* The X-ray set is a public pediatric collection from a single source and the heart set is a self-reported survey: results
  will not transfer to clinical practice. The dashboard says so on every page.
* On the X-ray data, validation scores are much higher than test scores (the official test set is known to be harder).
* The VQC readout is a single qubit (to avoid barren plateaus) followed by a 2-parameter classical scale/bias head.
