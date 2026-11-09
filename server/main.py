"""FastAPI backend for the React dashboard.

    python server/run.py            # serves the API and the built React app on http://127.0.0.1:8000
"""
import csv
import io
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import config as C
from common import data_heart as DH
from common import registry as REG
from server.engine import Engine

ENGINE = Engine()
LOCK = threading.Lock()                       # one prediction at a time on the GPU
app = FastAPI(title="QuantumML Diagnostic Workstation API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_KEEP = ("id", "label", "method", "family", "representation", "encoding", "n_qubits", "n_params", "regime", "n_train",
         "train_time_s", "val", "test", "roc", "hyper")
_ROLES = [("quantum", "QUANTUM · primary"), ("classical_same", "CLASSICAL · same input"),
          ("classical_best", "CLASSICAL · best available")]
_STATE = {"ready": False}


def slim(r):
    out = {k: r.get(k) for k in _KEEP}
    out["rep_label"] = REG.REP_LABEL.get(r["representation"], r["representation"])
    return out


def role_list(module):
    prim, out, seen = ENGINE.primary(module), [], set()
    for key, label in _ROLES:
        r = prim.get(key)
        if r and r["id"] not in seen:
            out.append(dict(key=key, role=label, id=r["id"]))
            seen.add(r["id"])
    return out


def _warm():
    try:
        ENGINE.warm_up()
    except Exception as e:                                                       # noqa: BLE001
        print("warm-up skipped:", e)
    _STATE["ready"] = True


threading.Thread(target=_warm, daemon=True).start()


# ---------------------------------------------------------------------- general
@app.get("/api/health")
def health():
    return dict(ready=_STATE["ready"], quick=C.QUICK, xray_models=len(ENGINE.records("xray")),
                heart_models=len(ENGINE.records("heart")), disclaimer=C.DISCLAIMER)


@app.get("/api/models/{module}")
def models(module: str):
    if module not in ("xray", "heart"):
        raise HTTPException(404, "unknown module")
    return [slim(r) for r in ENGINE.records(module)]


@app.get("/api/primary/{module}")
def primary(module: str):
    if module not in ("xray", "heart"):
        raise HTTPException(404, "unknown module")
    return dict(roles=role_list(module))


# ---------------------------------------------------------------------- chest X-ray
@app.get("/api/worklist")
def worklist():
    df = ENGINE.worklist()
    if df.empty:
        return []
    df = df.rename(columns={"ID": "id", "Study": "study", "Reference": "reference"}).astype(object)
    return df.where(pd.notna(df), None).to_dict(orient="records")


@app.get("/api/study/{study_id}/image")
def study_image(study_id: int):
    df = ENGINE.worklist()
    if df.empty or study_id not in set(df["ID"].tolist()):
        raise HTTPException(404, "study not found")
    return FileResponse(ENGINE.study_path(study_id))


@app.post("/api/analyse")
def analyse(file: UploadFile | None = File(default=None), study_id: int | None = Form(default=None),
            run_all: bool = Form(default=False)):
    if not ENGINE.records("xray"):
        raise HTTPException(503, "No trained X-ray models found. Run scripts/train_xray.py first.")
    reference = None
    if study_id is not None:
        df = ENGINE.worklist()
        row = df[df["ID"] == study_id]
        if row.empty:
            raise HTTPException(404, "study not found")
        reference = str(row.iloc[0]["Reference"])
        img = Image.open(ENGINE.study_path(study_id)).convert("RGB")
    elif file is not None:
        try:
            img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        except Exception:                                                        # noqa: BLE001
            raise HTTPException(400, "That file could not be read as an image.")
    else:
        raise HTTPException(400, "Send an image file or a study_id.")
    roles = role_list("xray")
    recs = ENGINE.records("xray") if run_all else [ENGINE.get("xray", r["id"]) for r in roles]
    with LOCK:
        probs = ENGINE.predict_xray(img, recs)
    return dict(reference=reference, size=list(img.size), roles=[dict(r, prob=probs[r["id"]]) for r in roles],
                predictions=[dict(id=r["id"], prob=probs[r["id"]]) for r in recs])


# ---------------------------------------------------------------------- heart disease
_NUM_RANGE = {"BMI": (12, 95, 27.0), "PhysicalHealth": (0, 30, 0), "MentalHealth": (0, 30, 0), "SleepTime": (1, 24, 7)}
_OPTIONS = {"yn": (["No", "Yes"], "No"), "sex": (["Female", "Male"], "Female"), "age": (DH.AGE_ORDER, "55-59"),
            "gen": (DH.GEN_ORDER, "Good"), "diab": (DH.DIABETIC_ORDER, "No")}


@app.get("/api/heart/schema")
def heart_schema():
    out = []
    for name, kind, label in DH.FEATURES:
        f = dict(name=name, kind=kind, label=label)
        if kind == "num":
            lo, hi, default = _NUM_RANGE[name]
            f.update(min=lo, max=hi, default=default)
        else:
            f["options"], f["default"] = _OPTIONS[kind]
        out.append(f)
    return out


@app.post("/api/heart/predict")
def heart_predict(values: dict = Body(...)):
    if not ENGINE.records("heart"):
        raise HTTPException(503, "No trained heart models found. Run scripts/train_heart.py first.")
    missing = [n for n in DH.FEATURE_NAMES if n not in values or values[n] in (None, "")]
    if missing:
        raise HTTPException(400, f"Missing values: {', '.join(missing)}")
    try:
        clean = {n: (float(values[n]) if k == "num" else values[n]) for n, k, _ in DH.FEATURES}
    except (TypeError, ValueError):
        raise HTTPException(400, "Numeric fields must be numbers.")
    roles = role_list("heart")
    with LOCK:
        probs = ENGINE.predict_heart(clean, ENGINE.records("heart"))
    return dict(roles=[dict(r, prob=probs[r["id"]]) for r in roles],
                predictions=[dict(id=r["id"], prob=probs[r["id"]]) for r in ENGINE.records("heart")])


# ---------------------------------------------------------------------- reader sign-off log
@app.post("/api/decision")
def decision(body: dict = Body(...)):
    if not body.get("decision"):
        raise HTTPException(400, "Choose the reader's decision first.")
    path = C.ARTIFACTS / "reader_log.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "reference_label", "quantum_p_pneumonia", "classical_p_pneumonia", "reader_decision", "notes"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), body.get("reference") or "", body.get("quantum"),
                    body.get("classical"), body["decision"], body.get("notes") or ""])
    return dict(logged=str(path))


# ---------------------------------------------------------------------- the quantum circuit of the trained models
_CIRCUIT_CACHE = {}


def _build_circuit_info(module):
    """Configuration of the trained 16-qubit model of a module, with gate counts and depth taken from a real Qiskit circuit
    (feature map + ansatz + measurement of qubit 0), laid out as drawn on the Methods page: the ZZ terms are diagonal and
    commute, so they are applied as an even-pairs layer and an odd-pairs layer (verified equal to the model's simulator)."""
    import numpy as np
    from qiskit import QuantumCircuit

    recs = ENGINE.records(module)
    qk = next((r for r in recs if r.get("kind") == "qk"), None)
    vq = next((r for r in recs if r.get("kind") == "vqc"), None)
    if not qk or not vq:
        raise HTTPException(404, f"No trained quantum models for {module}.")
    n, fm_reps, an_reps = qk["n_qubits"], 2, 2
    scale = float(qk["hyper"].get("best_variant", {}).get("fm_scale", 1.0))
    th = 0.5 * scale * np.pi / 2

    fm = QuantumCircuit(n)
    for _ in range(fm_reps):
        fm.h(range(n))
        for q in range(n):
            fm.p(2 * th, q)
        for parity in (0, 1):
            for i in range(parity, n - 1, 2):
                fm.cx(i, i + 1)
                fm.p(2 * (np.pi - th) ** 2, i + 1)
                fm.cx(i, i + 1)
    an = QuantumCircuit(n)
    for layer in range(an_reps + 1):
        for name in ("rz", "ry", "rz"):
            for q in range(n):
                getattr(an, name)(0.1, q)
        if layer < an_reps:
            for q in range(n - 1):
                an.cx(q, q + 1)
            an.cx(n - 1, 0)
    full = QuantumCircuit(n, 1)
    full.compose(fm, inplace=True)
    full.compose(an, inplace=True)
    full.measure(0, 0)
    ops = lambda c: {k: int(v) for k, v in c.count_ops().items()}                # noqa: E731
    names = DH.FEATURE_NAMES if module == "heart" else [f"PC{i + 1}" for i in range(n)]
    return dict(
        module=module, n_qubits=n, fm_scale=scale, fm_reps=fm_reps, ansatz_reps=an_reps, inputs=names,
        counts=dict(feature_map=ops(fm), ansatz=ops(an), total=ops(full)),
        depth=dict(feature_map=fm.depth(), ansatz=an.depth(), total=full.depth()),
        trainable_angles=3 * n * (an_reps + 1), head_params=2,
        qsvm=dict(gamma=qk["hyper"].get("gamma"), C=qk["hyper"].get("C"), support_vectors=qk["hyper"].get("n_support"),
                  n_train=qk["n_train"], test_balanced_accuracy=qk["test"]["balanced_accuracy"]),
        vqc=dict(epochs_run=vq["hyper"].get("epochs_run"), lr=vq["hyper"].get("lr"), batch=vq["hyper"].get("batch"),
                 n_train=vq["n_train"], test_balanced_accuracy=vq["test"]["balanced_accuracy"]))


@app.get("/api/circuit/{module}")
def circuit_info(module: str):
    if module not in ("xray", "heart"):
        raise HTTPException(404, "unknown module")
    if module not in _CIRCUIT_CACHE:
        _CIRCUIT_CACHE[module] = _build_circuit_info(module)
    return _CIRCUIT_CACHE[module]


# ---------------------------------------------------------------------- the built React app
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
