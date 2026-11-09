"""STEP 4 - train and evaluate every heart-disease model (quantum and classical) and save the results.

  Data      : CDC survey table, 319,795 rows, 8.56 % positive. Stratified 70 / 15 / 15 train / validation / test.
  Inputs    : the 16 numeric survey features scaled to [0,1] (one qubit per feature).
  Quantum   : ZZ feature map on 16 qubits, as a quantum-kernel SVM and as a VQC (balanced subset).
  Classical : logistic regression, RBF-SVM (balanced subset only), MLP, gradient boosting - same 16 features.
  Protocol  : model selection on a balanced validation sample; every model is scored once on the same balanced
              test sample (all models see exactly the same test patients).

    python scripts/train_heart.py
    python scripts/train_heart.py --quick
    python scripts/train_heart.py --skip-classical    # quantum models only
"""
import os
import sys

if "--quick" in sys.argv:
    os.environ["QML_QUICK"] = "1"
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

import config as C
from common import classical_models as CM
from common import data_heart as DH
from common import quantum_models as QM
from common import registry as REG
from common import results as R
from common.encoders import n_qubits_of
from common.metrics import balanced_indices
from common.runner import finish, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--test-per-class", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-quantum", action="store_true")
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.per_class, args.test_per_class, args.epochs = 60, 200, 2
    only = [s for s in args.only.split(",") if s]
    want = lambda mid: (not only or any(s in mid for s in only)) and (args.force or not R.exists("heart", mid))   # noqa: E731

    X_raw, y = DH.load_heart()
    print(f"{len(y):,} records, {y.mean():.2%} positive, {X_raw.shape[1]} features")
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=C.SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=C.SEED)
    rng = np.random.default_rng(C.SEED)
    sub = balanced_indices(y, tr, args.per_class, rng)
    val_sel = balanced_indices(y, va, 1000 if not args.quick else 100, rng)
    te_sel = balanced_indices(y, te, args.test_per_class, rng)
    print(f"train {len(tr):,} | balanced subset {len(sub)} | validation (balanced) {len(val_sel)} | test (balanced) {len(te_sel)}")

    proc = DH.HeartProcessor().fit(X_raw[tr])
    Xs = proc.scale(X_raw)
    joblib.dump(proc, C.MODELS / "heart" / "processor.joblib")
    reps = {"H16": Xs}                                          # all 16 features, one qubit each

    # ------------------------------------------------------------------ classical
    if not args.skip_classical:
        for rep in ("H16",):
            X = reps[rep]
            for kind in ("lr", "svm", "mlp", "hgb"):
                for regime in ("sub", "full"):
                    if kind == "svm" and regime == "full":
                        continue                                    # RBF-SVM on 224k rows is impractical
                    mid = f"heart__{rep}__{kind}__{regime}"
                    if not want(mid):
                        continue
                    ids = sub if regime == "sub" else tr
                    print(f"[classical] {mid}  (n_train={len(ids):,})")
                    est, hp, vbal, secs = CM.fit_classical(kind, X[ids], y[ids], X[val_sel], y[val_sel])
                    path = C.MODELS / "heart" / f"{mid}.joblib"
                    joblib.dump(est, path)
                    meta = dict(family="classical", method=CM.NAMES[kind], representation=rep, encoding=None,
                                label=f"{CM.NAMES[kind]} - {REG.REP_LABEL[rep]}", n_qubits=None, n_params=None,
                                regime=regime, n_train=int(len(ids)), hyper=hp, train_time_s=secs, kind="sklearn",
                                model_file=rel(path))
                    finish("heart", mid, meta, est.predict_proba(X[val_sel])[:, 1], y[val_sel],
                           est.predict_proba(X[te_sel])[:, 1], y[te_sel], te_sel)

    # ------------------------------------------------------------------ quantum
    if not args.skip_quantum:
        for qid, spec in REG.HEART_Q.items():
            spec = dict(spec)
            spec["n_qubits"] = n_qubits_of(spec)
            mid_k, mid_v = f"heart__{qid}__qk", f"heart__{qid}__vqc"
            if not (want(mid_k) or want(mid_v)):
                continue
            X = reps[spec["rep"]]
            Xtr, ytr, Xva, yva = X[sub], y[sub], X[val_sel], y[val_sel]
            base_meta = dict(family="quantum", representation=spec["rep"], encoding=qid, n_qubits=spec["n_qubits"],
                             label=f"{spec['label']} ({spec['n_qubits']} qubits)", regime="sub", n_train=int(len(sub)))

            qk_path = C.MODELS / "heart" / f"{mid_k}.joblib"
            if want(mid_k) or not qk_path.exists():
                print(f"[quantum kernel] {mid_k}  ({spec['n_qubits']} qubits, {len(sub)} training patients)")
                t0 = time.time()
                qk = QM.QuantumKernelSVM(spec)
                info = qk.fit_select(Xtr, ytr, Xva, yva, REG.variants_for(spec), REG.KERNEL_GAMMAS, REG.KERNEL_CS)
                qk.save(qk_path)
                print(f"    selected {info}")
                meta = dict(base_meta, method="Quantum kernel SVM", n_params=None, hyper=info,
                            train_time_s=time.time() - t0, kind="qk", model_file=rel(qk_path))
                finish("heart", mid_k, meta, qk.predict_proba(Xva), yva, qk.predict_proba(X[te_sel]), y[te_sel], te_sel)
            else:
                qk = QM.QuantumKernelSVM.load(qk_path)

            if want(mid_v):
                print(f"[quantum VQC] {mid_v}  ({spec['n_qubits']} qubits)")
                t0 = time.time()
                vqc = QM.VQC(qk.spec, reps=2, entanglement="ring", readout="z0")
                info = vqc.fit(Xtr, ytr, Xva, yva, epochs=args.epochs, batch=args.batch, lr=args.lr)
                path = C.MODELS / "heart" / f"{mid_v}.joblib"
                vqc.save(path)
                meta = dict(base_meta, method="Variational quantum classifier", n_params=vqc.n_params,
                            hyper=dict(reps=2, entanglement="ring", readout="qubit 0", lr=args.lr, batch=args.batch, **info),
                            train_time_s=time.time() - t0, kind="vqc", model_file=rel(path), history=vqc.history)
                finish("heart", mid_v, meta, vqc.predict_proba(Xva), yva, vqc.predict_proba(X[te_sel]), y[te_sel], te_sel)

    print("\ndone. results in", C.RESULTS / "heart")


if __name__ == "__main__":
    main()
