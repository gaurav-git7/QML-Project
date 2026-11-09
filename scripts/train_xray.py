"""STEP 3 - train and evaluate every chest-X-ray model (quantum and classical) and save the results.

  Quantum   : ONE encoding, HYB16 = DenseNet features -> PCA 16 -> ZZ feature map on 16 qubits, trained as a
              quantum-kernel SVM and as a VQC.
  Classical : logistic regression, RBF-SVM, MLP on the SAME 16 PCA features, plus the 1,024 CNN features as a
              'ceiling', each trained on the balanced subset the quantum models use ('sub') and on the full
              training set ('full').
  Protocol  : hyper-parameters / early stopping use only the balanced validation set; the official test set (624
              images) is scored once, at the end of each model.

    python scripts/train_xray.py                      # everything (see README for a time estimate)
    python scripts/train_xray.py --quick              # smoke test, a few minutes, writes to artifacts_quick/
    python scripts/train_xray.py --only __qk          # only model ids containing these strings (here: the kernel SVM)
    python scripts/train_xray.py --skip-classical     # or --skip-quantum
Finished models are skipped on re-run (use --force to retrain), so an interrupted run can simply be restarted.
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

import config as C
from common import classical_models as CM
from common import data_xray as DX
from common import quantum_models as QM
from common import registry as REG
from common import results as R
from common.encoders import n_qubits_of
from common.metrics import balanced_indices
from common.runner import finish, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--per-class", type=int, default=1000, help="balanced training subset size per class")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64, help="VQC batch size (lower it if the GPU runs out of memory)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-quantum", action="store_true")
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.per_class, args.epochs = 40, 2
    only = [s for s in args.only.split(",") if s]
    want = lambda mid: (not only or any(s in mid for s in only)) and (args.force or not R.exists("xray", mid))   # noqa: E731

    cache = np.load(C.CACHE / "xray_cache.npz", allow_pickle=True)
    pix128, y, split = cache["pix128"], cache["y"], cache["split"]
    dense = np.load(C.CACHE / "xray_dense.npz")["feat"]
    tr, va, te = (np.flatnonzero(split == c) for c in (0, 1, 2))
    rng = np.random.default_rng(C.SEED)
    sub = balanced_indices(y, tr, args.per_class, rng)            # balanced training subset (quantum + 'sub' classical)
    val_sel = balanced_indices(y, va, 400, rng)                   # balanced validation set for all model selection
    print(f"train pool {len(tr)} | balanced subset {len(sub)} | validation (balanced) {len(val_sel)} | test {len(te)}")

    reps = {}

    def get_rep(name):
        if name not in reps:
            if name.startswith("PIX"):
                reps[name] = DX.pixel_rep(pix128, int(name[3:]))
            elif name == "DENSE":
                reps[name] = dense
            else:                                                 # PCA<n>
                n = int(name[3:])
                proj = DX.FeatureProjector(n).fit(dense[tr])
                joblib.dump(proj, C.MODELS / "xray" / f"projector_{n}.joblib")
                reps[name] = proj.transform(dense)
        return reps[name]

    # ------------------------------------------------------------------ classical
    if not args.skip_classical:
        for rep in REG.XRAY_CLASSICAL_REPS:
            for kind in ("lr", "svm", "mlp"):
                for regime in ("sub", "full"):
                    mid = f"xray__{rep}__{kind}__{regime}"
                    if not want(mid):
                        continue
                    X = get_rep(rep)
                    idx = sub if regime == "sub" else tr
                    print(f"[classical] {mid}  (n_train={len(idx)}, d={X.shape[1]})")
                    est, hp, vbal, secs = CM.fit_classical(kind, X[idx], y[idx], X[val_sel], y[val_sel])
                    path = C.MODELS / "xray" / f"{mid}.joblib"
                    joblib.dump(est, path)
                    meta = dict(family="classical", method=CM.NAMES[kind], representation=rep, encoding=None,
                                label=f"{CM.NAMES[kind]} - {REG.REP_LABEL[rep]}", n_qubits=None, n_params=None,
                                regime=regime, n_train=int(len(idx)), hyper=hp, train_time_s=secs, kind="sklearn",
                                model_file=rel(path))
                    finish("xray", mid, meta, est.predict_proba(X[val_sel])[:, 1], y[val_sel],
                           est.predict_proba(X[te])[:, 1], y[te], te)

    # ------------------------------------------------------------------ quantum
    if not args.skip_quantum:
        for qid, spec in REG.XRAY_Q.items():
            spec = dict(spec)
            spec["n_qubits"] = n_qubits_of(spec)
            mid_k, mid_v = f"xray__{qid}__qk", f"xray__{qid}__vqc"
            if not (want(mid_k) or want(mid_v)):
                continue
            X = get_rep(spec["rep"])
            Xtr, ytr, Xva, yva = X[sub], y[sub], X[val_sel], y[val_sel]
            base_meta = dict(family="quantum", representation=spec["rep"], encoding=qid, n_qubits=spec["n_qubits"],
                             label=f"{spec['label']} ({spec['n_qubits']} qubits)", regime="sub", n_train=int(len(sub)))

            # ---- quantum kernel SVM
            qk_path = C.MODELS / "xray" / f"{mid_k}.joblib"
            if want(mid_k) or not qk_path.exists():
                print(f"[quantum kernel] {mid_k}  ({spec['n_qubits']} qubits, {len(sub)} training images)")
                t0 = time.time()
                qk = QM.QuantumKernelSVM(spec)
                info = qk.fit_select(Xtr, ytr, Xva, yva, REG.variants_for(spec), REG.KERNEL_GAMMAS, REG.KERNEL_CS)
                qk.save(qk_path)
                print(f"    selected {info}")
                meta = dict(base_meta, method="Quantum kernel SVM", n_params=None, hyper=info,
                            train_time_s=time.time() - t0, kind="qk", model_file=rel(qk_path))
                finish("xray", mid_k, meta, qk.predict_proba(Xva), yva, qk.predict_proba(X[te]), y[te], te)
            else:
                qk = QM.QuantumKernelSVM.load(qk_path)

            # ---- variational quantum classifier (uses the encoder variant the kernel search preferred)
            if want(mid_v):
                print(f"[quantum VQC] {mid_v}  ({spec['n_qubits']} qubits)")
                t0 = time.time()
                vqc = QM.VQC(qk.spec, reps=2, entanglement="ring", readout="z0")
                info = vqc.fit(Xtr, ytr, Xva, yva, epochs=args.epochs, batch=args.batch, lr=args.lr)
                path = C.MODELS / "xray" / f"{mid_v}.joblib"
                vqc.save(path)
                meta = dict(base_meta, method="Variational quantum classifier", n_params=vqc.n_params,
                            hyper=dict(reps=2, entanglement="ring", readout="qubit 0", lr=args.lr, batch=args.batch, **info),
                            train_time_s=time.time() - t0, kind="vqc", model_file=rel(path), history=vqc.history)
                finish("xray", mid_v, meta, vqc.predict_proba(Xva), yva, vqc.predict_proba(X[te]), y[te], te)

    print("\ndone. results in", C.RESULTS / "xray")


if __name__ == "__main__":
    main()
