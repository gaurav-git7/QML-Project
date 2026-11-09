"""Model loading and prediction for the dashboard. Everything is lazy: a model is loaded the first time it is used."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import config as C
from common import data_heart as DH
from common import data_xray as DX
from common import features_cnn as FC
from common import quantum_models as QM
from common import results as R


def _key(r):
    return (r["val"]["balanced_accuracy"], r["val"]["auc"])


def _name(p):
    return str(p).replace("\\", "/").split("/")[-1]


class Engine:
    def __init__(self):
        self.dev = QM.get_device()
        self._models, self._densenet, self._proj, self._proc, self._paths, self._wl = {}, None, {}, None, None, None
        self.reload()

    def reload(self):
        self.res = {m: R.load_results(m) for m in ("xray", "heart")}

    def records(self, module):
        return self.res[module]

    def get(self, module, mid):
        return next((r for r in self.res[module] if r["id"] == mid), None)

    # ------------------------------------------------------------ which models the dashboard highlights
    def primary(self, module):
        """Best quantum model (validation balanced accuracy), its classical counterpart on the same input, and the best
        classical model of any input."""
        recs = self.res[module]
        q = [r for r in recs if r["family"] == "quantum"]
        c = [r for r in recs if r["family"] == "classical"]
        out = {}
        if q:
            out["quantum"] = max(q, key=_key)
        if c:
            out["classical_best"] = max(c, key=_key)
        if q and c:
            same = [r for r in c if r["representation"] == out["quantum"]["representation"] and r["regime"] == "sub"]
            if same:
                out["classical_same"] = max(same, key=_key)
        return out

    # ------------------------------------------------------------ model loading
    def _model(self, rec):
        mid = rec["id"]
        if mid not in self._models:
            path = C.ARTIFACTS / rec["model_file"]
            if rec["kind"] == "sklearn":
                self._models[mid] = joblib.load(path)
            elif rec["kind"] == "qk":
                self._models[mid] = QM.QuantumKernelSVM.load(path, self.dev)
            else:
                self._models[mid] = QM.VQC.load(path, self.dev)
        return self._models[mid]

    def _predict(self, rec, X):
        m = self._model(rec)
        p = m.predict_proba(X)
        p = p[:, 1] if rec["kind"] == "sklearn" else p
        return float(p[0])

    # ------------------------------------------------------------ chest X-ray
    def _xray_inputs(self, pil, reps):
        arr = DX.to_gray_uint8(pil)
        a128 = DX.to_square(arr, C.XRAY_STORE_SIZE)[None]
        out, dense = {}, None
        for rep in reps:
            if rep.startswith("PIX"):
                out[rep] = DX.pixel_rep(a128, int(rep[3:]))
                continue
            if dense is None:
                if self._densenet is None:
                    self._densenet = FC.load_densenet(self.dev)
                a224 = DX.to_square(arr, C.DENSENET_SIZE)[None]
                dense = FC.dense_features(self._densenet, a224, self.dev)
            if rep == "DENSE":
                out[rep] = dense
            else:
                n = int(rep[3:])
                if n not in self._proj:
                    self._proj[n] = joblib.load(C.MODELS / "xray" / f"projector_{n}.joblib")
                out[rep] = self._proj[n].transform(dense)
        return out

    def predict_xray(self, pil, recs):
        inputs = self._xray_inputs(pil, {r["representation"] for r in recs})
        return {r["id"]: self._predict(r, inputs[r["representation"]]) for r in recs}

    # ------------------------------------------------------------ heart disease
    def predict_heart(self, values, recs):
        if self._proc is None:
            self._proc = joblib.load(C.MODELS / "heart" / "processor.joblib")
        xs = self._proc.scale(DH.row_to_raw(values))
        inputs = {"H16": xs}
        return {r["id"]: self._predict(r, inputs[r["representation"]]) for r in recs}

    # ------------------------------------------------------------ worklist (held-out test studies)
    def _cache_paths(self):
        if self._paths is None:
            self._paths = np.load(C.CACHE / "xray_cache.npz", allow_pickle=True)["paths"]
        return self._paths

    def study_path(self, ds_index):
        return C.DATA / str(self._cache_paths()[int(ds_index)])

    def worklist(self):
        """DataFrame of every held-out test study with the precomputed outputs of the primary models."""
        if self._wl is not None:
            return self._wl
        prim = self.primary("xray")
        if "quantum" not in prim:
            return pd.DataFrame()
        qp, y, idx = R.load_probs("xray", prim["quantum"]["id"])
        cl = prim.get("classical_same") or prim.get("classical_best")
        cp = R.load_probs("xray", cl["id"])[0] if cl else np.full(len(qp), np.nan)
        paths = self._cache_paths()
        df = pd.DataFrame({
            "ID": idx,
            "Study": [_name(paths[i]) for i in idx],
            "Reference": np.where(y == 1, "PNEUMONIA", "NORMAL"),
            "qProb": np.round(np.asarray(qp, dtype=float), 3),
            "cProb": np.round(np.asarray(cp, dtype=float), 3),
        })
        df["qRead"] = np.where(df["qProb"] >= 0.5, "PNEUMONIA", "NORMAL")
        df["cRead"] = np.where(df["cProb"] >= 0.5, "PNEUMONIA", "NORMAL")
        df["agree"] = df["qRead"] == df["cRead"]
        df["qCorrect"] = df["qRead"] == df["Reference"]
        df["cCorrect"] = df["cRead"] == df["Reference"]
        self._wl = df
        return df

    def warm_up(self):
        """Load the DenseNet and the highlighted models so the first analysis in the browser is fast."""
        prim = self.primary("xray")
        recs = list(prim.values())
        if recs:
            self.predict_xray(Image.new("RGB", (256, 256), 128), recs)
