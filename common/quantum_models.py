"""The two quantum classifiers.

QuantumKernelSVM : the fidelity of two encoded states, F(x,y) = |<psi(x)|psi(y)>|^2, is the kernel of an SVM
                   (the kernel is rescaled by a monotone map exp(gamma (F-1)/s) chosen on the validation set).
VQC              : variational quantum classifier = encoder + ring-entangled RZ-RY-RZ ansatz + local readout, trained
                   with Adam on an exact differentiable simulation. A 2-parameter classical head (scale, bias) turns the
                   readout probability into a logit.
Both expose predict_proba(X) -> P(class 1) for X = the classical inputs in [0,1] (pixels, PCA features, ...).
"""
import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F_
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.svm import SVC

from .encoders import make_encoder, prepare_spec, n_qubits_of
from .qsim import Ansatz


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_t(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


# ====================================================================== quantum kernel SVM
class QuantumKernelSVM:
    def __init__(self, spec, device=None):
        self.spec = dict(spec)
        self.dev = device or get_device()
        self.history = []

    # ---- states / fidelities
    @torch.no_grad()
    def _states(self, enc, X, chunk=256):
        return torch.cat([enc(_to_t(X[i:i + chunk], self.dev)) for i in range(0, len(X), chunk)])

    @torch.no_grad()
    def _fidelity(self, enc, XA, S_B, chunk=256):
        rows = []
        for i in range(0, len(XA), chunk):
            ov = enc(_to_t(XA[i:i + chunk], self.dev)) @ S_B.conj().T
            rows.append((ov.real ** 2 + ov.imag ** 2).double().cpu().numpy())
        return np.concatenate(rows)

    @staticmethod
    def _kernel(F, gamma, s):
        return F if gamma == 0 else np.exp(gamma * (F - 1.0) / s)

    # ---- training: choose encoder variant, gamma and C on the validation set
    def fit_select(self, Xtr, ytr, Xva, yva, variants, gammas, Cs, log=print):
        best = None
        for var in variants:
            spec = prepare_spec({**self.spec, **var}, Xtr)
            enc = make_encoder(spec, self.dev)
            t0 = time.time()
            S_tr = self._states(enc, Xtr)
            Ftr = self._fidelity(enc, Xtr, S_tr)
            Fva = self._fidelity(enc, Xva, S_tr)
            del S_tr
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
            off = Ftr[~np.eye(len(Ftr), dtype=bool)]
            s = float(max(off.std(), 1e-6))
            log(f"    variant {var or '-'}: gram built in {time.time() - t0:.0f}s, "
                f"off-diagonal fidelity mean {off.mean():.4f} std {s:.4f}")
            for g in gammas:
                Ktr, Kva = self._kernel(Ftr, g, s), self._kernel(Fva, g, s)
                for C in Cs:
                    svc = SVC(kernel="precomputed", C=C).fit(Ktr, ytr)
                    d = svc.decision_function(Kva)
                    bal = balanced_accuracy_score(yva, (d > 0).astype(int))
                    auc = roc_auc_score(yva, d) if len(np.unique(yva)) == 2 else 0.0
                    self.history.append(dict(variant=str(var), gamma=g, C=C, val_balanced_accuracy=float(bal)))
                    if best is None or (bal, auc) > (best["bal"], best["auc"]):
                        best = dict(bal=bal, auc=auc, spec=spec, gamma=g, C=C, s=s, svc=svc, d_va=d, var=var)
        # keep only what inference needs
        svc = best["svc"]
        self.spec, self.gamma, self.C, self.s = best["spec"], best["gamma"], best["C"], best["s"]
        self.sv_X = np.asarray(Xtr)[svc.support_].astype(np.float32)
        self.dual = svc.dual_coef_[0].astype(np.float64)
        self.b = float(svc.intercept_[0])
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(best["d_va"].reshape(-1, 1), yva)
        self.cal_a, self.cal_b = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        return dict(best_variant=best["var"], gamma=best["gamma"], C=best["C"], n_support=len(self.sv_X),
                    val_balanced_accuracy=float(best["bal"]))

    # ---- inference
    @torch.no_grad()
    def decision_function(self, X):
        enc = make_encoder(self.spec, self.dev)
        S_sv = self._states(enc, self.sv_X)
        K = self._kernel(self._fidelity(enc, X, S_sv), self.gamma, self.s)
        return K @ self.dual + self.b

    def predict_proba(self, X):
        d = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-(self.cal_a * d + self.cal_b)))

    def save(self, path):
        joblib.dump({k: getattr(self, k) for k in ("spec", "gamma", "C", "s", "sv_X", "dual", "b", "cal_a", "cal_b",
                                                    "history")}, path)

    @classmethod
    def load(cls, path, device=None):
        d = joblib.load(path)
        m = cls(d["spec"], device)
        for k, v in d.items():
            setattr(m, k, v)
        return m


# ====================================================================== variational quantum classifier
class VQC:
    def __init__(self, spec, reps=2, entanglement="ring", readout="z0", seed=0, device=None):
        self.spec, self.reps, self.entanglement, self.readout, self.seed = dict(spec), reps, entanglement, readout, seed
        self.dev = device or get_device()
        self.theta = None
        self.w, self.b = 3.0, 0.0
        self.history = []

    def _build(self):
        self.enc = make_encoder(self.spec, self.dev)
        self.ans = Ansatz(n_qubits_of(self.spec), self.reps, self.entanglement, self.readout, self.dev)

    @property
    def n_params(self):
        return (self.reps + 1) * 3 * n_qubits_of(self.spec) + 2

    def _logit(self, x, theta, w, b):
        st = self.ans.apply(self.enc(x), theta)
        return w * (2.0 * self.ans.class1_prob(st) - 1.0) + b

    @torch.no_grad()
    def _predict_logits(self, X, batch=128):
        theta = torch.as_tensor(self.theta, dtype=torch.float32, device=self.dev)
        out = [self._logit(_to_t(X[i:i + batch], self.dev), theta, self.w, self.b).cpu().numpy()
               for i in range(0, len(X), batch)]
        return np.concatenate(out)

    def predict_proba(self, X, batch=128):
        if not hasattr(self, "enc"):
            self._build()
        return 1.0 / (1.0 + np.exp(-self._predict_logits(X, batch)))

    def fit(self, Xtr, ytr, Xva, yva, epochs=25, batch=64, lr=0.05, patience=6, log=print):
        self._build()
        n = n_qubits_of(self.spec)
        g = torch.Generator().manual_seed(self.seed)
        theta = torch.nn.Parameter(0.1 * torch.randn(self.reps + 1, 3, n, generator=g).to(self.dev))
        w = torch.nn.Parameter(torch.tensor(3.0, device=self.dev))
        b = torch.nn.Parameter(torch.tensor(0.0, device=self.dev))
        opt = torch.optim.Adam([theta, w, b], lr=lr)
        Xtr_t, ytr_t = torch.as_tensor(np.asarray(Xtr), dtype=torch.float32), torch.as_tensor(np.asarray(ytr), dtype=torch.float32)
        best = dict(loss=float("inf"), theta=None, w=None, b=None)
        bad = 0
        for ep in range(1, epochs + 1):
            t0 = time.time()
            perm = torch.randperm(len(Xtr_t), generator=g)
            tot, cnt = 0.0, 0
            for i in range(0, len(perm), batch):
                idx = perm[i:i + batch]
                xb, yb = Xtr_t[idx].to(self.dev), ytr_t[idx].to(self.dev)
                loss = F_.binary_cross_entropy_with_logits(self._logit(xb, theta, w, b), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
                cnt += len(idx)
            self.theta, self.w, self.b = theta.detach().cpu().numpy(), float(w), float(b)
            logits = self._predict_logits(Xva)
            vloss = float(F_.binary_cross_entropy_with_logits(torch.as_tensor(logits), torch.as_tensor(np.asarray(yva), dtype=torch.float32)))
            vbal = balanced_accuracy_score(yva, (logits > 0).astype(int))
            self.history.append(dict(epoch=ep, train_loss=tot / cnt, val_loss=vloss, val_balanced_accuracy=float(vbal)))
            log(f"    epoch {ep:3d}/{epochs}  train loss {tot / cnt:.4f}  val loss {vloss:.4f}  "
                f"val bal.acc {vbal:.3f}  ({time.time() - t0:.0f}s)")
            if vloss < best["loss"] - 1e-4:
                best = dict(loss=vloss, theta=self.theta.copy(), w=self.w, b=self.b)
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    log("    early stopping")
                    break
        self.theta, self.w, self.b = best["theta"], best["w"], best["b"]
        return dict(best_val_loss=best["loss"], epochs_run=len(self.history))

    def save(self, path):
        joblib.dump(dict(spec=self.spec, reps=self.reps, entanglement=self.entanglement, readout=self.readout,
                         theta=self.theta, w=self.w, b=self.b, history=self.history), path)

    @classmethod
    def load(cls, path, device=None):
        d = joblib.load(path)
        m = cls(d["spec"], d["reps"], d["entanglement"], d["readout"], device=device)
        m.theta, m.w, m.b, m.history = d["theta"], d["w"], d["b"], d["history"]
        m._build()
        return m
