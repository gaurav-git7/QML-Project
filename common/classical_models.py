"""Classical baselines: logistic regression, RBF-SVM, small neural network (and gradient boosting for the heart data).
Each is tuned on the same balanced validation set that the quantum models use, with a small hyper-parameter grid."""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

NAMES = {"lr": "Logistic regression", "svm": "SVM (RBF)", "mlp": "Neural network (MLP)", "hgb": "Gradient boosting"}


def _grid(kind, n_features):
    if kind == "lr":
        return [dict(C=c) for c in (0.001, 0.01, 0.1, 1.0)]
    if kind == "svm":
        return [dict(C=c, gamma=g / n_features) for c in (1.0, 10.0) for g in (0.25, 1.0, 4.0)]
    if kind == "mlp":
        return [dict(alpha=a) for a in (1e-4, 1e-2)]
    return [dict(learning_rate=0.1, max_iter=200)]


def _make(kind, hp, probability=False):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], max_iter=3000, class_weight="balanced"))
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=hp["C"], gamma=hp["gamma"], class_weight="balanced",
                                                   probability=probability, random_state=0))
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier((128, 64), alpha=hp["alpha"], early_stopping=True,
                                                             max_iter=300, random_state=0))
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=0, **hp)


def oversample_balance(X, y, rng):
    """Duplicate minority-class rows until both classes are equally frequent (MLP has no class weights)."""
    idx = [np.flatnonzero(y == c) for c in (0, 1)]
    m = max(len(i) for i in idx)
    sel = np.concatenate([np.concatenate([i, rng.choice(i, m - len(i), replace=True)]) for i in idx])
    return X[sel], y[sel]


def _score(est, X, y):
    if hasattr(est, "decision_function") and not hasattr(est, "predict_proba"):
        pred = (est.decision_function(X) > 0).astype(int)
    else:
        pred = est.predict(X)
    return balanced_accuracy_score(y, pred)


def fit_classical(kind, Xtr, ytr, Xva, yva, seed=0):
    """Grid search on the validation set. Returns (fitted estimator with predict_proba, best hyper-parameters,
    validation balanced accuracy, seconds)."""
    rng = np.random.default_rng(seed)
    if kind == "mlp":
        Xtr, ytr = oversample_balance(Xtr, ytr, rng)
    t0 = time.time()
    best = None
    for hp in _grid(kind, Xtr.shape[1]):
        est = _make(kind, hp).fit(Xtr, ytr)
        bal = _score(est, Xva, yva)
        if best is None or bal > best[0]:
            best = (bal, hp)
    est = _make(kind, best[1], probability=True).fit(Xtr, ytr)       # refit the winner with probability outputs
    return est, best[1], float(best[0]), time.time() - t0
