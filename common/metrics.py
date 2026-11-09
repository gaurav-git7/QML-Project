"""Binary-classification metrics shared by both modules (class 1 = disease present)."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def binary_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return dict(accuracy=float((tp + tn) / max(len(y), 1)), balanced_accuracy=float((sens + spec) / 2),
                sensitivity=float(sens), specificity=float(spec), precision=float(prec), f1=float(f1),
                auc=auc, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(len(y)))


def roc_points(y, p, n=101):
    """ROC curve resampled on a fixed FPR grid so it is small enough to store in JSON."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return dict(fpr=[0.0, 1.0], tpr=[0.0, 1.0])
    fpr, tpr, _ = roc_curve(y, p)
    grid = np.linspace(0, 1, n)
    return dict(fpr=grid.tolist(), tpr=np.interp(grid, fpr, tpr).tolist())


def balanced_indices(y, pool, n_per_class, rng):
    """Class-balanced random subset of `pool` (indices into y) with at most n_per_class per class."""
    pool = np.asarray(pool)
    per = [pool[y[pool] == c] for c in (0, 1)]
    m = int(min(n_per_class, min(len(a) for a in per)))
    chosen = np.concatenate([rng.choice(a, size=m, replace=False) for a in per])
    return rng.permutation(chosen)
