"""Shared helper that evaluates one trained model and writes its result record."""
from pathlib import Path

import config as C
from . import metrics as M
from . import results as R


def rel(path):
    return Path(path).relative_to(C.ARTIFACTS).as_posix()


def finish(module, mid, meta, p_va, y_va, p_te, y_te, te_idx):
    """meta: descriptive fields (family, method, representation, ...). p_*: P(class 1); y_*: labels."""
    rec = dict(id=mid, **meta, val=M.binary_metrics(y_va, p_va), test=M.binary_metrics(y_te, p_te),
               roc=M.roc_points(y_te, p_te))
    R.save_result(module, rec, p_te, y_te, te_idx)
    t = rec["test"]
    print(f"  -> {mid}: test balanced acc {t['balanced_accuracy']:.3f} | sensitivity {t['sensitivity']:.3f} | "
          f"specificity {t['specificity']:.3f} | AUC {t['auc']:.3f}")
    return rec
