"""Reading / writing the per-model result files that the dashboard displays."""
import json

import numpy as np

import config as C


def _json_path(module, mid):
    return C.RESULTS / module / f"{mid}.json"


def exists(module, mid):
    return _json_path(module, mid).exists()


def save_result(module, rec, probs, y, idx):
    """rec: metadata + metrics dict; probs/y/idx: test-set probabilities, labels and dataset indices."""
    rec = dict(rec, id=rec["id"], module=module)
    _json_path(module, rec["id"]).write_text(json.dumps(rec, indent=1, default=float))
    np.savez_compressed(C.RESULTS / module / f"{rec['id']}.npz", probs=np.asarray(probs, dtype=np.float32),
                        y=np.asarray(y), idx=np.asarray(idx))


def load_results(module):
    out = []
    for p in sorted((C.RESULTS / module).glob("*.json")):
        out.append(json.loads(p.read_text()))
    return out


def load_probs(module, mid):
    d = np.load(C.RESULTS / module / f"{mid}.npz")
    return d["probs"], d["y"], d["idx"]
