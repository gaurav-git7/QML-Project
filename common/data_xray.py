"""Chest X-ray data: indexing, splits, and the image -> numbers representations used by every model.

Pipeline (same one the PDF 'How to Encode an Image into Quantum States' describes):
    image -> grayscale -> resize -> normalise -> (quantum encoding | classical model)

Representations (name -> what the model sees)
    PIX<S>   S x S grayscale pixels (contrast-stretched to [0,1]), flattened      (S = 4, 16, 64, 128)
    DENSE    1,024 features from a frozen ImageNet DenseNet-121
    PCA<n>   the DENSE features reduced to n numbers in [0,1] (PCA + percentile scaling)
"""
import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as C

CLASSES = ["NORMAL", "PNEUMONIA"]            # label 0 / 1
_IMG_EXT = {".jpeg", ".jpg", ".png"}


# ---------------------------------------------------------------- indexing and splits
def index_images():
    """[(path relative to DATA, label, official split name)] for every image on disk."""
    rows = []
    for split in ("train", "val", "test"):
        for label, cls in enumerate(CLASSES):
            folder = C.XRAY_DIR / split / cls
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in _IMG_EXT and not p.name.startswith("._"):
                    rows.append((p.relative_to(C.DATA).as_posix(), label, split))
    return rows


def make_split_codes(rows, val_fraction=0.2, seed=C.SEED):
    """0 = train, 1 = validation, 2 = test.
    The official test folder stays the test set. The official val folder has only 16 images, so it is merged
    with the official train folder and a stratified validation split is carved out of that pool."""
    labels = np.array([r[1] for r in rows])
    official = np.array([r[2] for r in rows])
    codes = np.full(len(rows), 2, dtype=np.int8)
    pool = np.flatnonzero(official != "test")
    tr, va = train_test_split(pool, test_size=val_fraction, stratify=labels[pool], random_state=seed)
    codes[tr], codes[va] = 0, 1
    return codes


# ---------------------------------------------------------------- image helpers (shared with the dashboard)
def to_gray_uint8(img):
    """PIL image or ndarray -> 2-D uint8 grayscale array."""
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return arr.astype(np.uint8)
    return np.array(img.convert("L"), dtype=np.uint8)


def to_square(arr, size):
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def load_gray(path):
    with Image.open(path) as im:
        return to_gray_uint8(im)


def _stretch(a):
    """Per-image contrast stretch (1st-99th percentile -> [0,1]); X-ray exposure varies a lot between studies."""
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def pixel_rep(arr128, size):
    """(N,128,128) uint8 -> (N, size*size) float32 in [0,1]."""
    out = np.empty((len(arr128), size * size), dtype=np.float32)
    for i, a in enumerate(arr128):
        s = _stretch(a)
        if size != s.shape[0]:
            s = cv2.resize(s, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = s.reshape(-1)
    return out


# ---------------------------------------------------------------- DenseNet features -> n numbers in [0,1]
class FeatureProjector:
    """DenseNet features (1024) -> n numbers in [0,1]: standardise, PCA, then 2nd-98th percentile scaling.
    Fitted on the training images only."""

    def __init__(self, n):
        self.n = n

    def fit(self, feats):
        self.scaler = StandardScaler().fit(feats)
        self.pca = PCA(self.n, random_state=C.SEED).fit(self.scaler.transform(feats))
        z = self.pca.transform(self.scaler.transform(feats))
        self.lo, self.hi = np.percentile(z, 2, axis=0), np.percentile(z, 98, axis=0)
        return self

    def transform(self, feats):
        z = self.pca.transform(self.scaler.transform(feats))
        return np.clip((z - self.lo) / np.maximum(self.hi - self.lo, 1e-6), 0, 1).astype(np.float32)
