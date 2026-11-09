"""STEP 1 - read every chest X-ray once and cache what all models need.

  * 128x128 grayscale copies of all images           -> artifacts/cache/xray_cache.npz
  * 1,024 frozen DenseNet-121 features per image     -> artifacts/cache/xray_dense.npz   (runs on the GPU)
  * the train / validation / test split codes        (official test folder = test; stratified 80/20 train/val)

    python scripts/prepare_xray.py            # full run
    python scripts/prepare_xray.py --quick    # ~360 images, smoke test (writes to artifacts_quick/)
"""
import os
import sys

if "--quick" in sys.argv:
    os.environ["QML_QUICK"] = "1"
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config as C
from common import data_xray as DX
from common import features_cnn as FC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    rows = DX.index_images()
    print(f"found {len(rows)} images under {C.XRAY_DIR}")
    if args.quick:                                   # first 60 images of every (split, class)
        keep, seen = [], {}
        for r in rows:
            k = (r[2], r[1])
            if seen.get(k, 0) < 60:
                keep.append(r)
                seen[k] = seen.get(k, 0) + 1
        rows = keep
        print(f"quick mode: using {len(rows)} images")

    codes = DX.make_split_codes(rows)
    labels = np.array([r[1] for r in rows], dtype=np.int64)
    paths = np.array([r[0] for r in rows])

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        grays = list(ex.map(lambda p: DX.load_gray(C.DATA / p), paths))
    print(f"loaded {len(grays)} images in {time.time() - t0:.0f}s")
    a128 = np.stack([DX.to_square(g, C.XRAY_STORE_SIZE) for g in grays])
    a224 = np.stack([DX.to_square(g, C.DENSENET_SIZE) for g in grays])
    del grays

    device = FC.get_device()
    print(f"DenseNet-121 feature extraction on {device} ...")
    t0 = time.time()
    feats = FC.dense_features(FC.load_densenet(device), a224, device)
    print(f"features {feats.shape} in {time.time() - t0:.0f}s")

    np.savez_compressed(C.CACHE / "xray_cache.npz", pix128=a128, y=labels, split=codes, paths=paths)
    np.savez_compressed(C.CACHE / "xray_dense.npz", feat=feats)
    for code, name in enumerate(["train", "val", "test"]):
        m = codes == code
        print(f"{name:5s}: {m.sum():5d} images  (NORMAL {(labels[m] == 0).sum()}, PNEUMONIA {(labels[m] == 1).sum()})")
    print("saved to", C.CACHE)


if __name__ == "__main__":
    main()
