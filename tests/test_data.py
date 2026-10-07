"""FIVES split logic and vessel masks on a tiny fake FIVES tree (no download needed)."""

import numpy as np
from PIL import Image

from hypdenoise.data import RFMiDDataset, fives_splits, get_splits, load_vessel_mask


def _fake_fives(root, n_train_per_class=20, n_test_per_class=5):
    k = 0
    for split, n in (("train", n_train_per_class), ("test", n_test_per_class)):
        (root / split / "Original").mkdir(parents=True)
        (root / split / "Ground truth").mkdir(parents=True)
        for cls in "ADGN":
            for _ in range(n):
                k += 1
                img = np.random.default_rng(k).integers(0, 255, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(img).save(root / split / "Original" / f"{k}_{cls}.png")
                m = np.zeros((64, 64), np.uint8)
                m[30:34, :] = 255  # one horizontal "vessel"
                Image.fromarray(m).save(root / split / "Ground truth" / f"{k}_{cls}.png")
    return root


def test_fives_splits_are_disjoint_and_balanced(tmp_path):
    s = fives_splits(_fake_fives(tmp_path), n_val_per_class=5)
    assert len(s["val"]) == 20 and len(s["train"]) == 60 and len(s["test"]) == 20
    assert not set(s["train"]) & set(s["val"])
    assert sorted(p.stem[-1] for p in s["val"]).count("A") == 5
    assert all(p in s["masks"] for p in s["test"])


def test_get_splits_dispatch_and_mask(tmp_path):
    _fake_fives(tmp_path)
    s = get_splits({"data": {"dataset": "fives", "root": str(tmp_path)}})
    m = load_vessel_mask(s["masks"][s["test"][0]], 32)
    assert m.shape == (1, 1, 32, 32)
    assert 0 < m.mean() < 0.3


def test_cache_and_repeats(tmp_path):
    s = fives_splits(_fake_fives(tmp_path, 2, 1), n_val_per_class=1)
    ds = RFMiDDataset(s["train"], 64, 32, repeats=3, cache=True, deterministic_seed=0)
    assert len(ds) == 3 * len(s["train"])
    a, b = ds[0][1], RFMiDDataset(s["train"], 64, 32, deterministic_seed=0)[0][1]
    assert (a == b).all()
