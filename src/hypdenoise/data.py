"""RFMiD dataset, Poisson-Gaussian noise model and the fixed evaluation protocol.

The noise model and the training pipeline are identical to the original internship
notebooks (Phases 2.1-5). What is new is that *all* models are evaluated through
``make_eval_dataset`` so every reported number uses exactly the same noisy inputs.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EXTS = (".png", ".jpg", ".jpeg")

# (alpha, sigma) in 8-bit intensity units. Var(y) = alpha * mu + sigma^2.
NOISE_LEVELS = {
    "light": {"alpha": 0.5, "sigma": 3.0},
    "medium": {"alpha": 1.0, "sigma": 5.0},
    "heavy": {"alpha": 2.0, "sigma": 8.0},
}

DATA_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/andrewmvd/retinal-disease-classification"),
    Path("/kaggle/input/retinal-disease-classification"),
    Path("./data/retinal-disease-classification"),
]


def find_data_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    root = next((p for p in DATA_ROOT_CANDIDATES if p.exists()), None)
    if root is None:
        raise FileNotFoundError(f"RFMiD not found in {DATA_ROOT_CANDIDATES}")
    return root


def find_split_images(root: Path, set_dir: str, inner_dir: str) -> list[Path]:
    for c in [root / set_dir / set_dir / inner_dir, root / set_dir / inner_dir, root / inner_dir]:
        if c.is_dir():
            imgs = sorted(p for p in c.iterdir() if p.suffix.lower() in EXTS)
            if imgs:
                return imgs
    raise FileNotFoundError(f"No images for {set_dir}/{inner_dir} under {root}")


def official_splits(root: Path) -> dict[str, list[Path]]:
    """Official RFMiD splits: 1920 / 640 / 640 images."""
    return {
        "train": find_split_images(root, "Training_Set", "Training"),
        "val": find_split_images(root, "Evaluation_Set", "Validation"),
        "test": find_split_images(root, "Test_Set", "Test"),
    }


def add_poisson_gaussian_noise(clean_uint8, alpha, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = clean_uint8.astype(np.float64)
    shot = rng.standard_normal(x.shape) * np.sqrt(np.clip(alpha * x, 0.0, None))
    read = rng.standard_normal(x.shape) * sigma
    return np.clip(x + shot + read, 0, 255).astype(np.uint8)


class RFMiDDataset(Dataset):
    """Returns (noisy, clean, noise_level_name, index).

    crop:
        "random"  random patch (seeded if deterministic_seed is set) - training / validation
        "center"  central patch
        "none"    full resized image (image_size x image_size) - test protocol
    """

    def __init__(
        self,
        paths,
        image_size: int = 384,
        patch_size: int = 256,
        crop: str = "random",
        augment: bool = True,
        fixed_noise: Optional[str] = None,
        deterministic_seed: Optional[int] = None,
    ):
        assert crop in ("random", "center", "none")
        self.paths = list(paths)
        self.image_size, self.patch_size = image_size, patch_size
        self.crop, self.augment = crop, augment
        self.fixed_noise, self.deterministic_seed = fixed_noise, deterministic_seed
        self._level_names = list(NOISE_LEVELS)

    def __len__(self):
        return len(self.paths)

    def _load(self, p):
        with Image.open(p) as im:
            im = im.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
            return np.array(im, dtype=np.uint8)

    def _crop(self, img, rng):
        H, W = img.shape[:2]
        if self.crop == "none" or self.patch_size <= 0 or H <= self.patch_size:
            return img
        if self.crop == "center":
            top, left = (H - self.patch_size) // 2, (W - self.patch_size) // 2
        else:
            top = int(rng.integers(0, H - self.patch_size + 1))
            left = int(rng.integers(0, W - self.patch_size + 1))
        return img[top : top + self.patch_size, left : left + self.patch_size]

    def _augment(self, img, rng):
        if rng.random() < 0.5:
            img = np.fliplr(img).copy()
        if rng.random() < 0.5:
            img = np.flipud(img).copy()
        k = int(rng.integers(0, 4))
        return np.rot90(img, k=k).copy() if k else img

    def __getitem__(self, idx):
        if self.deterministic_seed is not None:
            item_seed = self.deterministic_seed * 10_000 + idx
        else:
            item_seed = random.randint(0, 2**31 - 1)
        rng = np.random.default_rng(item_seed)
        clean = self._crop(self._load(self.paths[idx]), rng)
        if self.augment:
            clean = self._augment(clean, rng)
        if self.fixed_noise is not None:
            level = self.fixed_noise
        else:
            level = self._level_names[np.random.default_rng(item_seed).integers(0, 3)]
        p = NOISE_LEVELS[level]
        noisy = add_poisson_gaussian_noise(
            clean, p["alpha"], p["sigma"], rng=np.random.default_rng(item_seed + 1)
        )
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
        return to_t(noisy), to_t(clean), level, idx


def make_train_val_loaders(cfg: dict, splits: dict, seed: int):
    d = cfg["data"]
    train_ds = RFMiDDataset(splits["train"], d["image_size"], d["patch_size"], crop="random", augment=True)
    # Validation (checkpoint selection): seeded random crops + mixed noise levels, same for all runs.
    val_ds = RFMiDDataset(
        splits["val"], d["image_size"], d["patch_size"], crop="random", augment=False,
        deterministic_seed=d["val_seed"],
    )
    g = torch.Generator()
    g.manual_seed(seed)
    bs, nw = cfg["train"]["batch_size"], d["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
        drop_last=True, persistent_workers=nw > 0, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
    )
    return train_loader, val_loader


def make_eval_dataset(cfg: dict, paths, level: str) -> RFMiDDataset:
    """THE test protocol. Every model in the paper is evaluated through this function."""
    e = cfg["eval"]
    return RFMiDDataset(
        paths, cfg["data"]["image_size"], cfg["data"]["patch_size"], crop=e["crop"],
        augment=False, fixed_noise=level, deterministic_seed=e["noise_seed"],
    )
