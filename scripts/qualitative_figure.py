"""Qualitative figure: zoomed vessel-rich crops, heavy noise, Euclidean / Poincare / Lorentz.

    python scripts/qualitative_figure.py --results results --seed 0 --out figures/qualitative

Image choice is NOT cherry-picked: by default the test images at the 50th and 75th percentile
of the per-image hyperbolic gain (mean of P and L minus E, heavy noise, PSNR_fov, seed-averaged)
are shown. The crop inside each image is the window with the most vessel structure
(gradient energy of the clean image inside the field of view). Pass --images to override.
Inference only (~3 models x 2 images), a few minutes on CPU, seconds on GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hypdenoise.data import get_splits, make_eval_dataset
from hypdenoise.models import build_model

MODELS = [("euclidean", "Euclidean"), ("poincare", "Poincaré"), ("lorentz", "Lorentz")]


def pick_images(results: Path, level: str, quantiles=(0.5, 0.75)) -> list[str]:
    frames = []
    for name, _ in MODELS:
        for csv in results.glob(f"{name}_seed*/test_per_image.csv"):
            df = pd.read_csv(csv)
            frames.append(df[df.level == level].assign(model=name))
    df = pd.concat(frames)
    col = "psnr_fov" if "psnr_fov" in df.columns and df["psnr_fov"].notna().all() else "psnr"
    df = df.groupby(["model", "image"])[col].mean().unstack(0)
    gain = (df["poincare"] + df["lorentz"]) / 2 - df["euclidean"]
    gain = gain.sort_values()
    return [gain.index[int(q * (len(gain) - 1))] for q in quantiles]


def best_crop(clean: np.ndarray, size: int, thr: float) -> tuple[int, int]:
    """Top-left corner of the size x size window with the highest gradient energy inside the FOV."""
    g = clean.mean(axis=2)
    gy, gx = np.gradient(g)
    e = (gx ** 2 + gy ** 2) * (g > thr)
    # integral image -> window sums
    ii = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s = ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]
    # ignore windows touching the FOV border (strong disc-edge gradient is not vessel structure)
    fov = (g > thr).astype(float)
    fi = np.pad(fov, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    inside = (fi[size:, size:] - fi[:-size, size:] - fi[size:, :-size] + fi[:-size, :-size]) >= size * size
    s = np.where(inside, s, -1)
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return int(y), int(x)


def psnr(a, b):
    return 10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", default="heavy")
    ap.add_argument("--images", nargs="*", default=None, help="test image file names")
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--err-scale", type=float, default=5.0, help="error maps: |x - clean| x this")
    ap.add_argument("--out", default="figures/qualitative")
    a = ap.parse_args()
    results, device = Path(a.results), torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for name, _ in MODELS:
        ckpt = torch.load(results / f"{name}_seed{a.seed}" / "best.pt", map_location=device, weights_only=False)
        m = build_model(ckpt["config"]).to(device)
        m.load_state_dict(ckpt["model_state"])
        models[name] = m.eval()
        cfg = ckpt["config"]

    test_paths = get_splits(cfg)["test"]
    names = a.images or pick_images(results, a.level)
    print("images:", names)
    ds = make_eval_dataset(cfg, test_paths, a.level)  # exact same noisy input as the test protocol
    by_name = {p.name: i for i, p in enumerate(test_paths)}
    thr = cfg["eval"].get("fov_threshold", 10 / 255)

    cols = ["Noisy"] + [t for _, t in MODELS] + ["Clean"]
    fig, axes = plt.subplots(2 * len(names), len(cols), figsize=(1.45 * len(cols), 1.5 * 2 * len(names)))
    for r, img_name in enumerate(names):
        noisy, clean, _, _ = ds[by_name[img_name]]
        outs = {k: m(noisy[None].to(device)).clamp(0, 1)[0].cpu() for k, m in models.items()}
        to_np = lambda t: t.permute(1, 2, 0).numpy()
        c = to_np(clean)
        y, x = best_crop(c, a.crop, thr)
        sl = (slice(y, y + a.crop), slice(x, x + a.crop))
        panels = [to_np(noisy)] + [to_np(outs[k]) for k, _ in MODELS] + [c]
        for j, (p, title) in enumerate(zip(panels, cols)):
            ax_img, ax_err = axes[2 * r, j], axes[2 * r + 1, j]
            ax_img.imshow(p[sl])
            if r == 0:
                ax_img.set_title(title, fontsize=8)
            if title != "Clean":
                ax_img.text(2, a.crop - 4, f"{psnr(p[sl], c[sl]):.2f} dB", color="white", fontsize=6,
                            bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
                ax_err.imshow(np.clip(np.abs(p[sl] - c[sl]).mean(2) * a.err_scale, 0, 1), cmap="gray", vmin=0, vmax=1)
            else:
                ax_err.axis("off")
            for ax in (ax_img, ax_err):
                ax.set_xticks([]); ax.set_yticks([])
        axes[2 * r, 0].set_ylabel(img_name.split(".")[0], fontsize=7)
        axes[2 * r + 1, 0].set_ylabel(f"|error| ×{a.err_scale:g}", fontsize=7)
    plt.tight_layout(pad=0.2)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print("saved", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
