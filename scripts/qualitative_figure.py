"""Qualitative figure: zoomed vessel-rich crops, heavy noise, Euclidean / Poincare / Lorentz.

    # RFMiD test set
    python scripts/qualitative_figure.py --results results --seed 1 --out figures/qualitative_rfmid
    # FIVES test set with the RFMiD-trained checkpoints (cross-dataset), vessel annotation overlaid
    python scripts/qualitative_figure.py --results results --seed 1 --data-config configs/eval_fives.yaml \
        --tag fives --metric psnr_vessel --out figures/qualitative_fives

Image choice is NOT cherry-picked: by default the test images at the 50th and 75th percentile
of the per-image hyperbolic gain (mean of P and L minus E, seed-averaged, on --metric) are shown,
read from test_per_image[_<tag>].csv. The crop is the window with the most vessel pixels (FIVES
annotation) or, without annotation, the most gradient energy inside the field of view.
Inference only, seconds on GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hypdenoise.data import get_splits, load_vessel_mask, make_eval_dataset
from hypdenoise.models import build_model
from hypdenoise.utils import _deep_update, load_config

MODELS = [("euclidean", "Euclidean"), ("poincare", "Poincaré"), ("lorentz", "Lorentz")]


def pick_images(results: Path, level: str, metric: str, tag: str | None, quantiles=(0.5, 0.75)) -> list[str]:
    fname = f"test_per_image_{tag}.csv" if tag else "test_per_image.csv"
    frames = []
    for name, _ in MODELS:
        for csv in results.glob(f"{name}_seed*/{fname}"):
            df = pd.read_csv(csv)
            frames.append(df[df.level == level].assign(model=name))
    df = pd.concat(frames).groupby(["model", "image"])[metric].mean().unstack(0)
    gain = ((df["poincare"] + df["lorentz"]) / 2 - df["euclidean"]).sort_values()
    return [gain.index[int(q * (len(gain) - 1))] for q in quantiles]


def _window_sums(a: np.ndarray, size: int) -> np.ndarray:
    ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]


def best_crop(clean: np.ndarray, size: int, thr: float, vessels: np.ndarray | None = None) -> tuple[int, int]:
    """Top-left corner of the size x size window, fully inside the field of view, with the most
    vessel pixels (if an annotation is given) or the most gradient energy."""
    g = clean.mean(axis=2)
    fov = (g > thr).astype(float)
    if vessels is not None:
        e = vessels.astype(float)
    else:
        gy, gx = np.gradient(g)
        e = (gx ** 2 + gy ** 2) * fov
    s = _window_sums(e, size)
    inside = _window_sums(fov, size) >= size * size
    s = np.where(inside, s, -1)
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return int(y), int(x)


def psnr(a, b, mask=None):
    se = (a - b) ** 2
    mse = se.mean() if mask is None else se[mask > 0].mean()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", default="heavy")
    ap.add_argument("--images", nargs="*", default=None, help="test image file names")
    ap.add_argument("--data-config", default=None, help="test on another dataset, e.g. configs/eval_fives.yaml")
    ap.add_argument("--tag", default=None, help="read test_per_image_<tag>.csv to pick images")
    ap.add_argument("--metric", default="psnr_fov", help="per-image metric used to pick images")
    ap.add_argument("--crop", type=int, default=None, help="crop size (default: image_size / 4)")
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
    if a.data_config:
        over = load_config(a.data_config)
        cfg = _deep_update(cfg, {k: over[k] for k in ("data", "eval") if k in over})

    splits = get_splits(cfg)
    test_paths, masks = splits["test"], splits.get("masks", {})
    size = cfg["data"]["image_size"]
    crop = a.crop or size // 4
    names = a.images or pick_images(results, a.level, a.metric, a.tag)
    print("images:", names)
    ds = make_eval_dataset(cfg, test_paths, a.level)  # exact same noisy input as the test protocol
    by_name = {p.name: i for i, p in enumerate(test_paths)}
    thr = cfg["eval"].get("fov_threshold", 10 / 255)

    cols = ["Noisy"] + [t for _, t in MODELS] + ["Clean"]
    fig, axes = plt.subplots(2 * len(names), len(cols), figsize=(1.45 * len(cols), 1.5 * 2 * len(names)))
    to_np = lambda t: t.permute(1, 2, 0).numpy()
    for r, img_name in enumerate(names):
        i = by_name[img_name]
        noisy, clean, _, _ = ds[i]
        outs = {k: m(noisy[None].to(device)).clamp(0, 1)[0].cpu() for k, m in models.items()}
        c = to_np(clean)
        v = load_vessel_mask(masks[test_paths[i]], size)[0, 0].numpy() if masks else None
        y, x = best_crop(c, crop, thr, v)
        sl = (slice(y, y + crop), slice(x, x + crop))
        vc = v[sl] if v is not None else None
        panels = [to_np(noisy)] + [to_np(outs[k]) for k, _ in MODELS] + [c]
        for j, (p, title) in enumerate(zip(panels, cols)):
            ax_img, ax_err = axes[2 * r, j], axes[2 * r + 1, j]
            ax_img.imshow(p[sl], interpolation="nearest")
            if r == 0:
                ax_img.set_title(title, fontsize=8)
            if title != "Clean":
                label = f"{psnr(p[sl], c[sl]):.2f} dB"
                if vc is not None and vc.sum() > 0:
                    label += f"\nvessels {psnr(p[sl], c[sl], vc):.2f}"
                ax_img.text(2, crop - 4, label, color="white", fontsize=5.5, va="bottom",
                            bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
                ax_err.imshow(np.clip(np.abs(p[sl] - c[sl]).mean(2) * a.err_scale, 0, 1), cmap="gray",
                              vmin=0, vmax=1, interpolation="nearest")
            elif vc is not None:
                ax_err.imshow(c[sl], interpolation="nearest")
                ax_err.contour(vc, levels=[0.5], colors=["#00e5ff"], linewidths=0.4)
                if r == 0:
                    ax_err.set_title("Vessel annotation", fontsize=6)
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
