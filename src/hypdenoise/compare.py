"""Compare evaluated runs: mean +/- 95% bootstrap CI and paired Wilcoxon tests.

    python -m hypdenoise.compare --results results --reference euclidean

Per-image metrics are first averaged over seeds, then models are compared image by image
(paired), separately for each noise level. Output: results/comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_all(results: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(results.glob("*_seed*/test_per_image.csv")):
        name, seed = csv_path.parent.name.rsplit("_seed", 1)
        df = pd.read_csv(csv_path)
        df["model"], df["seed"] = name, int(seed)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No test_per_image.csv under {results}")
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(results: Path, reference: str) -> pd.DataFrame:
    df = load_all(results)
    metrics = [m for m in ["psnr", "ssim", "lpips", "psnr_fov", "ssim_fov", "lpips_fov"] if m in df.columns]
    per_img = df.groupby(["model", "level", "image"])[metrics].mean().reset_index()
    n_seeds = df.groupby("model")["seed"].nunique().to_dict()
    rows = []
    for level in ["light", "medium", "heavy"]:
        lv = per_img[per_img.level == level]
        ref = lv[lv.model == reference].set_index("image")
        for model in sorted(lv.model.unique()):
            cur = lv[lv.model == model].set_index("image")
            common = cur.index.intersection(ref.index)
            row = {"level": level, "model": model, "n_seeds": n_seeds[model]}
            for m in metrics:
                x = cur.loc[common, m].to_numpy()
                row[m] = x.mean()
                row[f"{m}_ci_low"], row[f"{m}_ci_high"] = bootstrap_ci(x)
                if model != reference:
                    diff = x - ref.loc[common, m].to_numpy()
                    row[f"d_{m}_vs_{reference}"] = diff.mean()
                    lo, hi = bootstrap_ci(diff)
                    row[f"d_{m}_ci_low"], row[f"d_{m}_ci_high"] = lo, hi
                    row[f"p_{m}_wilcoxon"] = float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(results / "comparison.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--reference", default="euclidean")
    a = ap.parse_args()
    out = compare(Path(a.results), a.reference)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 4):
        print(out)


if __name__ == "__main__":
    main()
