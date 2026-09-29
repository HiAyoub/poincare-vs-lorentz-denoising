"""Evaluate a trained checkpoint with THE test protocol and save per-image metrics.

    python -m hypdenoise.evaluate --run results/lorentz_seed0

Protocol (identical for every model, see configs/base.yaml -> eval):
  * official RFMiD test split (640 images), resized to 384x384, full image (no crop)
  * each noise level (light / medium / heavy) evaluated separately
  * noise realisation fixed per image (noise_seed) -> every model sees the same noisy inputs
  * per-image PSNR, SSIM and LPIPS (AlexNet) saved to test_per_image.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import lpips
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from .data import NOISE_LEVELS, find_data_root, make_eval_dataset, official_splits
from .models import build_model
from .utils import save_json


@torch.no_grad()
def evaluate_run(run: Path, which: str = "best.pt", device=None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run / which, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    lp = lpips.LPIPS(net="alex", verbose=False).to(device)

    test_paths = official_splits(find_data_root(cfg["data"].get("root")))["test"]
    rows = []
    for level in NOISE_LEVELS:
        ds = make_eval_dataset(cfg, test_paths, level)
        loader = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                            num_workers=cfg["data"]["num_workers"])
        for noisy, clean, _, idx in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            den = model(noisy).clamp(0, 1)
            for j in range(noisy.size(0)):
                d, c, n = den[j : j + 1], clean[j : j + 1], noisy[j : j + 1]
                rows.append({
                    "image": test_paths[int(idx[j])].name, "level": level,
                    "psnr": peak_signal_noise_ratio(d, c, data_range=1.0).item(),
                    "ssim": structural_similarity_index_measure(d, c, data_range=1.0).item(),
                    "lpips": lp(d * 2 - 1, c * 2 - 1).item(),
                    "psnr_noisy": peak_signal_noise_ratio(n, c, data_range=1.0).item(),
                })

    with open(run / "test_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"run": run.name, "checkpoint": which, "epoch": ckpt["epoch"], "curvature": ckpt.get("curvature")}
    for level in NOISE_LEVELS:
        r = [x for x in rows if x["level"] == level]
        summary[level] = {m: sum(x[m] for x in r) / len(r) for m in ("psnr", "ssim", "lpips", "psnr_noisy")}
    save_json(summary, run / "test_summary.json")
    print(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory, e.g. results/lorentz_seed0")
    ap.add_argument("--checkpoint", default="best.pt")
    a = ap.parse_args()
    evaluate_run(Path(a.run), a.checkpoint)


if __name__ == "__main__":
    main()
