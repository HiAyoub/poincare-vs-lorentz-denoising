"""Train one model with one seed.

    python -m hypdenoise.train --config configs/lorentz.yaml --seed 0

Writes to <output_dir>/<name>_seed<seed>/:
    config.json, history.json, best.pt (highest val PSNR), last.pt
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
from hypll.optim import RiemannianAdam
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .data import find_data_root, make_train_val_loaders, official_splits
from .models import build_model, count_params
from .utils import load_config, run_dir, save_json, seed_everything


def build_optimizer(model, cfg):
    """Same optimizer implementation for every model (hypll RiemannianAdam).

    The curvature gets its own weight decay (default 0): weight decay on the raw
    curvature parameter pulls c towards softplus(0) = ln 2, independently of the task.
    """
    t = cfg["train"]
    curv = []
    if model.manifold is not None and model.manifold.c.value.requires_grad:
        curv = [model.manifold.c.value]
    curv_ids = {id(p) for p in curv}
    others = [p for p in model.parameters() if id(p) not in curv_ids]
    groups = [{"params": others, "weight_decay": t["weight_decay"]}]
    if curv:
        groups.append({"params": curv, "weight_decay": t["curvature_weight_decay"]})
    return RiemannianAdam(groups, lr=t["lr"], weight_decay=t["weight_decay"])


def build_scheduler(optimizer, cfg, steps_per_epoch):
    t = cfg["train"]
    total, warm = t["epochs"] * steps_per_epoch, t["warmup_epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _raw(p):
    return p.tensor if hasattr(p, "tensor") else p


def train(cfg: dict, seed: int) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = run_dir(cfg, seed)
    save_json({**cfg, "seed": seed}, out / "config.json")

    splits = official_splits(find_data_root(cfg["data"].get("root")))
    train_loader, val_loader = make_train_val_loaders(cfg, splits, seed)

    model = build_model(cfg).to(device)
    print(f"[{cfg['name']} seed={seed}] encoder={cfg['model']['encoder']} "
          f"params={count_params(model)/1e6:.2f}M device={device}")

    criterion = nn.L1Loss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    psnr, ssim = PeakSignalNoiseRatio(data_range=1.0).to(device), StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    hist = {k: [] for k in ["epoch", "train_l1", "val_l1", "val_psnr", "val_ssim", "lr", "curvature", "minutes"]}
    best, t0 = -1.0, time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        run, n = 0.0, 0
        for noisy, clean, _, _ in train_loader:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(noisy), clean)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([_raw(p) for p in model.parameters()], cfg["train"]["grad_clip"])
            optimizer.step()
            scheduler.step()
            run += loss.item() * noisy.size(0)
            n += noisy.size(0)

        model.eval()
        psnr.reset(); ssim.reset()
        vrun, vn = 0.0, 0
        with torch.no_grad():
            for noisy, clean, _, _ in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                den = model(noisy).clamp(0, 1)
                vrun += criterion(den, clean).item() * noisy.size(0)
                vn += noisy.size(0)
                psnr.update(den, clean); ssim.update(den, clean)
        vpsnr, vssim = psnr.compute().item(), ssim.compute().item()

        for k, v in zip(hist, [epoch, run / n, vrun / vn, vpsnr, vssim, scheduler.get_last_lr()[0],
                               model.curvature(), (time.time() - t0) / 60]):
            hist[k].append(v)
        c_str = f" c={model.curvature():.4f}" if model.curvature() is not None else ""
        print(f"epoch {epoch:02d}/{cfg['train']['epochs']} train_L1={run/n:.4f} val_PSNR={vpsnr:.2f} "
              f"val_SSIM={vssim:.4f}{c_str} ({hist['minutes'][-1]:.1f} min)", flush=True)

        state = {"epoch": epoch, "model_state": model.state_dict(), "val_psnr": vpsnr,
                 "curvature": model.curvature(), "config": cfg, "seed": seed}
        if vpsnr > best:
            best = vpsnr
            torch.save(state, out / "best.pt")
        save_json(hist, out / "history.json")
    torch.save(state, out / "last.pt")
    summary = {"name": cfg["name"], "seed": seed, "best_val_psnr": best,
               "final_curvature": model.curvature(), "minutes": hist["minutes"][-1],
               "params": count_params(model)}
    save_json(summary, out / "train_summary.json")
    print("done:", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=2")
    a = ap.parse_args()
    train(load_config(a.config, a.set), a.seed)


if __name__ == "__main__":
    main()
