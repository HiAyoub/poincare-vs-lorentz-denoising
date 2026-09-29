from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def _deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = _deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Loads a YAML config. A config may name a parent with `inherit: base.yaml`.
    Overrides use dotted keys, e.g. ["train.epochs=2", "model.encoder=lorentz"]."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "inherit" in cfg:
        parent = load_config(str(path.parent / cfg.pop("inherit")))
        cfg = _deep_update(parent, cfg)
    for o in overrides or []:
        key, val = o.split("=", 1)
        d = cfg
        *parents, last = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_dir(cfg: dict, seed: int) -> Path:
    d = Path(cfg["output_dir"]) / f"{cfg['name']}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
