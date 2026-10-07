"""Final RFMiD analysis: tables, paired tests (L vs P, P vs E, L vs E), figures."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

RES = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
LEVELS = ["light", "medium", "heavy"]; MODELS = ["euclidean", "poincare", "lorentz"]
METRICS = ["psnr_fov", "ssim_fov", "lpips_fov", "psnr", "ssim", "lpips"]

frames = []
for f in sorted(RES.glob("*_seed*/test_per_image.csv")):
    name, seed = f.parent.name.rsplit("_seed", 1)
    d = pd.read_csv(f); d["model"], d["seed"] = name, int(seed); frames.append(d)
df = pd.concat(frames, ignore_index=True)
print("rows", len(df), df.groupby("model").seed.nunique().to_dict())

def boot(x, n=10000, s=0):
    r = np.random.default_rng(s); m = r.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

# 1) main table: per seed mean, then mean ± std over seeds
seedmeans = df.groupby(["model", "level", "seed"])[METRICS].mean().reset_index()
tab = seedmeans.groupby(["model", "level"])[METRICS].agg(["mean", "std"])
tab.to_csv(OUT / "table_seed_mean_std.csv")

# 2) paired comparisons
rows = []
for a, b in [("lorentz", "poincare"), ("poincare", "euclidean"), ("lorentz", "euclidean")]:
    for lv in LEVELS:
        for m in METRICS:
            A = df[(df.model == a) & (df.level == lv)].pivot(index="image", columns="seed", values=m)
            B = df[(df.model == b) & (df.level == lv)].pivot(index="image", columns="seed", values=m)
            A, B = A.loc[B.index], B
            seed_d = (A - B).mean(axis=0).values                 # per-seed mean difference (same seed = same init & data order)
            img_d = (A.mean(axis=1) - B.mean(axis=1)).values          # per-image diff of seed-averaged predictions
            lo, hi = boot(img_d)
            rows.append(dict(A=a, B=b, level=lv, metric=m, diff=img_d.mean(), ci_low=lo, ci_high=hi,
                             p_wilcoxon=wilcoxon(img_d).pvalue, seed_diffs=np.round(seed_d, 4).tolist(),
                             seeds_same_sign=int(max((seed_d > 0).sum(), (seed_d < 0).sum())),
                             frac_images_A_better=float(((img_d < 0) if "lpips" in m else (img_d > 0)).mean())))
cmp = pd.DataFrame(rows); cmp.to_csv(OUT / "paired_comparisons.csv", index=False)

# 3) curvature + val curves
hist = {}
for f in sorted(RES.glob("*_seed*/history.json")):
    hist[f.parent.name] = json.loads(f.read_text())
curv = {k: v["curvature"][-1] for k, v in hist.items() if v["curvature"][-1] is not None}
mins = {k: v["minutes"][-1] for k, v in hist.items()}

with open(OUT / "summary.txt", "w") as fh:
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    for m in ["psnr_fov", "ssim_fov", "lpips_fov", "psnr"]:
        t = seedmeans.groupby(["level", "model"])[m].agg(["mean", "std"]).unstack("model")
        fh.write(f"\n== {m} (mean / std over 3 seeds) ==\n{t.reindex(LEVELS).round(4)}\n")
    fh.write("\n== paired (A-B), image-level CI on seed-averaged predictions ==\n")
    show = cmp[cmp.metric.isin(["psnr_fov", "ssim_fov", "lpips_fov"])].copy()
    for c in ["diff", "ci_low", "ci_high"]: show[c] = show[c].round(4)
    show["p_wilcoxon"] = show["p_wilcoxon"].map(lambda x: f"{x:.1e}")
    show["frac_images_A_better"] = show["frac_images_A_better"].round(3)
    fh.write(show.to_string(index=False))
    fh.write(f"\n\ncurvature final: {json.dumps({k: round(v,4) for k,v in curv.items()})}\n")
    fh.write(f"minutes/run: {json.dumps({k: round(v,1) for k,v in mins.items()})}\n")
print(open(OUT / "summary.txt").read())

# 4) figures (matplotlib, paper style)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {"euclidean": "#2a78d6", "poincare": "#eb6834", "lorentz": "#1baf7a"}
MK = {"euclidean": "o", "poincare": "s", "lorentz": "^"}
LAB = {"euclidean": "Euclidean", "poincare": "Poincaré", "lorentz": "Lorentz"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444",
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "savefig.dpi": 300})

# Fig A: ΔPSNR_fov and ΔLPIPS_fov vs Euclidean across noise levels
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
x = np.arange(3)
for ax, m, ttl in [(axes[0], "psnr_fov", "Δ PSNR vs Euclidean (dB, retina only)"),
                   (axes[1], "lpips_fov", "Δ LPIPS vs Euclidean (lower is better)")]:
    ax.axhline(0, color="#999", lw=1, zorder=1)
    for k, mod in enumerate(["poincare", "lorentz"]):
        sub = cmp[(cmp.A == mod) & (cmp.B == "euclidean") & (cmp.metric == m)].set_index("level").loc[LEVELS]
        off = (k - 0.5) * 0.18
        ax.errorbar(x + off, sub["diff"], yerr=[sub["diff"] - sub.ci_low, sub.ci_high - sub["diff"]],
                    fmt=MK[mod], color=COL[mod], ms=6, lw=2, capsize=3, label=LAB[mod], zorder=3,
                    mec="white", mew=1)
        for i, lv in enumerate(LEVELS):   # per-seed differences as small dots
            ax.scatter(np.full(3, i + off + 0.06), sub.loc[lv, "seed_diffs"], s=8, color=COL[mod], alpha=0.5, zorder=2, lw=0)
    ax.set_xticks(x, ["light", "medium", "heavy"]); ax.set_xlabel("noise level"); ax.set_title(ttl, fontsize=8, loc="left")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_delta_vs_euclidean.png"); fig.savefig(OUT / "fig_delta_vs_euclidean.pdf")

# Fig B: curvature trajectories
fig, ax = plt.subplots(figsize=(3.3, 2.4))
for name, h in sorted(hist.items()):
    mod = name.rsplit("_seed", 1)[0]
    if mod == "euclidean": continue
    ax.plot(h["epoch"], h["curvature"], color=COL[mod], lw=2 if name.endswith("0") else 1.2,
            alpha=1 if name.endswith("0") else 0.55, label=LAB[mod] if name.endswith("seed0") else None)
ax.axhline(np.log(2), color="#999", ls="--", lw=1); ax.text(0.5, np.log(2) - 0.022, "ln 2", ha="left", fontsize=7, color="#666")
ax.set_xlabel("epoch"); ax.set_ylabel("learned curvature c"); ax.legend(frameon=False)
ax.set_title("Curvature (3 seeds each, no weight decay on c)", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_curvature.png"); fig.savefig(OUT / "fig_curvature.pdf")

# Fig C: validation PSNR curves
fig, ax = plt.subplots(figsize=(3.3, 2.4))
for name, h in sorted(hist.items()):
    mod = name.rsplit("_seed", 1)[0]
    ax.plot(h["epoch"], h["val_psnr"], color=COL[mod], lw=1.2, alpha=0.8, label=LAB[mod] if name.endswith("seed0") else None)
ax.set_ylim(36.5, 40); ax.set_xlabel("epoch"); ax.set_ylabel("val PSNR (dB)"); ax.legend(frameon=False, loc="lower right")
ax.set_title("Validation PSNR (3 seeds each)", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_val_psnr.png"); fig.savefig(OUT / "fig_val_psnr.pdf")
print("figures saved")
