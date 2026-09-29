# Experiments: plan, status and history

## 1. Runs for the paper

Status: ⬜ not started · 🟡 running · ✅ done (evaluated)

| ID | Config | Seeds | Est. GPU time (T4) | Status | Purpose |
|---|---|---|---|---|---|
| E | `euclidean` | 0, 1, 2 | ~2 h each | ⬜ | baseline |
| P | `poincare`  | 0, 1, 2 | ~3 h each | ⬜ | hyperbolic encoder, Poincaré ball |
| L | `lorentz`   | 0, 1, 2 | ~3 h each | ⬜ | same encoder, Lorentz model |

Total ≈ 24 GPU-hours → about one Kaggle week (30 h quota). Order: `smoke_test` → L seed 0 → P seed 0 →
E seed 0 → remaining seeds.

**Optional, if time allows** (the "where does the HyperbolicCV gain come from" ladder):
Lorentz-native FC layer · strided-conv downsampling · Lorentz BatchNorm · Euclidean control with
strided conv + BatchNorm. Plus a diagnostic: fraction of Poincaré activations clipped at the ball boundary.

## 2. Protocol decisions (and why)

| Decision | Why |
|---|---|
| One codebase, one `evaluate.py` for all models | the internship tables mixed random-crop and center-crop evaluations (±0.3–0.5 dB), larger than the effects being measured |
| Test on full 384×384 images, fixed noise seed | no crop ambiguity; every model sees the same noisy inputs |
| Batch 4 for **all** models | the internship baseline used batch 16 vs. 4 for hyperbolic models → 4× more optimisation steps for the hyperbolic runs |
| fp32 for all models (no AMP) | AMP was on for Euclidean only; numerical precision must not differ, it is what we study |
| Same optimiser implementation (hypll RiemannianAdam) for all | internship used AdamW for Euclidean, RiemannianAdam for hyperbolic |
| No weight decay on the curvature | hypll stores c = softplus(raw); weight decay drives raw → 0, i.e. c → ln 2 ≈ 0.693 regardless of the task (Phase 3.1 ended at exactly 0.6931) |
| 3 seeds + paired Wilcoxon + bootstrap CI | internship differences (0.1–0.2 dB) were single-seed |
| Lorentz lift `expmap0(2x)`, skips `logmap0(h)/2` | hypll's Poincaré tangent vectors at the origin are ½ of the Lorentz ones; this makes both encoders the same function at equal weights |

Consequence: the new numbers are **not** directly comparable with the internship report; everything
is re-run under this protocol.

## 3. History: internship notebooks (not in this repo)

Kept in the personal folder `hyperbolic_dl_article/` for reference only.

| Old notebook | Old phase | What it was | Fate |
|---|---|---|---|
| `phase2-eucledian-base.ipynb` | 2.1 | Euclidean U-Net, L1, bs16, AMP | → `configs/euclidean.yaml` (protocol changed) |
| `phase2-earlier-hyperbolic-poincar.ipynb` | 3.1 | single Poincaré block near bottleneck | dropped (partial intervention, weaker) |
| `hyperbolic-encoder-eucledian-encoder.ipynb` | 4 | full Poincaré encoder, L1 | → `configs/poincare.yaml` |
| `lorentz.ipynb` | 5 | Lorentz encoder with HyperbolicCV layers (BN, strided conv) | motivates the paper (val PSNR 40.28 vs 39.49); not a controlled comparison → optional ladder |
| `08-…perceptual`, `09-…perceptual`, `notebook1e96…`, `notebook3720…` | 5a/5b/5c | L1 + VGG perceptual loss variants | out of scope for this paper |
| `lpips-comparison.ipynb` | — | center-crop re-evaluation of 2.1 / 3.1 / 4 | replaced by `evaluate.py` |
| `06-lorentz-encoder-hypll.ipynb` | 6 | first Lorentz-hypll notebook (Phase 4 clone) | replaced by `configs/lorentz.yaml` |

Known issues of the internship report (fixed by the protocol above): curvature is softplus-parameterised
(the reported "0.1 → 0.69 drift" is really 0.744 → ln 2 under weight decay); mixed evaluation protocols;
single seeds; batch-size / AMP / optimiser confounds.

## 4. Log

| Date | What |
|---|---|
| 2026-09-28 | Audit of internship notebooks; Lorentz manifold written for hypll (59 tests pass on Kaggle) |
| 2026-09-29 | Repo created; protocol fixed |
