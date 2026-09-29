# Geometry vs. architecture in hyperbolic U-Nets for retinal image denoising

Hyperbolic encoders are often reported to help medical imaging models, but the comparisons usually change
several things at once (geometry, layers, normalisation, optimiser). This repository isolates **the model of
hyperbolic space** as the only variable:

| Experiment | Encoder | What changes vs. the row above |
|---|---|---|
| `euclidean` | Euclidean U-Net encoder | — (baseline) |
| `poincare`  | hyperbolic encoder, Poincaré ball (hypll HNN++ layers) | encoder geometry |
| `lorentz`   | **same** hyperbolic encoder, Lorentz model | only the manifold object |

The Lorentz and Poincaré encoders use the *same* hypll layers and parameters and compute the *same* function
at identical weights (they are isometric; checked in `tests/test_models.py`). Any difference in results
therefore comes from the **numerical representation** (e.g. the Poincaré ball clipping points near its
boundary in fp32), not from the geometry or the architecture.

The Lorentz manifold for hypll was written for this project (`third_party/hypll/hypll/manifolds/lorentz`)
and will be proposed upstream.

## Repository layout

```
configs/            one YAML per experiment; everything shared lives in base.yaml
src/hypdenoise/
    data.py         RFMiD splits, Poisson–Gaussian noise, THE evaluation protocol
    models.py       Euclidean / Poincaré / Lorentz U-Net denoisers
    train.py        training loop (one config, one seed)
    evaluate.py     per-image PSNR / SSIM / LPIPS on the test set
    compare.py      mean ± 95% bootstrap CI and paired Wilcoxon tests across models
notebooks/
    kaggle_runner.ipynb   clones the repo and runs one experiment on a Kaggle T4
tests/              fast CPU checks (shapes, Poincaré ≡ Lorentz at equal weights)
third_party/hypll/  hypll (commit 62ea788) + our Lorentz manifold and its tests
docs/EXPERIMENTS.md experiment plan, protocol decisions, history of the project
results/            run outputs (git-ignored except aggregated tables)
```

## Protocol (identical for every model)

* **Data**: RFMiD official splits (1920 train / 640 val / 640 test), images resized to 384×384.
* **Noise**: Poisson–Gaussian, Var = α·μ + σ², levels light (0.5, 3), medium (1, 5), heavy (2, 8), 8-bit units.
  Training samples one level per image (blind denoising).
* **Training**: L1 loss, 30 epochs, batch 4, RiemannianAdam (lr 2e-4, wd 1e-4, no wd on curvature),
  1-epoch warm-up + cosine, grad-clip 1.0, **fp32 for all models**, random 256×256 crops + dihedral flips.
* **Model selection**: best validation PSNR (fixed validation crops and noise).
* **Test**: full 384×384 images, each noise level separately, fixed noise per image, per-image
  PSNR / SSIM / LPIPS(Alex). Seeds 0, 1, 2. Paired statistics across models.

## Usage

```bash
pip install -r requirements.txt
pip install -e third_party/hypll -e .
pytest tests -q

python -m hypdenoise.train    --config configs/lorentz.yaml --seed 0
python -m hypdenoise.evaluate --run results/lorentz_seed0
python -m hypdenoise.compare  --results results --reference euclidean
```

On Kaggle, use `notebooks/kaggle_runner.ipynb` (set `EXPERIMENT` and `SEEDS`).
