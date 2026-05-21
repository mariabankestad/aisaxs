# Data

This directory holds the configuration files used by the training and
fitting scripts, plus a small `metadata.pt` (normalization stats and the
`q` grid) that ships with the repository so the surrogate tutorial and
inference with the pretrained model work out of the box. The large
training tensors (a few hundred MB) are **not** distributed with the
repository -- they are hosted externally.

## What ships with the repo

```
data/lnp/setup_shell_shift/
└── metadata.pt          # 5 KB; normalization stats (y_mean, y_std,
                         # x_min, x_max) and the q grid. Enough for the
                         # surrogate tutorial and for inference with the
                         # pretrained surrogate.
```

`metadata.pt` is sufficient for everything except re-running the training
pipeline. For retraining, the larger files below are needed.

## Training data: external download

The pretraining tensors live at: **\[Zenodo / Hugging Face URL to be added\]**

Expected layout after download:

```
data/lnp/setup_shell_shift/
├── metadata.pt          # already in the repo (see above)
├── input_data.pt        # parameter vectors used for training
├── output_data.pt       # corresponding simulated SAXS curves
├── train.pt             # train split
├── val.pt               # validation split
├── test.pt              # held-out test split
└── splits.pt            # train/val/test indices (only needed if you want
                         # to re-run the preprocessing in
                         # `lnp/preprocess_ml_data.py`)
```

Place the files under `data/lnp/setup_shell_shift/`. The training and fitting
scripts expect this exact path.

## Checksums

\[SHA256 checksums to be added when the deposit is finalised\]

## Quick download

```bash
mkdir -p data/lnp/setup_shell_shift
# replace <URL> with the Zenodo/HF URL for each file
curl -L -o data/lnp/setup_shell_shift/input_data.pt   <URL>/input_data.pt
curl -L -o data/lnp/setup_shell_shift/output_data.pt  <URL>/output_data.pt
curl -L -o data/lnp/setup_shell_shift/train.pt        <URL>/train.pt
curl -L -o data/lnp/setup_shell_shift/val.pt          <URL>/val.pt
curl -L -o data/lnp/setup_shell_shift/test.pt         <URL>/test.pt
curl -L -o data/lnp/setup_shell_shift/splits.pt       <URL>/splits.pt
```

## Experimental data

The experimental MC3 LNP SAXS curve used in the paper is **not redistributed**
with this repository. To reproduce the experimental fits, supply your own
SAXS measurement and pass it to `fit_experimental_curve.py`.

## Generating your own synthetic data

The physical forward model used to simulate SAXS curves is documented in
[`notebooks/tutorial_simulation_model.ipynb`](../notebooks/tutorial_simulation_model.ipynb):
it builds the simulation context and produces monodisperse and polydisperse
curves from the physical parameters. Use it as the starting point for
generating your own training set, serialising the resulting parameter and
curve pairs into the layout shown above so the preprocessing and training
scripts can consume them.

## Configuration files

| File | Purpose |
|---|---|
| `lnp/preprocess_ml_data.py` | Cubic-Hermite interpolation utilities and a CLI to build train/val/test splits from a raw simulation set. Imported by all fitting scripts. |
| `lnp/config.yaml` | Parameter ranges, q-grid settings, and the raw-data paths used by the preprocessing pipeline. Loaded by the fitting scripts and the surrogate tutorial. |
