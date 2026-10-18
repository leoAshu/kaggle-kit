# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Notebook-based workspace for Kaggle competitions, built with PyTorch. Each competition gets its own top-level directory (currently only `titanic/`) with this layout:

- `data/` — raw Kaggle CSVs plus preprocessed tensor files (`*.pt`) written by the prep notebook
- `notebooks/` — numbered notebooks run in order (`00_eda_prep`, `01_baseline`, ...)
- `models/` — saved model weights
- `submissions/` — Kaggle submission CSVs, named after the notebook that produced them (e.g. `01_baseline.csv`)

`data/` and `submissions/` are gitignored. On a fresh clone, download the raw data with `kaggle competitions download -c titanic -p titanic/data && unzip titanic/data/titanic.zip -d titanic/data`, then run `00_eda_prep` to regenerate the `.pt` files.

There is no package, test suite, or linter; all code lives in notebook cells.

## Environment

- Python 3.13 virtualenv in `.venv/` (`source .venv/bin/activate`), dependencies in `requirements.txt`.
- `requirements.txt` ends with `-e .`, an editable install of this repo. There is no `pyproject.toml`/`setup.py` yet, so that line fails until one is added.
- Start Jupyter from inside a competition's `notebooks/` directory. Notebooks load and save data with relative paths like `../data/train.csv`.
- Run a notebook headless: `jupyter nbconvert --to notebook --execute --inplace titanic/notebooks/00_eda_prep.ipynb`
- Submit: `kaggle competitions submit -c titanic -f titanic/submissions/<name>.csv -m "<message>"` (needs `~/.kaggle/kaggle.json`).

## Pipeline architecture (titanic)

The `.pt` files decouple preprocessing from modeling. Modeling notebooks only load tensors and never touch the CSVs.

**`00_eda_prep.ipynb`** owns all feature engineering:
- `add_features(df)` holds the stateless row-wise steps: it derives `Deck` and `CabinCount` from `Cabin` and drops the ID/text/label columns.
- `Preprocessor.fit(df)` learns every statistic (imputation values, numeric means/stds, one-hot column set) from the split it is fitted on. `transform(df)` applies those statistics unchanged and reindexes to the fitted columns, so the train/val/test feature layouts match. Fit only on training data to avoid leakage.
- It writes three files:
  - `train_val.pt`: keys `X_train, Y_train, X_val, Y_val, columns`. Stratified 80/20 split with `random_state=42`; the preprocessor is fitted on the 80% only.
  - `train_full.pt`: keys `X, Y, columns`. Preprocessor fitted on all of `train.csv`.
  - `test.pt`: keys `X, PassengerId, columns`. Transformed with the *same* preprocessor as `train_full.pt`, so a model trained on `train_full.pt` is the one to use for test predictions.
- Labels are saved as `int64`. Modeling code converts them to `float` with shape `(N, 1)` for `BCEWithLogitsLoss`.

Re-run `00_eda_prep` whenever the features change. The column count of `X` changes with them, and models take `in_features=X.shape[1]`.

**`01_baseline.ipynb`** sets the modeling conventions later notebooks follow. It defines small reusable classes inline:
- `Dataset` wraps tensors into train/val `DataLoader`s.
- The model is an `nn.Module` that returns logits and has a `predict()` method returning 0/1 labels.
- `Trainer.fit(model, data)` runs the epoch loop on CUDA when available and displays a per-epoch DataFrame of train loss, val loss, and val accuracy.

Compare results against the majority-class baseline it prints.
