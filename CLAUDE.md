# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Workspace for Kaggle competitions, built with PyTorch. It has two parts:

- `src/kagglekit/` — an installable package with reusable training utilities shared across notebooks
- `competitions/<name>/` — one directory per competition (currently only `titanic/`), laid out as:
  - `data/` — raw Kaggle CSVs plus preprocessed tensor files (`*.pt`) written by the prep notebook
  - `notebooks/` — numbered notebooks run in order (`00_eda`, `01_prep`, `02_linear`, `03_mlp`, `04_k_fold_cv`)
  - `models/` — saved model weights
  - `submissions/` — Kaggle submission CSVs, named `<model>_<optimizer>_<lr>_<epochs>.csv` (e.g. `mlp_adam_0.001_66.csv`)

`competitions/*/data/` and `competitions/*/submissions/` are gitignored. On a fresh clone, download the raw data with `kaggle competitions download -c titanic -p competitions/titanic/data && unzip competitions/titanic/data/titanic.zip -d competitions/titanic/data`, then run `01_prep` to regenerate the `.pt` files.

There is no test suite. Lint, format and type-check with `ruff check .`, `ruff format .` and `pyright`, configured in `ruff.toml` and `pyrightconfig.json`. Ruff skips notebooks and uses single quotes.

## Environment

- Python 3.13 virtualenv in `.venv/`. `pip install -r requirements.txt` installs the dependencies plus `kagglekit` in editable mode (`-e .`, configured by `pyproject.toml`), so edits under `src/` take effect after a kernel restart.
- Start Jupyter from inside a competition's `notebooks/` directory. Notebooks load and save data with relative paths like `../data/train.csv`.
- Run a notebook headless (from its `notebooks/` dir): `jupyter nbconvert --to notebook --execute --inplace 02_linear.ipynb`
- Submit: `kaggle competitions submit -c titanic -f competitions/titanic/submissions/<name>.csv -m "<message>"` (needs `~/.kaggle/kaggle.json`).

## kagglekit

Import these from `kagglekit` instead of redefining them in notebooks:
- `Dataset` wraps tensors into train/val `DataLoader`s.
- `LinearClassifier` and `MLP(in_features, out_features, hidden_features=8)`, a ReLU network (an int gives one hidden layer of that width, a list such as `[32, 16, 8]` one layer per entry, `[]` no hidden layer), follow the model convention: `forward` returns logits, and `predict()` returns 0/1 labels.
- `Trainer.fit(model, data)` runs the epoch loop on CUDA when available and returns a history dict of per-epoch lists: `train_loss`, plus `val_loss` and `val_accuracy` when validation data is given. Notebooks wrap it in `pd.DataFrame(history)` to view it.
- `accuracy` scores predictions against targets.
- `plot_histories(histories, baseline=...)` compares runs, given as a dict of name to `fit` history. It draws one panel per metric (train loss, val loss, val accuracy) with one line per run, a dot at each run's best val-loss epoch, and a dashed majority-class baseline. It returns the figure and supports at most 8 runs, one per palette color.
- `summarize_histories(histories)` returns a DataFrame with one row per run, scored at its best val-loss epoch and sorted by val loss.

Add new reusable models and helpers to the package. Code specific to one experiment stays in its notebook.

## Pipeline architecture (titanic)

The `.pt` files decouple preprocessing from modeling. Modeling notebooks only load tensors and never touch the CSVs.

**`00_eda.ipynb`** is a teaching and reference notebook, not part of the data flow. It walks through the questions used to find each engineered feature (column meaning, profiling, target rates, redundancy, interactions, structure in text, error analysis) and ends with a table mapping each finding to its feature. Update it when a feature is added or removed for a reason worth recording.

**`01_prep.ipynb`** owns all feature engineering. It is competition-specific, so it stays in the notebook rather than the package.
- `add_features(df)` holds the stateless row-wise steps: `Title` (rare titles grouped), `Sex_Pclass`, `AgeMissing`, `HasCabin`, `TicketCount`, `LogFarePP` (log fare per person on the ticket) and `FamilyGroup` (alone / 2-4 / 5+). It drops the ID/text/label columns and the raw columns these replace. `TICKET_COUNTS` is counted over train and test together, which uses no labels, so it is a module constant rather than a fitted statistic.
- `Preprocessor.fit(df)` learns every statistic (age medians per title, fare median, numeric means/stds, one-hot column set) from the split it is fitted on. `transform(df)` applies those statistics unchanged and reindexes to the fitted columns, so the train/val/test feature layouts match. Fit only on training data to avoid leakage.
- `fit` also learns group survival from labels: `GroupDied`/`GroupLived` flag women and boys whose other labelled family members (`woman_child_groups`: same surname and fare) all died or all survived. Men get 0. Rows of the fitted split have their own label subtracted, matched by `PassengerId` because train and test indexes overlap.
- It writes three files:
  - `train_val.pt`: keys `X_train, Y_train, X_val, Y_val, columns`. Stratified 80/20 split with `random_state=42`; the preprocessor is fitted on the 80% only.
  - `train_full.pt`: keys `X, Y, columns`. Preprocessor fitted on all of `train.csv`.
  - `test.pt`: keys `X, PassengerId, columns`. Transformed with the *same* preprocessor as `train_full.pt`, so a model trained on `train_full.pt` is the one to use for test predictions.
- Labels are saved as `int64`. Modeling code converts them to `float` with shape `(N, 1)` for `BCEWithLogitsLoss`.

Re-run `01_prep` whenever the features change. The column count of `X` changes with them, and models take `in_features=X.shape[1]`.

**`02_linear.ipynb`** and **`03_mlp.ipynb`** train a `LinearClassifier` / `MLP` on `train_val.pt`, compare runs against the majority-class baseline, then retrain on `train_full.pt` to write a submission.

**`04_k_fold_cv.ipynb`** runs stratified 5-fold CV on `train.csv` with a `Preprocessor` fitted per fold, so it keeps its own copy of `add_features` and `Preprocessor`. Keep that copy in sync with `01_prep`. The CV score uses the epoch with the lowest mean val loss across folds.
