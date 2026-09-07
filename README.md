# MLVerdict

**Evidence-based model selection for tabular machine learning.**

[![PyPI](https://img.shields.io/pypi/v/mlverdict.svg)](https://pypi.org/project/mlverdict/)
[![Python](https://img.shields.io/pypi/pyversions/mlverdict.svg)](https://pypi.org/project/mlverdict/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/mlverdict.svg)](https://pypi.org/project/mlverdict/)

Install from [PyPI](https://pypi.org/project/mlverdict/). Source: [github.com/siva1252/ml_lib](https://github.com/siva1252/ml_lib).

MLVerdict is not a black-box AutoML that dumps a leaderboard and walks away. You give it a table and a target. It diagnoses the problem, picks metrics that will not lie, trains a shortlist of models, optionally tunes the winners, and **writes down why one model is the decision** — including leakage signals, stability, held-out test, and a deployable artifact.

```python
from mlverdict import Verdict

run = Verdict(enable_hpo=True).fit("train.csv", "churn")
print(run)                       # readable verdict, not a raw dict
run.artifact().save("model.joblib")
```

---

## Why this exists

Typical sklearn / boosting workflow:

1. Guess classification vs regression.
2. Report accuracy on an imbalanced label.
3. Train five models with default settings.
4. Pick the highest number.
5. Discover later that an ID column leaked, or that `f1 = 0` while accuracy looked fine.

MLVerdict automates the **decision path**, not just training:

| Step | What it does |
| --- | --- |
| Understand | Profile rows, types, imbalance, IDs, missingness |
| Diagnose | Binary / multiclass / regression — or stop if the target is ambiguous |
| Plan | Validation strategy + primary metric (e.g. `pr_auc` when classes are skewed) |
| Experiment | Linear, trees, sklearn boosting; XGBoost / LightGBM / CatBoost if installed |
| HPO | Bounded Optuna search on the top candidates only |
| Evaluate | Performance + stability + generalization + latency + complexity |
| Decide | Hard constraints first, then ranked evidence — no hardcoded favorite |
| Package | Preprocess + model + schema + decision record, ready to save and serve |

---

## Install

**Python 3.10+**

```bash
pip install mlverdict
```

Upgrade later:

```bash
pip install -U mlverdict
```

Optional gradient-boosting libraries (same comparison, extra candidates):

```bash
pip install "mlverdict[all]"
```

Install from source (this repo):

```bash
git clone https://github.com/siva1252/ml_lib.git
cd ml_lib
pip install -e ".[dev]"
```

### What you get with `pip install`

| Package | Role |
| --- | --- |
| `mlverdict` | The engine (`Verdict`, `Run`, `ModelArtifact`, CLI) |
| pandas, numpy | Tables |
| scikit-learn | Linear / tree / HGB models, metrics, CV |
| optuna | Bounded hyperparameter search |
| joblib | Save / load artifacts |

---

## Two-minute start

```python
from mlverdict import Verdict

run = Verdict(enable_hpo=True).fit("train_with_label.csv", "churn")
print(run)
```

That is the whole modeling loop. `print(run)` is the product surface: winner, why this metric, leaderboard table, held-out scores, traps such as high accuracy with zero recall, watchlist, and how to save the model.

Need a CSV first? From this repo:

```bash
python examples/quickstart.py
```

### CLI

```bash
python -m mlverdict fit train_with_label.csv --target churn --save model.joblib
python -m mlverdict serve model.joblib
```

`POST /predict` with `{"records": [{...}, ...]}`.

### Save, reload, predict

```python
from mlverdict import ModelArtifact

run.artifact().save("model.joblib")
art = ModelArtifact.load("model.joblib")
art.predict(new_rows)
```

The joblib file is **preprocess + model + schema**. You do not rewrite training code to deploy.

---

## What `fit()` actually does

```text
CSV / DataFrame
    → lock a final test split (untouched until the end)
    → dataset DNA + quality + leakage signals
    → problem type
    → metric + validation plan
    → shortlist compatible models
    → cross-validation baselines
    → optional HPO on the top 2
    → multi-criteria ranking
    → score the held-out test once
    → production-readiness checks
    → trained artifact
```

**Always in the comparison:** Logistic Regression or Ridge, Random Forest, Extra Trees, HistGradientBoosting.

**If you installed extras:** XGBoost, LightGBM, CatBoost.

Tiny data skips overkill boosters. Identifiers such as `customer_id` are flagged and dropped from features.

---

## Reading the verdict

Do not print `run.best()` and `run.leaderboard()` for humans. Those are for code.

| Call | For |
| --- | --- |
| `print(run)` | The decision, in English |
| `run.report()` | Full markdown record |
| `run.best()` | Dict for scripts |
| `run.leaderboard()` | DataFrame for scripts |
| `run.artifact()` | Deployable pipeline |

If held-out **accuracy is high** but **precision / recall / f1 are 0**, the model never predicted the rare class at the 0.5 cutoff. Ranking metrics (`pr_auc`, `roc_auc`) can still look fine. MLVerdict prints an **ATTENTION** block for that trap. Production packaging checks (save/reload) are not a certificate that the business metric is good.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Coverage includes problem detection, leakage/quality, candidate selection, end-to-end fit on synthetic and sklearn datasets (breast cancer, iris, wine, diabetes), display/report formatting, and CLI save.

---

## Project layout

```text
ml_lib/
├── src/mlverdict/          # library (pip installable)
│   ├── api/                # Verdict, Run, CLI
│   ├── data/               # load, profile, DNA
│   ├── problem/            # classification vs regression
│   ├── metrics/            # metric plan
│   ├── models/             # catalog + sklearn / booster adapters
│   ├── experiments/        # CV + bounded HPO
│   ├── evaluation/         # stability, latency, composite score
│   ├── decision/           # winner + written record
│   ├── reporting/          # print(run) + markdown report
│   └── production/         # artifact + local serve
├── tests/                  # pytest
├── examples/               # runnable quickstart
├── pyproject.toml
├── LICENSE                 # MIT
└── README.md
```

---

## Versioning PyPI vs Git

This repo is GitHub. [PyPI](https://pypi.org/project/mlverdict/) is a separate upload.

| You do | GitHub | `pip install mlverdict` |
| --- | --- | --- |
| `git push` | updates | no |
| bump version + `python -m build` + `twine upload` | no | yes |

Same version number cannot be uploaded twice. After this README, the published package is **0.1.1**.

---

## License

MIT. See [LICENSE](LICENSE).
