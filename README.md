# MLVerdict

**An evidence-based decision engine for tabular machine learning.**

MLVerdict does not ask you to assemble scikit-learn by hand. It takes a table,
determines the learning problem, selects a metric that will not flatter a bad
model, trains a constrained set of estimators, and **records a verdict**: which
model is justified, under which assumptions, with which risks.

```bash
pip install mlverdict
```

[PyPI](https://pypi.org/project/mlverdict/) · [GitHub](https://github.com/siva1252/ml_lib)

```python
from mlverdict import Verdict

run = Verdict(enable_hpo=True).fit("data.csv", "target")
print(run)
run.artifact().save("model.joblib")
```

`data.csv` and `"target"` are **your** path and **your** column name. The
package does not include a sample dataset.

---

## What we built

A **supervised decision path** for rectangular data (CSV / DataFrame), not a
notebook template and not a neural-net trainer.

| Stage | System responsibility |
| --- | --- |
| 1. Isolate | Hold out a final test split before any model is chosen |
| 2. Profile | Schema, scale, missingness, imbalance, identifier columns |
| 3. Diagnose | Binary classification, multiclass, or regression — or refuse if the target is ambiguous |
| 4. Govern | Validation strategy + primary metric matched to the problem |
| 5. Search | Linear, tree, and boosting families (optional XGBoost / LightGBM / CatBoost) |
| 6. Refine | Bounded hyperparameter search on the shortlist only |
| 7. Decide | Rank on performance, stability, generalization, latency, and complexity |
| 8. Certify | Score the untouched test **once**; package preprocess + model as one artifact |

The output is a written judgment (`print(run)`), not a raw dict of scores.
If accuracy looks strong while the rare class is never predicted, the verdict
says so.

---

## Step 1 — Classify the problem (do this before you call `fit`)

Machine learning on a table is not one task. First decide **supervised vs
unsupervised**. MLVerdict automates the supervised branch. The unsupervised
branch is a different class of methods; the check below tells you which one
you are in.

### The check

Ask one question:

> After I remove a single column, can I still state the business question?

| Answer | You are in | Typical column |
| --- | --- | --- |
| No — that column **is** the question (will they leave, what is the price, which class) | **Supervised** | `churn`, `price`, `label`, `default`, `diagnosis` |
| Yes — the question is about **structure in the table itself** | **Unsupervised** | there is no outcome column |

### Supervised (labels exist)

You already know the correct answer on historical rows. The model’s job is to
predict that same field on **new** rows.

| Subtype | How it looks | Example |
| --- | --- | --- |
| Binary classification | Target has two values | churn / not, fraud / not |
| Multiclass classification | Target has a small set of names or codes | product line, species |
| Regression | Target is a quantity | price, demand, score |

**How to confirm:** open the file, find the column you will want in production
when it is *not yet known*, and pass that name as `target`.

### Unsupervised (no labels)

You are not predicting a known field. You are asking the data to describe
itself.

| Intent | How it looks | Typical methods (not this package) |
| --- | --- | --- |
| Clustering | “Are there natural groups of customers / sensors?” | k-means, GMM, hierarchical |
| Anomaly detection | “Which rows are rare without a fraud label?” | isolation forest, one-class SVM |
| Dimensionality reduction | “Can I compress 200 columns to a few factors?” | PCA, UMAP |

**How to confirm:** there is no column you would call the *answer*. If you
invented a target tomorrow, you would need humans or another system to label
it first — until then, the problem is unsupervised.

### What MLVerdict does with each

| Branch | This library |
| --- | --- |
| Supervised + a real `target` | **In scope.** `fit(path, target)` runs the full decision engine. |
| Unsupervised / no target | **Out of scope in current releases.** We do not silently cluster your table or invent a label. Use a dedicated unsupervised stack, or label a target and return here. |

Calling `fit` without a target is invalid on purpose: a verdict without an
outcome is not a supervised decision.

---

## Step 2 — Install

Requires Python 3.10+.

```bash
pip install mlverdict
```

Optional extra candidates (same race, more boosting libraries):

```bash
pip install "mlverdict[all]"
```

Git clone is **not** part of using the library. It is only for reading source
or running the project’s tests.

---

## Step 3 — Run the supervised engine

```python
from mlverdict import Verdict

run = Verdict(enable_hpo=True).fit("your_file.csv", "your_target_column")
print(run)
```

| Argument | Contract |
| --- | --- |
| First | Path to **your** CSV, or a pandas `DataFrame` |
| Second | Exact name of the **supervised** target column |

You do not select algorithms. The engine includes Logistic Regression or
Ridge, Random Forest, Extra Trees, and HistGradientBoosting, then optional
boosters if installed. Families that do not apply (wrong task, tiny data,
missing extra) are excluded with a reason.

---

## Step 4 — Read the verdict

`print(run)` is the product surface: selected model, why that metric, the
leaderboard, held-out test, quality/leakage watchlist, and save instructions.

| API | Role |
| --- | --- |
| `print(run)` | Human-readable decision |
| `run.report()` | Full markdown record |
| `run.best()` / `run.leaderboard()` | Programmatic access |
| `run.artifact()` | Fitted preprocess + model |

---

## Step 5 — Persist and score new rows

```python
from mlverdict import ModelArtifact

run.artifact().save("model.joblib")
model = ModelArtifact.load("model.joblib")
model.predict(new_rows)
```

The artifact is the training pipeline, not weights alone. New rows must carry
the same feature columns (not the target).

---

## Search space (transparency, not a menu)

These estimators are the **candidates the engine may train**. They are listed
so the comparison is inspectable. You do not import them.

- Linear: Logistic Regression, Ridge  
- Trees: Random Forest, Extra Trees  
- Boosting: HistGradientBoosting; XGBoost, LightGBM, CatBoost when present  

---

## Development

```bash
git clone https://github.com/siva1252/ml_lib.git
cd ml_lib
pip install -e ".[dev]"
pytest
```

---

## License

MIT. See [LICENSE](LICENSE).
