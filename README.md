# MLVerdict

**Evidence-based decision engine for tabular data.**

The first thing the library does is **not** train a model. It asks the same
question the product is built on: **did you give a target?**

```text
Did you pass a target?
│
├── YES → Supervised
│         ├── Classification   (binary or multiclass, from the target)
│         └── Regression       (numeric target)
│         └── then the full Phase 1 lifecycle
│             (profile → models → verdict → artifact)
│
└── NO  → Unsupervised
          │
          ├── no task=                         → UNDECIDED (objective unknown)
          ├── task="clustering"                → runs clustering
          ├── task="anomaly_detection"         → runs anomaly detection
          └── task="dimensionality_reduction"  → runs PCA projection
```

Missing a target does **not** mean “run k-means”. You must name the task.

```bash
pip install mlverdict
```

[PyPI](https://pypi.org/project/mlverdict/) · [Source](https://github.com/siva1252/ml_lib)

---

## Where this is checked

On every `fit()` call, **before** profiling, CV, HPO, or artifacts:

1. `Verdict.fit(...)` receives `target` and optional `task`
2. `resolve_fit_intent` in `src/mlverdict/problem/intent.py` classifies the call
3. Supervised (`target=`) → classification / regression Phase 1
4. Unsupervised (`task=` and no target) → clustering / anomaly / PCA Phase 1
5. No target and no task → **stop** (`UNDECIDED`). No invented label. No fake metrics.

You can inspect what was decided:

```python
run.status                          # DECIDED | UNDECIDED | BLOCKED
run.extras.get("learning_mode")     # omitted on supervised success;
                                    # "unsupervised_candidate" or "unsupervised"
run.extras.get("unsupervised_task") # None, or clustering / anomaly_detection /
                                    # dimensionality_reduction
run.notes                           # plain-language reason when the run halted
run.best()                          # winner + scores when a model was selected
run.leaderboard()
run.artifact()                      # packaged pipeline when DECIDED
print(run)                          # human-readable verdict
```

---

## What you give vs what you get

Use **your** file and **your** column names. Nothing is shipped as sample data.

### 1. Supervised — you give a target

**You give:** a table + the name of the column you want to predict.

```python
from mlverdict import Verdict

run = Verdict().fit("customer_churn.csv", "churn")
run = Verdict().fit("customer_churn.csv", target="churn")  # same
print(run.status)        # DECIDED when Phase 1 can select a model
print(run.best())        # winner + scores
print(run.leaderboard())
print(run)
run.artifact().save("model.joblib")
```

**We give:** problem type (classification vs regression from that column),
labeled metrics, model comparison, written verdict, held-out test, deployable
artifact.

**How you know it is this branch:** you passed `target`. No `task` needed.

---

### 2. No target, no task — unsupervised candidate, not chosen

**You give:** a table only.

```python
run = Verdict().fit("customers.csv")

print(run.status)                          # UNDECIDED
print(run.extras["learning_mode"])         # unsupervised_candidate
print(run.extras["unsupervised_task"])     # None
print(run.notes)
print(run.best())                          # None
print(run.artifact())                      # None
```

**We give:** a halt, not a model. Clustering vs anomaly vs compression must
be stated with `task=`.

---

### 3. Clustering

**You give:** a table and `task="clustering"`. No target.

```python
from mlverdict import Verdict

run = Verdict().fit("customers.csv", task="clustering")
print(run)
print(run.best())
print(run.leaderboard())
labels = run.artifact().predict(new_rows)   # cluster ids
run.artifact().save("clusters.joblib")
```

**What runs:** KMeans, Gaussian Mixture, Birch (models that can `predict` on
new rows). Primary metric is **silhouette** (plus Calinski-Harabasz and
Davies-Bouldin). These are unlabeled structure scores, not accuracy.

---

### 4. Anomaly detection

```python
run = Verdict().fit("transactions.csv", task="anomaly_detection")
print(run)
flags = run.artifact().predict(new_rows)    # 1 = inlier, -1 = outlier
```

**What runs:** Isolation Forest, One-Class SVM, Local Outlier Factor
(`novelty=True`). Primary metric is **decision_std** (spread of anomaly
scores); **outlier_rate** is reported. There is no labeled precision unless
you have labels — and this path does not invent them.

---

### 5. Dimensionality reduction

```python
run = Verdict().fit("features.csv", task="dimensionality_reduction")
print(run)
z = run.artifact().predict(new_rows)        # projected components
```

**What runs:** PCA (with `predict()` = `transform()` so the same artifact
API as the other paths). Primary metric is **explained_variance**;
reconstruction RMSE is secondary.

Invalid task (`task="forecasting"`) raises `ConfigurationError`.
`target=` and `task=` together is rejected — pick one branch of the tree.

---

## How to try it (copy-paste)

```python
from mlverdict import DecisionStatus, Verdict
import pandas as pd
import numpy as np

rng = np.random.default_rng(0)
a = rng.normal(size=(60, 2))
b = rng.normal(loc=6, size=(60, 2))
customers = pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])

# no target, no task → stop
undecided = Verdict().fit(customers)
assert undecided.status == DecisionStatus.UNDECIDED

# clustering → real algorithms, real artifact
run = Verdict(enable_hpo=False).fit(customers, task="clustering")
assert run.status == DecisionStatus.DECIDED
assert run.artifact() is not None
print(run)
```

From this repo, the same checks live in `tests/test_entry.py`:

```bash
pytest tests/test_entry.py
```

---

## What runs after the branch is accepted

Same skeleton for supervised and unsupervised:

```text
Load dataset → lock final test → profile / DNA → problem type
→ quality + leakage signals → validation + metrics → candidates
→ baselines → HPO → CV → multi-criteria evaluation → decision
→ untouched final test → production readiness → artifact
```

Supervised metrics are labeled (PR-AUC, RMSE, …). Unsupervised metrics are
internal (silhouette, score spread, explained variance). The library will
not fabricate a target to reuse classification scores.

---

## Persist (DECIDED runs)

```python
from mlverdict import ModelArtifact
run.artifact().save("model.joblib")
ModelArtifact.load("model.joblib").predict(new_rows)
```

---

## Remaining work

- Deep learning / non-tabular data
- Hosted deploy beyond `python -m mlverdict serve`
- Labeled evaluation for anomaly detection when a user later supplies labels

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
