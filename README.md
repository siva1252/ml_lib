# MLVerdict

MLVerdict is an evidence-based decision engine for **tabular** data.

The first check is not a model. It is a contract:

**Did you pass a target?**

- **Yes** → supervised. Classification or regression is read from that column.
- **No** → unsupervised. You must name the task. A missing target is not clustering.

After that branch, the engine profiles the table, records risk, compares real
algorithms, writes why a model won, and packages a reloadable artifact.
It does not invent labels. It does not hide blockers.

```bash
pip install mlverdict
```

[PyPI](https://pypi.org/project/mlverdict/) · [Source](https://github.com/siva1252/ml_lib)

```text
target given?
│
├── YES → Supervised
│         ├── Classification
│         └── Regression
│
└── NO  → Unsupervised
          ├── no task=                         → UNDECIDED (stop)
          ├── task="clustering"
          ├── task="anomaly_detection"
          └── task="dimensionality_reduction"
```

---

## Use

```python
from mlverdict import Verdict

run = Verdict().fit("customer_churn.csv", "churn")
print(run)
run.artifact().save("model.joblib")
```

```python
from mlverdict import Verdict

run = Verdict().fit("customers.csv", task="clustering")
print(run)
run.artifact().save("clusters.joblib")
```

```python
from mlverdict import Verdict

run = Verdict().fit("transactions.csv", task="anomaly_detection")
print(run)
run.artifact().save("anomalies.joblib")
```

```python
from mlverdict import Verdict

run = Verdict().fit("features.csv", task="dimensionality_reduction")
print(run)
run.artifact().save("pca.joblib")
```

`print(run)` is the verdict. `target=` and `task=` together are rejected.
No target and no task returns `UNDECIDED` — nothing is trained.

```python
from mlverdict import ModelArtifact

model = ModelArtifact.load("model.joblib")
print(model.metadata)
print(model.predict(new_rows))
```

---

## How supervised is checked

You passed a **target**. That is the only way into this path.

1. **Intent.** `resolve_fit_intent` sees a target and no `task`. Supervised Phase 1 starts.
2. **Hold-out.** A final test slice is locked first. It is not used to pick the model.
3. **DNA.** Columns are profiled. The target’s dtype and cardinality decide binary, multiclass, or regression. Ambiguous targets stay `UNDECIDED` until you pass `problem_type=`.
4. **Quality.** Missing labels, too few rows, no usable features, duplicates, IDs, constants, infinities. Blockers stop the run.
5. **Leakage.** Target-derived names, post-outcome columns, perfect correlation with `y`, ID columns. Signals — not a “clean data” certificate.
6. **Validation.** Stratified k-fold when labels allow; grouped or time splits if you pass `group_col` / `time_col`.
7. **Metrics.** From the labels: PR-AUC / ROC-AUC / F1 for classification, RMSE for regression. Accuracy is not the default on imbalanced classes.
8. **Models.** Linear, trees, boosting (optional XGBoost / LightGBM / CatBoost). Shortlist, then bounded HPO, then CV on train only.
9. **Decision.** Score, stability, generalization gap, latency, constraints. Reasons are written down.
10. **Confirm.** The locked test is scored **once** against true `y`. Then the artifact is gated (predict, reload, schema).

You know this path ran: you passed a target; `run.best()` has labeled scores; `artifact.schema.target_name` is that column.

---

## How unsupervised is checked

You did **not** pass a target. You passed `task=`. That is the only way into this path.

1. **Intent.** No target and no task → `UNDECIDED`. No silent k-means. `task=` must be `clustering`, `anomaly_detection`, or `dimensionality_reduction`.
2. **Hold-out.** A test slice is still locked first. There is no `y` to stratify on.
3. **DNA.** Every column is a feature. Target kind is `none`.
4. **Quality.** Same structural checks (size, usable features, IDs, missingness). There is no missing-label check because there is no label.
5. **Leakage.** ID and duplicate signals still apply. Target-correlation is skipped — there is no target.
6. **Validation.** K-fold (or grouped/time if you pass those columns). Preprocess still fits on training folds only.
7. **Metrics.** Unlabeled, on purpose:
   - clustering → **silhouette** (plus Calinski-Harabasz, Davies-Bouldin)
   - anomaly → **decision_std**, report **outlier_rate** (`1` inlier / `-1` outlier)
   - PCA → **explained_variance**, reconstruction RMSE
8. **Models.** KMeans / GMM / Birch; Isolation Forest / One-Class SVM / LOF; PCA. Each must `predict` on new rows.
9. **Decision.** Same multi-criteria ranking on those internal scores. Not accuracy. Not a fabricated label.
10. **Confirm.** Held-out features are scored with the same internal metric. Artifact metadata stores `learning_mode="unsupervised"` and the task.

You know this path ran: `run.extras["unsupervised_task"]` is set; `artifact.schema.target_name` is empty; metrics are silhouette / score-spread / variance — never ROC-AUC.

---

## What the library checks

Shared on both paths, after the branch is accepted:

| Check | What is inspected |
| --- | --- |
| Dataset DNA | Rows, dtypes, missingness, duplicates, constants, IDs, high cardinality, datetime / entity hints, scale, IID assumption |
| Quality | Too few rows, no usable features, duplicates, missing values, constants, potential IDs, infinities — blockers halt training |
| Leakage | Name patterns, IDs, duplicates; with a target, also y-correlation. Signals only |
| Validation | Strategy, fold count, locked final test before selection |
| Candidates | Compatible families only — not the full catalog blindly |
| Experiments | Baselines, bounded HPO, CV, train/val gap, latency |
| Decision | Why this model, who was rejected, assumptions, limitations |
| Production | Predict works, joblib reload matches, schema present |
| Artifact metadata | Winner, latency, schema version `1.0`, feature schema, metrics, full decision record, run config |

It does not check images, text, or deep learning. It does not certify a dataset as leakage-free.

---

## License

MIT. See [LICENSE](LICENSE).
