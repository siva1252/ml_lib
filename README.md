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

Text labels work as-is (`"yes"/"no"`, `"churn"/"stay"`). For binary targets, precision, recall,
F1 and PR-AUC measure the **minority class** by default; choose another with `positive_label=`:

```python
run = Verdict().fit("loans.csv", "status", positive_label="default")
```

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
3. **DNA.** Columns are profiled. The target’s dtype and cardinality decide binary, multiclass, or regression. Integer class codes `0..k-1` (iris, wine) are multiclass, with a warning. Genuinely ambiguous targets, such as a `1.0–5.0` rating, stay `UNDECIDED` until you pass `problem_type=`.
4. **Quality.** Missing labels, too few rows, no usable features, duplicates, IDs, constants, infinities. Blockers stop the run.
5. **Leakage.** Target-derived names, post-outcome columns, perfect correlation with `y`, ID columns. Signals — not a “clean data” certificate.
6. **Validation.** Stratified k-fold when labels allow; grouped or time splits if you pass `group_col` / `time_col`.
7. **Metrics.** From the labels: PR-AUC / ROC-AUC / F1 for classification, RMSE for regression. Accuracy is not the default on imbalanced classes. Binary metrics score the positive class (minority by default, or `positive_label=`).
8. **Models.** Every compatible model from the table below is cross-validated; the strongest tunable ones get bounded HPO. SVM and KNN are skipped automatically on data too large for them.
9. **Decision.** Score, stability, generalization gap, latency, constraints. Score gaps smaller than cross-validation noise are not treated as wins, so a simpler, faster, more stable model can take a statistical tie. Reasons are written down.
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
   - clustering → **silhouette** (plus Calinski-Harabasz, Davies-Bouldin). Silhouette is computed on a fixed-seed sample on large data.
   - anomaly → **anomaly_separation** (how far flagged rows sit from the rest), plus **outlier_rate** (`1` inlier / `-1` outlier)
   - PCA → **explained_variance**, reconstruction RMSE
8. **Models.** KMeans / GMM / Birch; Isolation Forest / One-Class SVM / LOF / Detector Ensemble; PCA. Each must `predict` on new rows.
   - clustering: every `k` from 2 to `max_clusters` is tried (on a sample for large data), not a few random guesses.
   - anomaly: every detector flags the same expected share (`anomaly_contamination`, default 5%), so no detector can "win" by flagging more rows.
   - PCA: keeps the fewest components that explain `pca_variance_threshold` (default 95%) of the variance.
9. **Decision.** Clustering uses the same multi-criteria ranking on internal scores. Anomaly detection uses the **Detector Ensemble** by default. Without labels no internal score reliably ranks detectors (each one fails on some outlier shape), and the max-vote ensemble stays close to the best single detector on all of them. Use constraints (`require_explainable=True` keeps Isolation Forest) to force a single detector.
10. **Confirm.** Held-out features are scored with the same internal metric. Artifact metadata stores `learning_mode="unsupervised"` and the task.

You know this path ran: `run.extras["unsupervised_task"]` is set; `artifact.schema.target_name` is empty; metrics are silhouette / score-spread / variance — never ROC-AUC.

---

## Models

| Task | Models |
| --- | --- |
| Classification | Logistic Regression, Decision Tree, K-Nearest Neighbors, Support Vector Machine, Gaussian Naive Bayes, Random Forest, Extra Trees, HistGradientBoosting, XGBoost\*, LightGBM\*, CatBoost\* |
| Regression | Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, K-Nearest Neighbors, Support Vector Machine, Random Forest, Extra Trees, HistGradientBoosting, XGBoost\*, LightGBM\*, CatBoost\* |
| Clustering | KMeans, Gaussian Mixture, Birch |
| Anomaly detection | Detector Ensemble (default), Isolation Forest, One-Class SVM, Local Outlier Factor |
| Dimensionality reduction | PCA |

\* Optional: `pip install "mlverdict[xgboost]"`, `"mlverdict[lightgbm]"`, `"mlverdict[catboost]"`, or `"mlverdict[all]"`.

Skipped on purpose: models that cannot `predict` on new rows (DBSCAN, agglomerative clustering, t-SNE),
because every MLVerdict artifact must score data it has not seen.

## Configuration

```python
from mlverdict import Verdict

Verdict(
    cv_splits=5,                    # folds on the training split
    max_hpo_trials=12,              # per tuned model
    anomaly_contamination=0.02,     # expected outlier share
    pca_variance_threshold=0.90,    # PCA size
    max_clusters=12,                # largest k tried
    constraints={"require_explainable": True, "max_latency_ms": 20},
)
```

The command line exposes the common ones: `mlverdict fit data.csv --target y --positive-label yes`,
`--contamination 0.02`, `--fast`, `--save model.joblib`, and `mlverdict serve model.joblib`.

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

Prediction is strict by default: a value like `"abc"` in a numeric column raises `ArtifactError`
instead of being silently imputed. Pass `predict(rows, strict=False)` to impute it instead.

---

## Security

Artifacts are joblib (pickle) files, and **loading one can run code**. Only call
`ModelArtifact.load` on files you created or got from a source you trust.

`mlverdict serve` is a development server: it binds to `127.0.0.1` by default, limits request size,
and has no authentication or TLS. Put it behind a real gateway before exposing it to a network.

---

## Development

```bash
pip install -e ".[dev,all]"
pytest
ruff check src tests
```

CI runs the tests on Python 3.10–3.13 for every push and pull request. See [CHANGELOG.md](CHANGELOG.md).

---

## License

MIT. See [LICENSE](LICENSE).
