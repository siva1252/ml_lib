# Changelog

All notable changes to MLVerdict. Versions follow [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-25

An independent audit ran every model on real and synthetic data, plus edge cases, the
command-line tool and the prediction server. This release fixes everything it found.

### Fixed
- **Binary metrics with text labels.** With labels like `"yes"/"no"`, PR-AUC, F1, precision and
  recall came back as NaN for every model, so the winner was picked without a real score.
  Binary metrics now always score an explicit positive class.
- **XGBoost classification.** It failed on every dataset with text labels (it only accepts
  `0..k-1`). All classifiers are now trained on encoded labels and return the original labels.
- **Models with no finite score** are now marked failed and can no longer be ranked or win.
- **Crash when every tuning trial failed** (`ValueError: No trials are completed yet`). The run now
  falls back to default parameters.
- **One `inf` value blocked the whole run.** Infinities are now treated as missing and imputed
  inside the pipeline.
- **Leakage detection** missed a numeric column that copies a text target. It now uses
  single-feature separation (binary) and the correlation ratio (multiclass).
- **Leakage false alarms.** `days_since_last_login` was flagged high-risk, and a one-letter target
  such as `y` matched any column containing that letter. Target names now match whole words only,
  and post-outcome name patterns are warnings unless they name the target.
- **Silent garbage at prediction time.** `"abc"` in a numeric column was imputed without notice.
  `predict()` now raises `ArtifactError` (opt out with `strict=False`).
- **Readiness checks that always passed.** Four checks were hard-coded to pass. They now probe the
  pipeline with an all-missing row and an unseen category, and verify the recorded environment
  and latency.
- **Leaderboard exaggeration.** Min-max scaling turned a 0.004 score gap into a 0-vs-1 swing.
  Gaps smaller than cross-validation noise are no longer treated as wins.
- **PCA kept nearly every component** (it maximized explained variance). It now keeps the fewest
  components that reach `pca_variance_threshold` (default 0.95).
- **Anomaly detection flagged about 29% of rows** (it maximized score spread). Every detector now
  flags the same `anomaly_contamination` share (default 5%).
- **Clustering picked k by luck** (3 random trials) and silhouette was O(n²) (140 s on 30k rows).
  Every k from 2 to `max_clusters` is now tried, with k chosen on a sample on large data and
  silhouette scored on a sample. 30k rows now take about 70 s and find the true k.

- **Robustness pass:**
  - A column mixing types (`1, "a", 2.5, True`) made every model fail; such columns are now
    treated as categories.
  - Integer column names, as from `pd.DataFrame(numpy_array)`, crashed the run; column names
    are now handled as strings in `fit` and `predict`.
  - Duplicate column names crashed with a `TypeError`; they now raise a clear `DataLoadError`.
  - `predict()` on an empty table crashed; it now returns an empty result.

### Added
- Models: Linear Regression, Lasso, ElasticNet, Decision Tree, K-Nearest Neighbors, Support Vector
  Machine (classification and regression), Gaussian Naive Bayes.
- **Detector Ensemble** (Isolation Forest + One-Class SVM + LOF, max-vote), the default for anomaly
  detection. It stayed close to the best single detector on every outlier shape tested, where each
  single detector fails badly on at least one.
- `fit(..., positive_label=...)` and `mlverdict fit --positive-label`.
- `mlverdict fit --contamination`.
- Config: `anomaly_contamination`, `pca_variance_threshold`, `max_clusters`, `cluster_scan_rows`,
  `silhouette_sample_size`, `max_rows_for_svm`, `max_rows_for_knn`.
- Integer class codes `0..k-1` (iris, wine) are detected as multiclass, with a warning.
  Ambiguous float scales such as a `1.0–5.0` rating still stop as `UNDECIDED`.
- Artifact metadata records the class order, positive label, and library versions.
- `ModelArtifact.classes_`.
- Prediction server: request size limit (10 MB), record limit, and 400/413/422 responses for bad
  input.
- GitHub Actions CI on Python 3.10–3.13, plus 31 regression tests for the fixes above.

### Changed
- The default anomaly metric is `anomaly_separation` (`decision_std` is still reported).
- SVM and One-Class SVM are skipped above `max_rows_for_svm` rows (default 10,000), and KNN above
  `max_rows_for_knn` rows (default 50,000).
- Models with nothing to tune (Linear Regression) no longer spend tuning budget.

## [0.2.1] - 2026-09-07

- Unsupervised clustering, anomaly detection and PCA on the same Phase 1 path as supervised.

## [0.1.0] - 2026-09-07

- First release: supervised classification and regression decision engine.
