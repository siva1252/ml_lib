# MLVerdict

You have a spreadsheet or CSV. One column is the answer you want to predict
(price, yes/no, churn, disease, …). You do **not** want to spend days writing
scikit-learn: which problem is this, which metric, which models, how to split,
how to explain the winner, how to save it.

**MLVerdict does that loop in a few lines.** You get a written verdict
(“this model, for this reason”) and a file you can use on new rows.

```bash
pip install mlverdict
```

No GitHub clone. No extra files from us. Use **your** data.

[PyPI](https://pypi.org/project/mlverdict/) · [Source](https://github.com/siva1252/ml_lib)

---

## What you get (why use this)

Without this library you typically:

- decide classification vs regression yourself
- pick a metric (accuracy is often misleading)
- train models one by one
- compare notebooks by hand
- hope you did not leak an ID column
- rewrite preprocessing when you deploy

With MLVerdict you still bring the **CSV + the name of the target column**.
The library then:

1. Reads the table
2. Figures out the problem type (yes/no, many classes, or a number)
3. Picks a metric that matches that problem
4. Trains a small set of standard models
5. Optionally tunes the best ones
6. Picks a winner with reasons (not a silent guess)
7. Warns you about traps (IDs, imbalance, “accuracy 80% but never finds the rare class”)
8. Saves one artifact: cleaning + model together

**Work saved:** the usual “first serious sklearn project” (days of glue code)
becomes minutes of `fit` + `print`. You still need decent data. The library
does not invent labels or magically fix a useless table.

---

## This is supervised only

| You have | MLVerdict |
| --- | --- |
| A target column (the thing to predict) | **Yes — this is the product** |
| No target (only clustering / “find groups” / anomaly with no labels) | **No.** That is unsupervised. This version does not do it. |

If you are not sure: look at your table. If one column is the answer you want
in the future (`churn`, `price`, `label`, …), you are in the right place.
Pass **that column’s real name** to `fit`.

---

## Use it (your file, your column)

`customers.csv` and `"churn"` below are **examples**.  
Replace them with your path and your target name. We do not ship a CSV.

```python
from mlverdict import Verdict

run = Verdict(enable_hpo=True).fit("customers.csv", "churn")
print(run)
```

| Piece | Meaning |
| --- | --- |
| `"customers.csv"` | **Your** file. Excel → Save as CSV. Or a pandas DataFrame. |
| `"churn"` | **Your** target column. If the column is `price`, write `"price"`. |
| `print(run)` | Plain-language result: winner, scores, warnings, what to do next |

That is enough for a first run. You do not pick algorithms. You do not clone
the repo.

House prices:

```python
run = Verdict().fit("houses.csv", "sale_price")
print(run)
```

### Save the model for later

```python
run.artifact().save("model.joblib")

from mlverdict import ModelArtifact
model = ModelArtifact.load("model.joblib")
model.predict(new_rows)   # new_rows = table without the target column
```

---

## What the library tries (you do not choose)

Users should not have to know these names. They are listed so you know we are
not hiding a random mystery model.

For normal **tables** (rows and columns), these families are the usual first
try in industry. MLVerdict trains the ones that fit your problem and **picks
one with evidence**:

- linear models (Logistic Regression or Ridge)
- tree models (Random Forest, Extra Trees)
- sklearn gradient boosting (HistGradientBoosting)
- XGBoost / LightGBM / CatBoost **only if you installed them**

You do not import them. You do not tune them unless you turn HPO on
(`enable_hpo=True`). The printout tells you which one won and why.

---

## Optional: you have no CSV yet

Only then run the demo. It **creates a fake table** and runs the same API.
Normal users skip this.

```bash
pip install mlverdict
git clone https://github.com/siva1252/ml_lib.git
cd ml_lib
python examples/quickstart.py
```

`git clone` is **not** required to use the library. It is only for the demo
script, reading source, or running tests.

---

## Optional: developers / tests

```bash
git clone https://github.com/siva1252/ml_lib.git
cd ml_lib
pip install -e ".[dev]"
pytest
```

---

## License

MIT. See [LICENSE](LICENSE).
