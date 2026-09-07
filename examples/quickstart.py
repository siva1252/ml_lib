"""Optional demo if you have no CSV yet. Normal usage is: pip install mlverdict
and fit() on YOUR file + YOUR target column.

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlverdict import Verdict

OUT = Path(__file__).resolve().parent / "train_with_label.csv"


def make_churn_csv(path: Path, n: int = 400, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    tenure = rng.integers(1, 60, size=n)
    spend = rng.gamma(2.0, 40.0, size=n)
    city = rng.choice(["a", "b", "c"], size=n)
    logits = -1.3 + 0.9 * x1 - 0.35 * x2 - 0.02 * tenure
    prob = 1.0 / (1.0 + np.exp(-logits))
    churn = (rng.random(n) < np.clip(prob, 0.08, 0.4)).astype(int)
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "x1": x1,
            "x2": x2,
            "tenure": tenure,
            "spend": spend,
            "city": city,
            "churn": churn,
        }
    )
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    csv = make_churn_csv(OUT)
    print(f"Wrote {csv} ({csv.stat().st_size} bytes)")
    run = Verdict(enable_hpo=False).fit(str(csv), "churn")
    print(run)
    print("Programmatic summary:", run.summary())


if __name__ == "__main__":
    main()
