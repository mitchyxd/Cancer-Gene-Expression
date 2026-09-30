"""
make_demo_data.py
-------------------
Generates a SMALL, SYNTHETIC gene-expression matrix so the analysis pipeline
can be run and unit-tested offline in seconds. This is NOT real patient data
— see README.md for how to pull the real dataset (Golub et al. 1999 leukemia
microarray data, via Kaggle).

The synthetic data is built to be biologically directionally correct, not
just random: a handful of genes are simulated with the same up/down
direction reported in the literature for ALL (Acute Lymphoblastic Leukemia)
vs AML (Acute Myeloid Leukemia), so that downstream differential-expression
and classification steps behave the way they would on real data:

  - MPO (myeloperoxidase) and CD33  : markedly HIGHER in AML (myeloid lineage)
  - CD79A / "MB-1" and TCF3         : markedly HIGHER in ALL (lymphoid lineage)
  - ADA (adenosine deaminase)       : higher in ALL
  - a long tail of "housekeeping-like" genes with no real class difference,
    to make the classification task non-trivial (as it is on real data)

Reference: Golub, T.R. et al. "Molecular Classification of Cancer: Class
Discovery and Class Prediction by Gene Expression Monitoring." Science 286,
531–537 (1999).
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

N_SAMPLES_PER_CLASS = 40
N_BACKGROUND_GENES = 500

MARKER_GENES = {
    # gene: (mean_in_ALL, mean_in_AML, std)
    "CD79A_MB1": (9.5, 3.0, 1.2),
    "TCF3": (8.0, 4.5, 1.0),
    "ADA": (7.5, 4.0, 1.1),
    "CD19": (9.0, 2.5, 1.3),
    "MPO": (2.5, 9.5, 1.2),
    "CD33": (3.0, 8.5, 1.1),
    "ELANE": (2.0, 8.0, 1.3),
    "LYZ": (3.5, 9.0, 1.0),
}


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    n_total = N_SAMPLES_PER_CLASS * 2
    sample_ids = [f"sample_{i}" for i in range(n_total)]
    labels = ["ALL"] * N_SAMPLES_PER_CLASS + ["AML"] * N_SAMPLES_PER_CLASS

    data = {}
    for gene, (mean_all, mean_aml, std) in MARKER_GENES.items():
        vals_all = rng.normal(mean_all, std, N_SAMPLES_PER_CLASS)
        vals_aml = rng.normal(mean_aml, std, N_SAMPLES_PER_CLASS)
        data[gene] = np.concatenate([vals_all, vals_aml])

    for i in range(N_BACKGROUND_GENES):
        mean = rng.uniform(3, 9)
        std = rng.uniform(0.5, 1.5)
        data[f"GENE_{i:04d}"] = rng.normal(mean, std, n_total)

    expr = pd.DataFrame(data, index=sample_ids)
    expr = expr.clip(lower=0)  # expression can't be negative

    expr.insert(0, "class", labels)
    out_path = out_dir / "demo_expression.csv"
    expr.to_csv(out_path)
    print(f"Wrote {expr.shape[0]} samples x {expr.shape[1]-1} genes to {out_path}")


if __name__ == "__main__":
    main()
