"""
download_real_data.py
-----------------------
Fetches and reshapes the REAL dataset used in this project: the Golub et al.
1999 leukemia microarray data (ALL vs AML), mirrored on Kaggle as
"crawford/gene-expression".

Reference: Golub, T.R., Slonim, D.K., Tamayo, P. et al. "Molecular
Classification of Cancer: Class Discovery and Class Prediction by Gene
Expression Monitoring." Science 286(5439), 531-537 (1999).
https://doi.org/10.1126/science.286.5439.531

Setup (one-time):
    pip install kaggle
    # Get an API token from https://www.kaggle.com/settings -> "Create New Token"
    # and place the downloaded kaggle.json at ~/.kaggle/kaggle.json

Usage:
    python src/download_real_data.py

This downloads the raw Kaggle files into data/raw/, then reshapes them from
the wide "genes-as-rows" microarray format into the tidy "samples-as-rows,
genes-as-columns, class column" format that pipeline.py expects (matching
make_demo_data.py's output schema), and writes data/real_expression.csv.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


def download_from_kaggle():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading crawford/gene-expression from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "crawford/gene-expression",
         "-p", str(RAW_DIR), "--unzip"],
        check=True,
    )


def reshape_to_tidy():
    actual = pd.read_csv(RAW_DIR / "actual.csv")
    actual = actual.set_index("patient")

    train = pd.read_csv(RAW_DIR / "data_set_ALL_AML_train.csv")
    test = pd.read_csv(RAW_DIR / "data_set_ALL_AML_independent.csv")

    def clean(df):
        # Files interleave "call" columns (A/P/M present/absent calls) with
        # numeric expression columns per sample; keep only numeric sample columns.
        keep_cols = ["Gene Description", "Gene Accession Number"]
        sample_cols = [c for c in df.columns if c not in keep_cols and "call" not in c.lower()]
        return df[["Gene Accession Number"] + sample_cols].set_index("Gene Accession Number")

    train_expr = clean(train)
    test_expr = clean(test)

    full_expr = pd.concat([train_expr, test_expr], axis=1)
    full_expr.columns = [int(c) for c in full_expr.columns]
    full_expr = full_expr.T.sort_index()  # samples as rows, genes as columns

    full_expr["class"] = actual.loc[full_expr.index, "cancer"].values
    cols = ["class"] + [c for c in full_expr.columns if c != "class"]
    full_expr = full_expr[cols]

    out_path = DATA_DIR / "real_expression.csv"
    full_expr.to_csv(out_path)
    print(f"Wrote {full_expr.shape[0]} samples x {full_expr.shape[1]-1} genes to {out_path}")


if __name__ == "__main__":
    try:
        download_from_kaggle()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Kaggle download failed ({e}). If you already have the raw CSVs, "
              f"place them in {RAW_DIR} and re-run with --skip-download.", file=sys.stderr)
        if "--skip-download" not in sys.argv:
            sys.exit(1)
    reshape_to_tidy()
