"""
pipeline.py
-----------
End-to-end cancer-subtype classification pipeline from gene expression data:

  1. Load + normalize expression matrix (samples x genes, with a class column)
  2. Differential expression: rank genes by two-sample t-test between classes
  3. Dimensionality reduction: PCA (+ optional t-SNE) to visualize class separation
  4. Unsupervised check: hierarchical clustering heatmap of top DE genes
  5. Supervised classification: Logistic Regression / Random Forest / SVM,
     compared with cross-validation, using only the top-N most differential
     genes as features (standard practice for p >> n microarray data)

Usage:
    python src/pipeline.py --data data/demo_expression.csv
    python src/pipeline.py --data data/real_expression.csv --top-genes 50
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def differential_expression(expr: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    """Two-sample t-test per gene between the two classes; returns genes
    ranked by |t-statistic| (a simple, classic, and still widely-used
    approach for microarray/RNA-seq biomarker discovery)."""
    groups = classes.unique()
    assert len(groups) == 2, "This pipeline currently supports binary classification"
    g1 = expr[classes == groups[0]]
    g2 = expr[classes == groups[1]]

    t_stats, p_vals = stats.ttest_ind(g1, g2, axis=0, equal_var=False)
    de = pd.DataFrame({
        "gene": expr.columns,
        "t_stat": t_stats,
        "p_value": p_vals,
        "mean_" + groups[0]: g1.mean().values,
        "mean_" + groups[1]: g2.mean().values,
    })
    de["abs_t"] = de.t_stat.abs()
    de = de.sort_values("abs_t", ascending=False).reset_index(drop=True)
    return de


def plot_pca_tsne(X_scaled, y, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    for label in np.unique(y):
        mask = y == label
        axes[0].scatter(pcs[mask, 0], pcs[mask, 1], label=label, alpha=0.7, s=40)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    axes[0].set_title("PCA")
    axes[0].legend()

    perplexity = min(30, max(5, X_scaled.shape[0] // 4))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
    ts = tsne.fit_transform(X_scaled)
    for label in np.unique(y):
        mask = y == label
        axes[1].scatter(ts[mask, 0], ts[mask, 1], label=label, alpha=0.7, s=40)
    axes[1].set_title("t-SNE")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / "pca_tsne.png", dpi=150)
    print(f"Saved {results_dir / 'pca_tsne.png'}")


def plot_heatmap(expr, classes, top_genes, results_dir):
    sub = expr[top_genes[:30]]  # cap for a readable heatmap
    sub_z = (sub - sub.mean()) / sub.std()

    row_colors = classes.map({classes.unique()[0]: "#e08214", classes.unique()[1]: "#4393c3"})
    g = sns.clustermap(sub_z.T, cmap="vlag", col_colors=row_colors,
                        figsize=(12, 8), xticklabels=False, cbar_pos=(0.02, 0.83, 0.03, 0.15))
    g.ax_heatmap.set_ylabel("Top differentially expressed genes")
    g.ax_heatmap.set_xlabel("Samples")
    g.savefig(results_dir / "de_gene_heatmap.png", dpi=150)
    print(f"Saved {results_dir / 'de_gene_heatmap.png'}")


def run_classification(X, y, results_dir):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, stratify=y_enc, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "SVM (linear kernel)": SVC(kernel="linear", probability=True, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fig, ax = plt.subplots(figsize=(6, 5))
    rows = []
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train_s, y_train)
        y_proba = model.predict_proba(X_test_s)[:, 1]
        y_pred = model.predict(X_test_s)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
        rows.append({
            "model": name, "cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std(),
            "test_auc": roc_auc, "test_accuracy": report["accuracy"],
        })
        print(f"{name:22s} CV AUC={cv_scores.mean():.3f}±{cv_scores.std():.3f}  "
              f"Test AUC={roc_auc:.3f}  Test Acc={report['accuracy']:.3f}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Classification ROC ({le.classes_[0]} vs {le.classes_[1]})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(results_dir / "classification_roc.png", dpi=150)
    print(f"Saved {results_dir / 'classification_roc.png'}")

    pd.DataFrame(rows).to_csv(results_dir / "classification_results.csv", index=False)
    return pd.DataFrame(rows)


def run(data_path: str, top_genes_n: int):
    df = pd.read_csv(data_path, index_col=0)
    classes = df["class"]
    expr = df.drop(columns=["class"])
    print(f"Loaded {expr.shape[0]} samples x {expr.shape[1]} genes "
          f"({classes.value_counts().to_dict()})")

    print("\nRunning differential expression analysis...")
    de = differential_expression(expr, classes)
    de.to_csv(RESULTS_DIR / "differential_expression.csv", index=False)
    print(f"Top 10 differentially expressed genes:\n{de.head(10)[['gene','t_stat','p_value']]}")

    top_genes = de.gene.tolist()[:top_genes_n]
    X = expr[top_genes].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nGenerating PCA / t-SNE plots...")
    plot_pca_tsne(X_scaled, classes.values, RESULTS_DIR)

    print("Generating clustered heatmap of top DE genes...")
    plot_heatmap(expr, classes, top_genes, RESULTS_DIR)

    print("\nRunning classification models...")
    results = run_classification(X, classes.values, RESULTS_DIR)
    print(f"\nBest model: {results.sort_values('test_auc', ascending=False).iloc[0]['model']}")
    return de, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data" / "demo_expression.csv"))
    parser.add_argument("--top-genes", type=int, default=20, help="Number of top DE genes to use as classifier features")
    args = parser.parse_args()
    run(args.data, args.top_genes)
