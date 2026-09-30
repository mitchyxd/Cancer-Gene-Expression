# Cancer Subtype Classification from Gene Expression Data

A full microarray/RNA-seq style analysis pipeline that classifies leukemia
subtype (ALL vs AML) directly from gene expression measurements — the same
task, and the same landmark dataset, that launched the field of molecular
cancer classification.

## Why this matters

Before this 1999 study, cancers were classified almost entirely by how
tissue looked under a microscope. Golub et al. showed that gene expression
profiles alone could tell apart cancer subtypes that look identical
morphologically — the foundational proof-of-concept behind everything from
modern tumor-profiling panels to Oncotype DX. This project reimplements
that class-discovery/class-prediction workflow end to end: differential
expression → dimensionality reduction → classification → biomarker
interpretation.

## Dataset

Real data: the **Golub et al. 1999 leukemia microarray dataset** (72
patients, ALL vs AML, ~7,129 genes measured on Affymetrix HU6800 arrays).
> Golub, T.R., Slonim, D.K., Tamayo, P. et al. "Molecular Classification of
> Cancer: Class Discovery and Class Prediction by Gene Expression
> Monitoring." *Science* 286(5439), 531–537 (1999).

```bash
pip install -r requirements.txt
# one-time: put your Kaggle API token at ~/.kaggle/kaggle.json
python src/download_real_data.py
python src/pipeline.py --data data/real_expression.csv --top-genes 50
```

**No Kaggle account handy?** A small synthetic demo matrix (80 samples x
508 genes) lets you run the full pipeline immediately:

```bash
python src/make_demo_data.py
python src/pipeline.py --data data/demo_expression.csv
```

The demo data is synthetic but *directionally realistic* — it simulates 8
named marker genes (MPO, CD33, ELANE, LYZ higher in AML; CD79A, CD19, TCF3,
ADA higher in ALL) at the expression levels/direction reported in the
literature, plus 500 uninformative "background" genes, so the differential
expression step has to actually find the signal rather than trivially
separating classes. See `src/make_demo_data.py` docstring for details —
this is clearly a toy fixture, not real patient data.

## Method

1. **Differential expression** (`differential_expression()`): two-sample
   t-test per gene, ranked by |t-statistic| — the classic Golub-paper-era
   approach, still a reasonable first pass on real data.
2. **Dimensionality reduction**: PCA and t-SNE on the top differentially
   expressed genes, to visualize whether the two cancer subtypes actually
   separate in expression space before trusting any classifier.
3. **Unsupervised sanity check**: hierarchical clustering heatmap — do
   samples cluster by their *known* subtype using only the top DE genes,
   with no label information given to the clustering itself?
4. **Supervised classification**: Logistic Regression, Random Forest, and
   linear-kernel SVM, evaluated with stratified 5-fold cross-validation
   plus a held-out test set (standard practice for p >> n genomic data:
   feature-select first, classify on the reduced set).

## Outputs (`results/`)

- `differential_expression.csv` — every gene ranked by significance
- `pca_tsne.png`, `de_gene_heatmap.png` — exploratory visualizations
- `classification_roc.png`, `classification_results.csv` — model comparison

## Possible extensions

- Multi-class version using the full TCGA Pan-Cancer RNA-Seq dataset
  (801 samples, 5 cancer types) — `differential_expression()` would need
  an ANOVA/Kruskal-Wallis generalization instead of a t-test
- Regularized feature selection (LASSO / elastic-net) instead of a fixed
  top-N t-test cutoff, to see how many genes are really needed
- Gene Set Enrichment Analysis (GSEA) on the DE gene list against MSigDB
  hallmark pathways, to connect the statistics back to biology
