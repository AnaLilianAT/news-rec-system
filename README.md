# Impact of Feature Embeddings on the Relevance–Diversity Trade-off in News Recommendation

This repository contains the code and experiment pipeline used in the paper:

> **A Controlled Study of Relevance--Diversity Trade-offs in News Recommendation with Embeddings**  
> Authors: **Anonymous for double-blind review**  
> Venue: **CSBC/SEMISH**, **2026**

We study how replacing a sparse binary news-feature representation with low-dimensional embeddings affects the relevance–diversity trade-off in a news recommendation pipeline. Experiments are conducted using collaborative filtering recommenders and post-filtering diversification (e.g., MMR / topic diversification), evaluated with ranking relevance and homogeneity/diversity metrics.

## Repository Contents

- Offline evaluation pipeline (temporal replay)
- Baseline: binary feature matrix
- Embeddings: TruncatedSVD computed from the same feature matrix
- Recommenders: user-based kNN, item-based kNN, matrix factorization baseline (SVD)
- Diversification: Maximal Marginal Relevance (MMR)
- Metrics: NDCG@10 and intra-list similarity

# NDCG vs GH Trade-off Analysis Pipeline

This section explains how to reproduce the experimental results reported in the paper, including the complete pipeline used to analyze the trade-off between accuracy (NDCG) and diversity (ILS) across embedding dimensions.

Before running these commands, make sure you have installed all dependencies listed in `requirements.txt`.

The pipeline consists of **3 sequential steps**:

2. **Dimension × Seed Sweep**: Train embeddings and evaluate across multiple dimensions and seeds
3. **Results Aggregation**: Calculate statistics (mean, std, CI95) per (dimension, algorithm)
4. **Trade-off Analysis**: Find optimal dimension for different NDCG-ILS weights

### Step 1: Embedding Dimension Sweep

```bash
python -m src.experiments.run_embedding_dim_seed_sweep \
    --step 1 \
    --n-seeds 20 \
    --embedding-method svd \
    --ranking-metric ndcg \
    --ndcg-cutoff 10 \
    --cleanup-intermediate \
```
**Outputs:**
- `outputs/experiments/embedding_dim_seed_sweep_runs.parquet`
- `outputs/embeddings/svd_features_d*_seed*.json` (cached embeddings)

### Step 2: Aggregate Results

```bash
python -m src.experiments.aggregate_embedding_dim_seed_sweep \
```
**Outputs:**
- `outputs/experiments/embedding_dim_seed_sweep_agg.parquet`

### Step 3: Trade-off Analysis

```bash
python -m src.experiments.analyze_ndcg_gh_tradeoff \
    --ndcg-cutoff 10
```
**Outputs:**
- `outputs/experiments/tradeoff_all_algorithms.csv`
- `outputs/experiments/tradeoff_<algorithm>.csv`
