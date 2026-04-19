# Exploring Architectural Modifications to PointNet++

**CSE 507 · Arizona State University · April 2026**

A systematic empirical study of eight controlled architectural modifications applied to [PointNet++](https://arxiv.org/abs/1706.02413) for 3-D point cloud classification on the ModelNet40 benchmark.

---

## Overview

PointNet++ (Qi et al., NeurIPS 2017) is a hierarchical deep learning framework for 3-D point cloud analysis. This project isolates and evaluates eight independent modifications to the original architecture — each experiment changes exactly one thing from the baseline — enabling direct attribution of any accuracy gain or loss.

**Baseline:** PointNet++ SSG on ModelNet40 — **92.56% instance accuracy**  
**Best result:** Exp 7 (MSG + Surface Normals) — **92.94% (+0.38 pp)**

---

## Results

| Experiment | Architecture | Instance Acc. | Class Acc. | Δ vs Baseline |
|---|---|---|---|---|
| Baseline | SSG | 92.56% | 90.03% | — |
| Exp 1: Cosine Distance | SSG | 91.84% | 89.23% | −0.72 pp |
| Exp 2: Attention Pooling | SSG | 91.80% | 88.09% | −0.76 pp |
| Exp 3: PCA Alignment | SSG | 91.92% | 88.65% | −0.64 pp |
| Exp 4: ANN Search | SSG | 92.37% | 89.85% | −0.19 pp |
| Exp 5: All Combined | SSG | 90.66% | 87.72% | −1.90 pp |
| Exp 6: MSG | MSG | 92.65% | 89.83% | +0.09 pp |
| **Exp 7: MSG + Normals** | MSG | **92.94%** | **90.45%** | **+0.38 pp** |
| Exp 8: Shared-Comp. ANN | MSG | 91.81% | 89.34% | −0.75 pp |

---

## Repository Structure

```
pointnet_experiments/
│
├── baseline/                        # Unmodified PointNet++ SSG
│   └── models/
│       ├── pointnet2_cls_ssg.py
│       ├── pointnet2_cls_msg.py
│       └── pointnet2_utils.py
│
├── exp1_cosine_distance/            # Exp 1: Cosine dissimilarity neighbourhood
│   └── models/
│       ├── pointnet2_cls_ssg.py
│       ├── pointnet2_cls_msg.py
│       └── pointnet2_utils.py      ← modified
│
├── exp2_attention_pooling/          # Exp 2: Soft attention pooling
│   └── models/
│       └── pointnet2_utils.py      ← modified
│
├── exp3_pca_alignment/              # Exp 3: PCA local frame alignment
│   └── models/
│       └── pointnet2_utils.py      ← modified
│
├── exp4_ann/                        # Exp 4: Approximate nearest neighbour
│   └── models/
│       └── pointnet2_utils.py      ← modified
│
├── exp5_combined/                   # Exp 5: All SSG modifications combined
│   └── models/
│       └── pointnet2_utils.py      ← modified
│
├── exp6_msg/                        # Exp 6: Multi-Scale Grouping (no code change)
│   └── models/
│       └── pointnet2_utils.py
│
├── exp7_msg_normals/                # Exp 7: MSG + Surface Normals
│   └── models/
│       ├── pointnet2_cls_msg.py    ← in_channel: 3 → 6
│       └── pointnet2_utils.py
│
├── exp8_shared_ann_msg/             # Exp 8: Shared-computation ANN in MSG
│   └── models/
│       └── pointnet2_utils.py      ← modified
│
├── scripts/
│   ├── generate_plots.py           # Accuracy bar chart, training curves, heatmap
│   ├── generate_multiview_viz.py   # Per-object multi-view inference visualisation
│   └── generate_inference_viz.py   # All-experiment inference grid
│
├── results/
│   ├── plot_accuracy_comparison.png
│   ├── plot_training_curves.png
│   ├── plot_heatmap.png
│   ├── plot_inference_summary.png
│   └── viz_*.png                   # Per-object multi-view inference plots
│
└── README.md
```

> **Not included in this repo (too large):**  
> `data/` — ModelNet40 dataset (~350 MB, download separately, see below)  
> `log/checkpoints/` — trained model weights (~21 MB each)

---

## Dataset

**ModelNet40** — 40 rigid-object categories, 9,843 train / 2,468 test samples, 1,024 points per object.

Download:
```bash
# Inside each experiment directory
python train_classification.py --process_data   # downloads and preprocesses automatically
# or manually from: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
```

---

## Setup

```bash
# Python environment (tested with PyTorch 2.8, Python 3.12)
pip install torch numpy matplotlib tqdm Pillow

# Clone the base PointNet++ repo into each experiment directory
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch
```

Replace the `models/pointnet2_utils.py` (and other modified files) in the base repo with the versions from the relevant experiment directory.

---

## Running Experiments

All experiments use the same training script. Run from inside each experiment directory.

### Baseline
```bash
cd baseline/
python train_classification.py \
    --model pointnet2_cls_ssg \
    --num_category 40 \
    --epoch 200 \
    --batch_size 24 \
    --learning_rate 0.001 \
    --log_dir baseline_run
```

### Exp 1 — Cosine Distance
```bash
cd exp1_cosine_distance/
python train_classification.py \
    --model pointnet2_cls_ssg \
    --num_category 40 \
    --epoch 200 \
    --batch_size 24 \
    --log_dir exp1_cosine_run
```

### Exp 2 — Attention Pooling
```bash
cd exp2_attention_pooling/
python train_classification.py \
    --model pointnet2_cls_ssg \
    --num_category 40 \
    --epoch 200 \
    --batch_size 24 \
    --log_dir exp2_attention_run
```

### Exp 3 — PCA Alignment
```bash
cd exp3_pca_alignment/
python train_classification.py \
    --model pointnet2_cls_ssg \
    --num_category 40 \
    --epoch 200 \
    --batch_size 24 \
    --log_dir exp3_pca_run
```

### Exp 4 — ANN Search
```bash
cd exp4_ann/
python train_classification.py \
    --model pointnet2_cls_ssg \
    --num_category 40 \
    --epoch 200 \
    --batch_size 24 \
    --log_dir exp4_ann_run
```

### Exp 5 — All Combined
```bash
cd exp5_combined/
python train_classification.py \
    --model pointnet2_cls_ssg \
    --num_category 40 \
    --epoch 200 \
    --batch_size 24 \
    --log_dir exp5_combined_run
```

### Exp 6 — MSG
```bash
cd exp6_msg/
python train_classification.py \
    --model pointnet2_cls_msg \
    --num_category 40 \
    --epoch 200 \
    --batch_size 24 \
    --log_dir exp6_msg_run
```

### Exp 7 — MSG + Surface Normals
```bash
cd exp7_msg_normals/
python train_classification.py \
    --model pointnet2_cls_msg \
    --num_category 40 \
    --epoch 200 \
    --batch_size 24 \
    --use_normals \
    --log_dir exp7_msg_normals_run
```

### Exp 8 — Shared-Computation ANN in MSG
```bash
cd exp8_shared_ann_msg/
python train_classification.py \
    --model pointnet2_cls_msg \
    --num_category 40 \
    --epoch 200 \
    --batch_size 24 \
    --log_dir exp8_shared_ann_run
```

---

## Experiment Descriptions

### Baseline — PointNet++ SSG
Unmodified PointNet++ Single-Scale Grouping. Three Set Abstraction layers (FPS → Ball Query → PointNet MLP → Max-Pool), followed by FC(512) → FC(256) → FC(40). Trained with Adam, lr=0.001, cosine LR decay, 200 epochs.

### Exp 1 — Cosine Distance
**File changed:** `models/pointnet2_utils.py` → `square_distance()`

Replaces the Euclidean squared distance in the ball-query neighbourhood search with cosine dissimilarity (`1 − cosine_similarity`). Tests whether angle-based similarity better captures local surface orientation than scale-sensitive Euclidean distance.

**Result:** −0.72 pp. Cosine metric is scale-invariant and breaks the spatial locality that PointNet++ depends on. FPS centroids are placed by Euclidean geometry; pairing them with a cosine neighbourhood is architecturally inconsistent.

### Exp 2 — Soft Attention Pooling
**File changed:** `models/pointnet2_utils.py` → `PointNetSetAbstraction.forward()`

Replaces max-pooling with soft attention: computes a scalar weight per neighbour as the softmax of the mean feature value across channels, then takes a weighted sum. Tests whether differentiable aggregation preserves more geometric detail than the hard max operator.

**Result:** −0.76 pp. Mean-channel attention collapses all channels into one scalar per neighbour — too coarse. Max-pooling independently tracks the strongest activation in every channel, which is inherently more discriminative. A learned per-channel attention (1D Conv C→1) would be needed to compete.

### Exp 3 — PCA Local Frame Alignment
**File changed:** `models/pointnet2_utils.py` → `sample_and_group()`

Before the MLP, each local patch is rotated into its PCA frame: compute the covariance matrix of the neighbourhood, decompose with `torch.linalg.eigh`, and project points onto the eigenvectors. Enforces a canonical local coordinate frame to reduce within-patch pose variation.

**Result:** −0.64 pp. ModelNet40 objects are pre-aligned (upright orientation), so canonical framing adds little value. Thin or planar patches produce near-degenerate covariance matrices, causing numerically unstable eigenvectors that add noise rather than removing it.

### Exp 4 — Approximate Nearest Neighbour (ANN)
**File changed:** `models/pointnet2_utils.py` → `query_ball_point()`

Replaces exact ball query with a random ANN approximation: sample 8× the required neighbours uniformly at random, compute distances only within this candidate pool, and select the K nearest. Reduces per-layer distance computation from O(N·S) to O(8K·S) with no external library.

**Result:** −0.19 pp — the smallest degradation of any SSG modification. For 1,024-point clouds a random pool of 8K reliably contains the true K nearest neighbours, making this near-lossless. Best practical trade-off: minimal accuracy cost with significant potential inference speedup.

### Exp 5 — All SSG Modifications Combined
**File changed:** `models/pointnet2_utils.py` — all four SSG modifications applied simultaneously.

Tests whether the four SSG changes interact synergistically or destructively when applied together.

**Result:** −1.90 pp — the worst result overall. The modifications interfere: cosine distance changes which candidates enter the ANN pool; PCA on cosine-selected patches is noisier still. Three independent noise sources compound rather than cancel. The sum of individual drops (2.31 pp) is slightly larger than the actual (1.90 pp), suggesting minor non-linear relief but still far worse than any single modification.

### Exp 6 — Multi-Scale Grouping (MSG)
**No code changes** — architecture switched from `pointnet2_cls_ssg.py` to `pointnet2_cls_msg.py`.

MSG queries three radii per Set Abstraction layer (SA-1: [0.1, 0.2, 0.4]; SA-2: [0.2, 0.4, 0.8]) and concatenates the feature vectors from all scales. Captures fine-grained and coarse-grained geometry simultaneously.

**Result:** +0.09 pp. Consistent but marginal gain on ModelNet40, where rigid pre-aligned shapes are well-characterised at a single scale. Larger gains expected on complex or real-scanned datasets.

### Exp 7 — MSG + Surface Normals *(Best)*
**File changed:** `models/pointnet2_cls_msg.py` — first SA layer `in_channel` updated from 3 to 6.

Extends MSG by appending pre-computed surface normals to each point: input becomes (x, y, z, nₓ, nᵧ, n_z). Surface normals encode local surface orientation, disambiguating categories with similar coarse geometry but different curvature.

**Result:** +0.38 pp → **92.94%**. Best configuration overall. MSG multi-scale features and normal-derived orientation cues are mutually reinforcing. Largest improvements on geometrically ambiguous classes (sofa vs. chair, monitor vs. tv_stand).

### Exp 8 — Shared-Computation ANN in MSG
**File changed:** `models/pointnet2_utils.py` → `PointNetSetAbstractionMsg.forward()`

Direct implementation of the future-work suggestion in the PointNet++ paper: *"it is worthwhile to think about how to accelerate inference... by sharing more computation in each local region."* Queries the largest MSG radius once via ANN, computes all distances once against a shared candidate pool, then filters to smaller radii from that superset — eliminating the 3× redundant distance computation across MSG scales.

**Result:** −0.75 pp. The implementation is architecturally correct but the shared pool is sized for the largest radius. For the smallest MSG radius, few pool candidates fall within the ball, producing sparse, noisy small-scale neighbourhoods. A radius-proportional pool (e.g. 32× for the smallest radius) or a FAISS IVF index would resolve this.

---

## Visualisations

### Accuracy Comparison
![Accuracy Comparison](results/plot_accuracy_comparison.png)

### Training Curves
![Training Curves](results/plot_training_curves.png)

### Accuracy Heatmap
![Heatmap](results/plot_heatmap.png)

### Inference on Test Set
![Inference Summary](results/plot_inference_summary.png)

---

## Key Findings

- **Architecture beats metric tuning.** Switching SSG → MSG gave +0.09 pp with zero MLP changes. Adding normals gave a further +0.29 pp. In contrast, all metric/pooling/alignment modifications hurt accuracy.
- **Attention needs to be learned.** A handcrafted mean-channel attention is strictly worse than max-pooling. A proper learned 1D conv gate (C→1) is required.
- **ANN is near-lossless.** −0.19 pp for an 8× pool random ANN — the best practical modification for inference speed.
- **Do not stack weak modifications.** Exp 5 (all SSG combined) dropped −1.90 pp, much worse than any individual change.
- **Shared-computation ANN is the right idea but needs a better pool.** Correctly implements the paper's future work; fix is radius-proportional pool sizing.

---

## Future Work

- Replace mean-channel attention (Exp 2) with a learned 1D conv (C→1, sigmoid) for proper per-channel gating
- Stabilise PCA (Exp 3) with covariance regularisation and SVD sign disambiguation
- Use radius-proportional ANN pools (Exp 8) to fix small-scale neighbourhood sparsity
- Combine Exp 4 (ANN) with Exp 7 (MSG+Normals) for peak accuracy with reduced inference cost
- Evaluate on ScanObjectNN and ShapeNet Part for harder benchmarks
- Knowledge distillation: compress Exp 7 into a lightweight SSG+ANN student

---

## References

1. Qi et al. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR 2017.
2. Qi et al. *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.* NeurIPS 2017.
3. Wang et al. *Dynamic Graph CNN for Learning on Point Clouds.* ACM ToG 2019.
4. Thomas et al. *KPConv: Flexible and Deformable Convolution for Point Clouds.* ICCV 2019.
5. Zhao et al. *Point Transformer.* ICCV 2021.

---

## Compute

Trained on the **ASU Sol Supercomputer** using **NVIDIA A100 80 GB** GPUs. PyTorch 2.8, Python 3.12. Each experiment trained for 200 epochs (~5–6 hours per run).
