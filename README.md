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
├── exp1_cosine_distance/models/     # Cosine dissimilarity neighbourhood
├── exp2_attention_pooling/models/   # Soft attention pooling
├── exp3_pca_alignment/models/       # PCA local frame alignment
├── exp4_ann/models/                 # Approximate nearest neighbour
├── exp5_combined/models/            # All SSG modifications combined
├── exp6_msg/models/                 # Multi-Scale Grouping
├── exp7_msg_normals/models/         # MSG + Surface Normals (best)
├── exp8_shared_ann_msg/models/      # Shared-computation ANN in MSG
│
├── scripts/
│   ├── generate_plots.py            # Accuracy bar chart, training curves, heatmap
│   └── generate_multiview_viz.py    # Per-object multi-view inference visualisation
│
└── results/
    ├── plot_accuracy_comparison.png
    ├── plot_training_curves.png
    ├── plot_heatmap.png
    ├── plot_inference_summary.png
    └── viz_*.png
```

> **Not included (too large for GitHub):**
> `data/` — ModelNet40 dataset (~350 MB, download separately)
> `log/checkpoints/` — trained model weights (~21 MB each)

---

## Dataset

**ModelNet40** — 40 rigid-object categories, 9,843 train / 2,468 test, 1,024 points per object.

```bash
# Inside any experiment directory, this downloads and preprocesses automatically:
python train_classification.py --process_data
```

---

## Setup

```bash
pip install torch numpy matplotlib tqdm Pillow

# Clone the base PointNet++ repo
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch

# Replace models/ with the files from the relevant experiment directory
cp exp1_cosine_distance/models/* Pointnet_Pointnet2_pytorch/models/
```

---

## Running Experiments

All experiments use the same training script from the base repo. Run from inside each experiment directory.

### Baseline
```bash
cd baseline/
python train_classification.py --model pointnet2_cls_ssg --num_category 40 \
    --epoch 200 --batch_size 24 --learning_rate 0.001 --log_dir baseline_run
```

### Exp 1 — Cosine Distance
```bash
cd exp1_cosine_distance/
python train_classification.py --model pointnet2_cls_ssg --num_category 40 \
    --epoch 200 --batch_size 24 --log_dir exp1_cosine_run
```

### Exp 2 — Attention Pooling
```bash
cd exp2_attention_pooling/
python train_classification.py --model pointnet2_cls_ssg --num_category 40 \
    --epoch 200 --batch_size 24 --log_dir exp2_attention_run
```

### Exp 3 — PCA Alignment
```bash
cd exp3_pca_alignment/
python train_classification.py --model pointnet2_cls_ssg --num_category 40 \
    --epoch 200 --batch_size 24 --log_dir exp3_pca_run
```

### Exp 4 — ANN Search
```bash
cd exp4_ann/
python train_classification.py --model pointnet2_cls_ssg --num_category 40 \
    --epoch 200 --batch_size 24 --log_dir exp4_ann_run
```

### Exp 5 — All Combined
```bash
cd exp5_combined/
python train_classification.py --model pointnet2_cls_ssg --num_category 40 \
    --epoch 200 --batch_size 24 --log_dir exp5_combined_run
```

### Exp 6 — MSG
```bash
cd exp6_msg/
python train_classification.py --model pointnet2_cls_msg --num_category 40 \
    --epoch 200 --batch_size 24 --log_dir exp6_msg_run
```

### Exp 7 — MSG + Surface Normals
```bash
cd exp7_msg_normals/
python train_classification.py --model pointnet2_cls_msg --num_category 40 \
    --epoch 200 --batch_size 24 --use_normals --log_dir exp7_msg_normals_run
```

### Exp 8 — Shared-Computation ANN in MSG
```bash
cd exp8_shared_ann_msg/
python train_classification.py --model pointnet2_cls_msg --num_category 40 \
    --epoch 200 --batch_size 24 --log_dir exp8_shared_ann_run
```

---

## Experiment Descriptions

### Baseline — PointNet++ SSG
Unmodified PointNet++ Single-Scale Grouping. Three Set Abstraction layers (FPS → Ball Query → PointNet MLP → Max-Pool), followed by FC(512) → FC(256) → FC(40). Adam optimiser, lr=0.001, cosine LR decay, 200 epochs.

### Exp 1 — Cosine Distance
**File:** `models/pointnet2_utils.py` → `square_distance()`

Replaces Euclidean squared distance with cosine dissimilarity (`1 − cosine_similarity`) in the ball-query neighbourhood search. Tests whether angle-based similarity better captures local surface orientation.

**Result:** −0.72 pp. Cosine metric is scale-invariant and breaks the spatial locality that PointNet++ depends on. FPS centroids are placed by Euclidean geometry; a cosine neighbourhood is architecturally inconsistent.

### Exp 2 — Soft Attention Pooling
**File:** `models/pointnet2_utils.py` → `PointNetSetAbstraction.forward()`

Replaces max-pooling with soft attention — a scalar weight per neighbour as the softmax of the mean feature value across channels, then a weighted sum.

**Result:** −0.76 pp. Mean-channel attention collapses all channels into one scalar per neighbour. Max-pooling independently tracks the strongest activation per channel, which is more discriminative. A learned per-channel 1D conv (C→1) would be needed to compete.

### Exp 3 — PCA Local Frame Alignment
**File:** `models/pointnet2_utils.py` → `sample_and_group()`

Before the MLP, each local patch is rotated into its PCA frame via `torch.linalg.eigh` on the neighbourhood covariance matrix. Enforces a canonical local coordinate frame.

**Result:** −0.64 pp. ModelNet40 objects are pre-aligned, so canonical framing adds little value. Thin or planar patches produce near-degenerate covariance matrices with numerically unstable eigenvectors.

### Exp 4 — Approximate Nearest Neighbour (ANN)
**File:** `models/pointnet2_utils.py` → `query_ball_point()`

Replaces exact ball query with random ANN: sample 8× the required neighbours uniformly at random, compute distances only within this pool, select K nearest. Reduces per-layer distance cost from O(N·S) to O(8K·S) with no external library.

**Result:** −0.19 pp — smallest degradation of any SSG modification. Near-lossless approximation, best practical trade-off between accuracy and inference speed.

### Exp 5 — All SSG Modifications Combined
**File:** `models/pointnet2_utils.py` — all four SSG modifications applied simultaneously.

**Result:** −1.90 pp — worst result overall. Modifications interfere: cosine distance alters which candidates enter the ANN pool; PCA on those patches is noisier still. Three noise sources compound rather than cancel.

### Exp 6 — Multi-Scale Grouping (MSG)
**Change:** switched to `pointnet2_cls_msg.py` — no other modifications.

Queries three radii per SA layer and concatenates features from all scales, capturing fine and coarse geometry simultaneously.

**Result:** +0.09 pp. Consistent but marginal gain on ModelNet40. Larger gains expected on complex or real-scanned datasets.

### Exp 7 — MSG + Surface Normals *(Best)*
**File:** `models/pointnet2_cls_msg.py` — first SA layer `in_channel` updated 3→6.

Extends MSG with per-point surface normals: input becomes (x, y, z, nₓ, nᵧ, n_z). Normals encode local surface orientation, disambiguating categories with similar coarse geometry but different curvature.

**Result:** +0.38 pp → **92.94%**. Best configuration. MSG multi-scale features and normal orientation cues are mutually reinforcing.

### Exp 8 — Shared-Computation ANN in MSG
**File:** `models/pointnet2_utils.py` → `PointNetSetAbstractionMsg.forward()`

Direct implementation of the PointNet++ paper's stated future work: query the largest MSG radius once via ANN, compute distances once, filter to smaller radii from that superset — eliminating 3× redundant distance computation.

**Result:** −0.75 pp. Architecturally correct but the shared pool is sized for the largest radius. For the smallest MSG radius, few pool candidates fall within the ball, producing sparse small-scale neighbourhoods. Fix: radius-proportional pool sizes.

---

## Visualisations

### Accuracy Comparison
![Accuracy Comparison](results/plot_accuracy_comparison.png)

### Training Curves
![Training Curves](results/plot_training_curves.png)

### Accuracy Heatmap
![Heatmap](results/plot_heatmap.png)

### Test Set Inference
![Inference Summary](results/plot_inference_summary.png)

---

## Key Findings

- **Architecture beats metric tuning** — MSG gave +0.09 pp with zero MLP changes; adding normals gave +0.29 pp more. All metric/pooling/alignment modifications hurt accuracy.
- **ANN is near-lossless** — −0.19 pp for an 8× pool. Best practical modification for inference speed.
- **Attention needs to be learned** — handcrafted mean-channel attention is worse than max-pooling.
- **Do not stack weak modifications** — Exp 5 dropped −1.90 pp, far worse than any individual change.
- **Shared ANN is the right idea** — Exp 8 correctly implements the paper's future work; fix is radius-proportional pool sizing.

---

## Future Work

- Learned per-channel attention (fix for Exp 2) — 1D Conv C→1, sigmoid
- Robust PCA with covariance regularisation + SVD sign disambiguation (fix for Exp 3)
- Radius-proportional ANN pools or FAISS IVF (fix for Exp 8)
- Combine ANN (Exp 4) with MSG+Normals (Exp 7) for peak accuracy with reduced inference cost
- Evaluate on ScanObjectNN and ShapeNet Part
- Knowledge distillation from Exp 7 into a lightweight SSG+ANN student

---

## References

1. Qi et al. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR 2017.
2. Qi et al. *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.* NeurIPS 2017.
3. Wang et al. *Dynamic Graph CNN for Learning on Point Clouds.* ACM ToG 2019.
4. Thomas et al. *KPConv: Flexible and Deformable Convolution for Point Clouds.* ICCV 2019.
5. Zhao et al. *Point Transformer.* ICCV 2021.

---

## Compute

Trained on **ASU Sol Supercomputer** using **NVIDIA A100 80 GB** GPUs. PyTorch 2.8, Python 3.12. Each experiment trained for 200 epochs (~2–3 hours per run).
