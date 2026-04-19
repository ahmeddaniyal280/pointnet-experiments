import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys
import os
import importlib

CLASSES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]

base_dir = os.path.expanduser('~/pointnet_experiments')
data_path = os.path.join(base_dir, 'Pointnet_Pointnet2_pytorch/data/modelnet/modelnet40_test_1024pts.dat')
with open(data_path, 'rb') as f:
    test_data = pickle.load(f)
points_list, labels_list = test_data[0], test_data[1]

EXPERIMENTS = [
    ('Exp1 Cosine',    'ssg', False, 'exp1_cosine_distance/log/classification/exp1_cosine_run/checkpoints/best_model.pth',      'exp1_cosine_distance/models/pointnet2_cls_ssg.py'),
    ('Exp2 Attention', 'ssg', False, 'exp2_attention_pooling/log/classification/exp2_attention_run/checkpoints/best_model.pth', 'exp2_attention_pooling/models/pointnet2_cls_ssg.py'),
    ('Exp3 PCA',       'ssg', False, 'exp3_pca_alignment/log/classification/exp3_pca_run/checkpoints/best_model.pth',           'exp3_pca_alignment/models/pointnet2_cls_ssg.py'),
    ('Exp4 ANN',       'ssg', False, 'exp4_ann/log/classification/exp4_ann_run/checkpoints/best_model.pth',                     'exp4_ann/models/pointnet2_cls_ssg.py'),
    ('Exp5 Combined',  'ssg', False, 'exp5_combined/log/classification/exp5_combined_run/checkpoints/best_model.pth',           'exp5_combined/models/pointnet2_cls_ssg.py'),
    ('Exp6 MSG',       'msg', False, 'exp6_msg/log/classification/exp6_msg_run/checkpoints/best_model.pth',                     'exp6_msg/models/pointnet2_cls_msg.py'),
    ('Exp7 MSG+Norm',  'msg', True,  'exp7_msg_normals/log/classification/exp7_msg_normals_run/checkpoints/best_model.pth',     'exp7_msg_normals/models/pointnet2_cls_msg.py'),
    ('Exp8 SharedANN', 'msg', False, 'exp8_shared_ann_msg/log/classification/exp8_shared_ann_run/checkpoints/best_model.pth',   'exp8_shared_ann_msg/models/pointnet2_cls_msg.py'),
]

def load_model(label, model_type, use_normals, ckpt_rel, model_rel):
    model_path = os.path.join(base_dir, model_rel)
    models_dir = os.path.dirname(model_path)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    spec = importlib.util.spec_from_file_location(f"cls_{label.replace(' ','_')}", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.get_model(40, normal_channel=use_normals)
    ckpt = torch.load(os.path.join(base_dir, ckpt_rel), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    sys.path.remove(models_dir)
    return model

print("Loading models...")
models = [load_model(*e) for e in EXPERIMENTS]
print("Done.\n")

def pc_normalize(pc):
    pc = pc - np.mean(pc, axis=0)
    return pc / (np.max(np.sqrt(np.sum(pc**2, axis=1))) + 1e-8)

def predict(model, pts, use_normals):
    cols = 6 if use_normals else 3
    p = torch.tensor(pts[:, :cols], dtype=torch.float32).unsqueeze(0).transpose(2, 1)
    with torch.no_grad():
        out, _ = model(p)
    return CLASSES[out.argmax(dim=1).item()]

# Select one sample per class
selected = ['airplane', 'chair', 'car', 'guitar', 'lamp', 'monitor',
            'toilet', 'vase', 'sofa', 'bottle', 'person', 'bed']
samples = {}
for pts, lbl in zip(points_list, labels_list):
    name = CLASSES[int(lbl)]
    if name in selected and name not in samples:
        samples[name] = (pts, int(lbl))
    if len(samples) == len(selected):
        break

# 3 views: (elev, azim)
VIEWS = [('Front', 20, -60), ('Top', 89, -90), ('Side', 0, 0)]

# ── One figure per object ────────────────────────────────────────────────────
print("Generating per-object figures...")
for cls_name, (pts, true_lbl) in samples.items():
    pc = pc_normalize(pts[:, :3])

    # Collect predictions
    preds = []
    for (exp_label, _, use_normals, *_), model in zip(EXPERIMENTS, models):
        pred = predict(model, pts, use_normals)
        preds.append((exp_label, pred, pred == cls_name))

    n_correct = sum(1 for _, _, c in preds if c)

    # 1 row: 3 view panels + 1 prediction panel
    fig = plt.figure(figsize=(16, 4), facecolor='black')
    fig.patch.set_facecolor('black')

    # -- 3 view subplots --
    for vi, (view_name, elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 4, vi + 1, projection='3d')
        ax.set_facecolor('black')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                   c='white', s=1.2, alpha=0.7, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.pane.set_facecolor('black'); ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_facecolor('black'); ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_facecolor('black'); ax.zaxis.pane.set_edgecolor('black')
        ax.grid(False)
        ax.set_title(view_name, color='gray', fontsize=9, pad=4)

    # -- Prediction text panel --
    ax_txt = fig.add_subplot(1, 4, 4)
    ax_txt.set_facecolor('black')
    ax_txt.axis('off')

    ax_txt.text(0.5, 1.0, f'True class: {cls_name}',
                transform=ax_txt.transAxes,
                ha='center', va='top', fontsize=11,
                color='white', fontweight='bold')
    ax_txt.text(0.5, 0.88, f'{n_correct}/8 models correct',
                transform=ax_txt.transAxes,
                ha='center', va='top', fontsize=9,
                color='#aaaaaa')

    for i, (exp_label, pred, correct) in enumerate(preds):
        y = 0.74 - i * 0.095
        tick = '✓' if correct else '✗'
        col = '#00e676' if correct else '#ff5252'
        ax_txt.text(0.05, y, f'{tick}  {exp_label}',
                    transform=ax_txt.transAxes,
                    ha='left', va='top', fontsize=8.5, color=col)
        ax_txt.text(0.98, y, pred,
                    transform=ax_txt.transAxes,
                    ha='right', va='top', fontsize=8.5,
                    color='white' if correct else '#ff5252',
                    style='italic')

    plt.suptitle(f'Point Cloud — {cls_name.upper()}',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.5)

    out = os.path.join(base_dir, f'viz_{cls_name}.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: viz_{cls_name}.png  ({n_correct}/8 correct)")

# ── Combined summary grid (all objects, front view only) ─────────────────────
print("\nGenerating combined summary grid...")
n = len(samples)
cols = 4
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4),
                          subplot_kw={'projection': '3d'},
                          facecolor='black')
axes = axes.flatten()

for idx, (cls_name, (pts, true_lbl)) in enumerate(samples.items()):
    pc = pc_normalize(pts[:, :3])
    preds = []
    for (_, _, use_normals, *_), model in zip(EXPERIMENTS, models):
        pred = predict(model, pts, use_normals)
        preds.append(pred == cls_name)
    n_correct = sum(preds)

    ax = axes[idx]
    ax.set_facecolor('black')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
               c='white', s=1.2, alpha=0.7, linewidths=0)
    ax.view_init(elev=20, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.set_facecolor('black'); ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_facecolor('black'); ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_facecolor('black'); ax.zaxis.pane.set_edgecolor('black')
    ax.grid(False)

    score_color = '#00e676' if n_correct >= 6 else ('#ffeb3b' if n_correct >= 4 else '#ff5252')
    ax.set_title(f'{cls_name}  [{n_correct}/8]',
                 color=score_color, fontsize=10, fontweight='bold', pad=4)

for idx in range(len(samples), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('All Test Objects — Front View  (score = models correct out of 8)',
             color='white', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(pad=0.3)
out = os.path.join(base_dir, 'plot_inference_summary.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='black')
plt.close()
print(f"Saved: plot_inference_summary.png")
print("\nDone. All files saved to ~/pointnet_experiments/")
