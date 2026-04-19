import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

logs_dir = os.path.expanduser('~/pointnet_experiments/logs')
output_dir = os.path.expanduser('~/pointnet_experiments')

experiments = {
    'Baseline':         'baseline_direct',
    'Exp1: Cosine':     'exp1_cosine',
    'Exp2: Attention':  'exp2_attention',
    'Exp3: PCA':        'exp3_pca',
    'Exp4: ANN':        'exp4_ann',
    'Exp5: Combined':   'exp5_combined',
    'Exp6: MSG':        'exp6_msg',
    'Exp7: MSG+Normals':'exp7_msg_normals',
    'Exp8: Shared ANN': 'exp8_shared_ann_msg',
}

def parse_log(filepath):
    train_acc, test_acc = [], []
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            t = re.search(r'Train Instance Accuracy: ([0-9.]+)', line)
            if t:
                train_acc.append(float(t.group(1)) * 100)
            v = re.search(r'Test Instance Accuracy: ([0-9.]+)', line)
            if v:
                test_acc.append(float(v.group(1)) * 100)
    return train_acc, test_acc

# ── Plot 1: Final accuracy bar chart ──────────────────────────────────────────
best_acc = {}
for name, logname in experiments.items():
    logpath = os.path.join(logs_dir, f'{logname}.out')
    _, test_acc = parse_log(logpath)
    best_acc[name] = max(test_acc) if test_acc else 0

baseline_val = best_acc['Baseline']
colors = []
for name, acc in best_acc.items():
    if name == 'Baseline':
        colors.append('#2196F3')
    elif acc >= baseline_val:
        colors.append('#4CAF50')
    else:
        colors.append('#F44336')

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(best_acc.keys(), best_acc.values(), color=colors, edgecolor='black', linewidth=0.5, zorder=3)
ax.axhline(y=baseline_val, color='navy', linestyle='--', linewidth=1.5, label=f'Baseline: {baseline_val:.2f}%', zorder=4)
ax.set_ylim(88, 95)
ax.set_ylabel('Best Test Instance Accuracy (%)', fontsize=12)
ax.set_title('PointNet++ Experiment Results on ModelNet40', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=25)
ax.grid(axis='y', alpha=0.4, zorder=0)

for bar, (name, acc) in zip(bars, best_acc.items()):
    diff = acc - baseline_val
    label = f'{acc:.2f}%\n({diff:+.2f}%)'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            label, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

legend_patches = [
    mpatches.Patch(color='#2196F3', label='Baseline'),
    mpatches.Patch(color='#4CAF50', label='Beats Baseline'),
    mpatches.Patch(color='#F44336', label='Below Baseline'),
]
ax.legend(handles=legend_patches + [plt.Line2D([0],[0], color='navy', linestyle='--', label=f'Baseline: {baseline_val:.2f}%')], fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot_accuracy_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Plot 1 saved: accuracy comparison bar chart")

# ── Plot 2: Training curves (test accuracy over epochs) ───────────────────────
colors_curve = [
    '#000000', '#E53935', '#8E24AA', '#1E88E5', '#00ACC1',
    '#FF6F00', '#43A047', '#6D4C41', '#00897B'
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, logname) in enumerate(experiments.items()):
    logpath = os.path.join(logs_dir, f'{logname}.out')
    train_acc, test_acc = parse_log(logpath)
    ax = axes[idx]
    epochs_train = range(1, len(train_acc) + 1)
    epochs_test = range(1, len(test_acc) + 1)
    ax.plot(epochs_train, train_acc, color=colors_curve[idx], alpha=0.4, linewidth=1, label='Train')
    ax.plot(epochs_test, test_acc, color=colors_curve[idx], linewidth=1.5, label='Test')
    ax.axhline(y=baseline_val, color='navy', linestyle='--', linewidth=1, alpha=0.7)
    best = max(test_acc) if test_acc else 0
    ax.set_title(f'{name}\nBest: {best:.2f}%', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('Accuracy (%)', fontsize=8)
    ax.set_ylim(20, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

plt.suptitle('Training & Test Accuracy Curves — All Experiments', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot_training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Plot 2 saved: training curves")

# ── Plot 3: Final comparison table heatmap ────────────────────────────────────
names = list(best_acc.keys())
accs = list(best_acc.values())
diffs = [a - baseline_val for a in accs]

fig, ax = plt.subplots(figsize=(12, 5))
data = np.array([accs, [baseline_val]*len(accs)])
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=89, vmax=94)

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Experiment', 'Baseline'], fontsize=10)

for j, (acc, diff) in enumerate(zip(accs, diffs)):
    ax.text(j, 0, f'{acc:.2f}%\n({diff:+.2f}%)', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(j, 1, f'{baseline_val:.2f}%', ha='center', va='center', fontsize=9)

plt.colorbar(im, ax=ax, label='Accuracy (%)')
ax.set_title('Accuracy Heatmap — All Experiments vs Baseline', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Plot 3 saved: heatmap")

print("\nAll plots saved to ~/pointnet_experiments/")
