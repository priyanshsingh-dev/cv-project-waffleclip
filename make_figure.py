# Builds images/tpr_results.png from results/benchmark.csv (ViT-B/32, top-1 accuracy).
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows = {r[0]: float(r[1]) for r in csv.reader(open('results/benchmark.csv')) if r}
get = lambda d, m: next(v for k, v in rows.items() if f'dataset={d};' in k and k.endswith(m))
datasets = ['dtd', 'eurosat', 'pets', 'cub']
names = ['DTD', 'EuroSAT', 'Oxford Pets', 'CUB-200']
series = [('CLIP', 'mode=clip', '#b8b6ad'),
          ('WaffleCLIP', 'reps=7', '#9ec5f4'),
          ('+ prototype refinement', 'reps=7; refine; refine_balance=0', '#5598e7'),
          ('+ class balance (full TPR, ours)', 'reps=7; refine', '#1c5cab')]

fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=160)
x, w = np.arange(len(datasets)), 0.2
for i, (label, suffix, color) in enumerate(series):
    vals = [get(d, suffix) for d in datasets]
    ax.bar(x + (i - 1.5) * w, vals, w - 0.02, color=color, label=label, zorder=2)
    if i == len(series) - 1:
        for xi, d, v in zip(x, datasets, vals):
            ax.text(xi + 1.5 * w, v + 0.8, f'+{v - get(d, "reps=7"):.1f}', ha='center', fontsize=9, color='#0b0b0b', weight='bold')
ax.set_xticks(x, names)
ax.set_ylabel('Top-1 accuracy (%)', color='#52514e')
ax.set_ylim(0, 105)
ax.grid(axis='y', color='#e6e5e0', zorder=0)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.tick_params(colors='#52514e', length=0)
ax.legend(frameon=False, loc='upper left', fontsize=8.5, ncol=2)
ax.set_title('Balanced Test-time Prototype Refinement on top of WaffleCLIP (ViT-B/32)', fontsize=11, loc='left')
plt.tight_layout()
plt.savefig('images/tpr_results.png', facecolor='#fcfcfb')
