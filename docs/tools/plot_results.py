"""Render the stored 165-object evaluation summaries without changing values."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / 'assets/data/depth-bins.csv').open(encoding='utf-8-sig')))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#b8c1ca',
                     'axes.labelcolor': '#334155', 'xtick.color': '#536171',
                     'ytick.color': '#536171'})
styles = [('MonoAMNet', '#778da9', 'o'), ('MonoDGP', '#c58340', 's'),
          ('MonoDETR', '#648b70', '^'), ('GEMO3D', '#2468aa', 'D')]
for metric, label, filename in [('center_3d_error', 'Mean 3D center error (m)', 'center-error-en'),
                                 ('bev_iou', 'Mean BEV IoU', 'bev-iou-en')]:
    fig, ax = plt.subplots(figsize=(6.1, 4.1), layout='constrained')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e8ecf0', linewidth=.7)
    for model, color, marker in styles:
        subset = sorted([r for r in rows if r['metric'] == metric and r['model'] == model],
                        key=lambda r: float(r['depth_center']))
        assert sum(int(r['count']) for r in subset) == 165
        ax.errorbar([float(r['depth_center']) for r in subset],
                    [float(r['y_mean']) for r in subset],
                    yerr=[float(r['y_ci95']) for r in subset],
                    label=model, color=color, marker=marker, markersize=4.5,
                    linewidth=2 if model == 'GEMO3D' else 1.3,
                    elinewidth=.8, capsize=2.5)
    ax.set(xlabel='Ground-truth depth (m)', ylabel=label, xlim=(5, 35))
    ax.set_ylim(0, 1 if metric == 'bev_iou' else 10)
    ax.set_xticks([5, 10, 15, 20, 25, 30, 35])
    ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper left' if metric != 'bev_iou' else 'lower left')
    fig.savefig(ROOT / f'assets/{filename}.svg', metadata={'Date': None})
    plt.close(fig)
