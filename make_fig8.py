"""Neuron types in the related-features experiment (Fig. 8).

Usage:  python make_fig8.py results/related_features_all.csv figures/fig8_related_features.png
The input is the concatenation of the summary.csv files written by clustered_features_toy.py
(columns cooc, fam_out, p, arm, frac_clean_mean, frac_grouped_mean, frac_mixed_mean).
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/related_features_all.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "fig8_related_features.png"
d = pd.read_csv(src)
d = d[d["fam_out"] == 1]
ARMS = [("dense", "Dense"), ("gse_s0.75", "GSE"), ("static_s0.75", "Static"),
        ("topk_k8", "Top-k"), ("pre_gse_s0.75", "Dense→GSE")]
COL = {"clean": "#2e7d32", "grouped": "#1f77b4", "mixed": "#d62728"}
ps = sorted(d["p"].unique())
cs = sorted(d["cooc"].unique())
fig, axes = plt.subplots(len(ps), len(cs), figsize=(6.5 * 1.0 * 2, 3.3 * 2), squeeze=False)
for i, p in enumerate(ps):
    for j, c in enumerate(cs):
        ax = axes[i][j]
        sub = d[(d["p"] == p) & (d["cooc"] == c)].set_index("arm")
        bottom = [0.0] * len(ARMS)
        for kind in ("clean", "grouped", "mixed"):
            vals = [float(sub.loc[a, f"frac_{kind}_mean"]) for a, _ in ARMS]
            ax.bar(range(len(ARMS)), vals, bottom=bottom, color=COL[kind], width=0.8,
                   label=kind if (i, j) == (0, 0) else None)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([n for _, n in ARMS], rotation=45, ha="right")
        ax.set_title(f"p = {p:g}, c = {c:g}", fontsize=10)
        if j == 0:
            ax.set_ylabel("fraction of neurons")
        else:
            ax.set_yticklabels([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
fig.legend(loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(dst, dpi=200, facecolor="white")
print("wrote", dst)
