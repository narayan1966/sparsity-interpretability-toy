"""Capacity against cleanliness (Fig. 3) from a main-grid summary.csv.

Usage:  python plot_capacity_vs_clean.py results/main_grid_summary.csv figures/fig3_capacity_cleanliness.png
"""
import csv, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else 'results/main_grid_summary.csv'
dst = sys.argv[2] if len(sys.argv) > 2 else 'fig3_capacity_cleanliness.png'
rows=list(csv.DictReader(open(src)))
col={'dense':'#444444','topk':'#d62728','gse':'#1f77b4','both':'#9467bd','static':'#8c8c8c'}
mk={'dense':'s','topk':'^','gse':'o','both':'D','static':'X'}
lab={'dense':'dense','topk':'top-k activations','gse':'GSE weights','both':'GSE + top-k','static':'static mask'}
fig,axes=plt.subplots(1,3,figsize=(15,4.8),dpi=170)
for ax,p in zip(axes,['0.01','0.02','0.05']):
    seen=set()
    for r in rows:
        if r['p']!=p: continue
        k=r['kind']; x=float(r['n_represented_mean']); y=float(r['abl_pr_w_mean'])
        xe=float(r['n_represented_sd']); ye=float(r['abl_pr_w_sd'])
        ax.errorbar(x,y,xerr=xe,yerr=ye,fmt=mk[k],color=col[k],ms=7,capsize=2,lw=1,alpha=.9,label=lab[k] if k not in seen else None)
        seen.add(k)
        t=None
        if k=='topk': t=f"k={int(float(r['k']))}"
        elif k in('gse','static'): t=f"{float(r['sparsity'])*100:g}%"
        elif k=='dense': t='dense'
        if t: ax.annotate(t,(x,y),xytext=(5,4),textcoords='offset points',fontsize=8,color=col[k])
    ax.set_yscale('log'); ax.grid(alpha=.3)
    ax.set_title(f'feature probability p = {p}',fontsize=11)
    ax.set_xlabel('features represented (more is better)')
    if p=='0.01': ax.set_ylabel('features disturbed per neuron removed (fewer is cleaner)')
    ax.text(.97,.04,'better',transform=ax.transAxes,ha='right',color='#2a7a2a',fontweight='bold',fontsize=9)
axes[0].legend(fontsize=8,loc='upper left')
fig.suptitle('Capacity against cleanliness (mean ± sd over 5 seeds, 64 hidden neurons). Best is the bottom right.',fontsize=11)
fig.tight_layout(); fig.savefig(dst,facecolor='white')
