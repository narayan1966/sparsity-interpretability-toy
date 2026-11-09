# Sparse training and neuron interpretability in a toy model of superposition

Code, results and figures for the paper *Does Sparse Training Make Neurons Easier to Read? Capacity and Monosemanticity in a Toy Model of Superposition* (Narayan Srinivasa).

A one-hidden-layer ReLU network is trained to compute the absolute value of 128 sparse input features through 64, 96 or 128 hidden neurons. We compare dense training with top-k activation sparsity, always-sparse weight training with Guided Stochastic Exploration (GSE), the two combined, and a static random mask. We measure how many features the network represents (capacity) and how many features each neuron affects (cleanliness).

## Contents

| Path | What it is |
|---|---|
| `sparsity_interp_toy.py` | Main experiment: all arms, metrics, neuron snapshots and visualizations (Sections III-A to III-D, III-F) |
| `pretrained_then_sparse.py` | Sparsifying a trained dense network (Section III-E) |
| `clustered_features_toy.py` | Related features: families of co-occurring features (Section III-G) |
| `mask_transfer.py` | Retraining on the mask that GSE finds (Section III-H) |
| `make_fig2.py` | Schematic of GSE (Fig. 2) |
| `plot_capacity_vs_clean.py` | Capacity against cleanliness (Fig. 3) |
| `make_fig8.py` | Neuron types for related features (Fig. 8) |
| `results/` | Summary tables for every experiment (see below) |
| `figures/` | The figures of the paper (`fig1.png` to `fig8.png`) |

Figures 4 to 7 come from the `snapshots` and `visualize` commands of `sparsity_interp_toy.py`. Fig. 1 is a hand-drawn schematic.

## Results files

| File | Experiment |
|---|---|
| `results/main_grid_summary.csv`, `main_grid_all_runs.csv` | Main grid, 64 neurons, 5 seeds, ranking within each matrix (Table I, Fig. 3) |
| `results/scope_global_summary.csv` | Same arms with the ranking across both matrices (Table III) |
| `results/width_m96_summary.csv`, `width_m128_summary.csv` | Width runs, 3 seeds (Table V) |
| `results/lr0.001_summary.csv`, `lr0.01_summary.csv` | Learning-rate runs, 128 neurons (Table IV) |
| `results/pretrained_summary.csv` | Starting from a trained dense network (Table VI) |
| `results/related_features_all.csv`, `related_features_lr0.001_summary.csv` | Related features (Table VII, Fig. 8) |
| `results/mask_transfer_summary.csv`, `mask_transfer_all_runs.csv` | Retraining on the GSE mask (Table VIII) |

## Environment

Python 3.14.4, PyTorch 2.11.0 (CUDA 12.8 build), NumPy 2.5.4, pandas 3.1.1, matplotlib 3.11.2. Hardware: a laptop with an NVIDIA GeForce RTX 5070 GPU and 16 GB of system RAM. Runs on a different device or PyTorch version follow the same trends but are not bit-identical, because small numerical differences grow over 20,000 training steps.

```
pip install torch numpy pandas matplotlib
```

## Reproducing the experiments

Every command stores each run as its own file and resumes where it stopped. `python <script>.py --help` lists all options. The paper ranks connections within each weight matrix, so the main experiment is run with `--scope local`.

Main grid (Table I, Fig. 3):

```
python sparsity_interp_toy.py run --out-dir results_main --scope local --probs 0.01 0.02 0.05 --ablate-samples 50000
python sparsity_interp_toy.py analyze --out-dir results_main
python plot_capacity_vs_clean.py results_main/summary.csv fig3.png
```

Ranking across both matrices (Table III): the same command with `--scope global --probs 0.05`.

Width and learning rate (Tables IV and V):

```
for m in 96 128; do
  python sparsity_interp_toy.py run --out-dir results_m$m --scope local --m-hidden $m \
    --arms dense,gse_s0.75,gse_s0.5 --probs 0.02 0.01 --seeds 3 --ablate-samples 50000
  python sparsity_interp_toy.py analyze --out-dir results_m$m
done
for lr in 0.001 0.01; do
  python sparsity_interp_toy.py run --out-dir results_lr$lr --scope local --m-hidden 128 --lr $lr \
    --arms dense,gse_s0.75 --probs 0.01 --seeds 3 --ablate-samples 50000
  python sparsity_interp_toy.py analyze --out-dir results_lr$lr
done
```

Starting from a trained dense network (Table VI):

```
python pretrained_then_sparse.py run --out-dir results_pre --probs 0.01 0.02 0.05 --m-hidden 64 --seeds 5
python pretrained_then_sparse.py run --out-dir results_pre --probs 0.01 --m-hidden 128 --seeds 5
python pretrained_then_sparse.py analyze --out-dir results_pre --ref-summary results_main/summary.csv
```

Related features (Table VII, Fig. 8):

```
python clustered_features_toy.py run --out-dir results_cluster --probs 0.01 0.02 --seeds 3 --conds 0:1,0.25:1,0.5:1,0.9:1
python clustered_features_toy.py analyze --out-dir results_cluster
python make_fig8.py results/related_features_all.csv fig8.png
```

Retraining on the mask GSE finds (Table VIII):

```
python mask_transfer.py run --out-dir results_mask --probs 0.01 0.02 0.05 --m-hidden 64 --seeds 5
python mask_transfer.py analyze --out-dir results_mask
```

Neuron snapshots and the neuron figures (Figs. 4 to 7):

```
python sparsity_interp_toy.py snapshots --out-dir results_main --scope local --probs 0.01
python sparsity_interp_toy.py visualize --out-dir results_main --scope local --probs 0.01
python make_fig2.py
```

## Settings used in the paper

n = 128 features, each active with probability p and uniform on [-1, 1] when active; the target is the absolute value; Adam with learning rate 3e-3 decayed by a cosine schedule to 1%, batch 1024, 20,000 steps. GSE swaps connections every 100 steps, the fraction replaced decays from 0.2 to zero by 60% of training, and the topology is frozen after that. Evaluation uses 50,000 held-out samples. The ablation measure uses 50,000 samples in the main experiments and 10,000 (the script default) in `mask_transfer.py`.

## Citation

If you use this code, please cite the paper above.

## License

MIT.
