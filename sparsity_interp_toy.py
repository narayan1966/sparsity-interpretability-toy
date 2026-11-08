#!/usr/bin/env python3
"""
sparsity_interp_toy.py - Stage 1: does sparsity make neurons more monosemantic?

Question
    At matched task loss, do weight sparsity (GSE, unstructured), activation sparsity
    (top-k on the hidden layer), or both make hidden neurons respond to fewer
    features than a dense network's neurons do?

Setup
    n synthetic features, each active with probability p, with value uniform in
    [-1, 1] when active (random sign).  Task: y = |x| through a hidden ReLU layer
    of m < n neurons:   x -> W1 -> ReLU (+ optional top-k) -> W2 -> y_hat.
    |x| needs a nonlinearity (relu(x) + relu(-x)), so a clean solution spends
    2 neurons per feature, i.e. at most m/2 features; more than that requires
    superposition.

Arms (see arm_list)
    dense | gse_s{S} | topk_k{K} | both_s{S}_k{K} | static_s{S}
    static = the same kind of random sparse mask, fixed at init (control for GSE's prune-grow).

Metrics (computed on a fixed held-out set)
    norm_loss      loss / loss of predicting the mean (1.0 = learned nothing)
    n_represented  features reconstructed with relative error < --rep-thresh
    w_pr           weight participation ratio of W1 rows (effective #features / neuron)
    act_pr         activation-correlation participation ratio (behavioral)
    abl_pr_w       ablation specificity, importance-weighted: effective #features
                   whose error rises when one neuron is ablated (main metric)
    abl_mono_frac  fraction of important neurons with ablation PR < --mono-thresh

Usage
    python sparsity_interp_toy.py pilot   --out-dir results     # timing estimate
    python sparsity_interp_toy.py run     --out-dir results     # full grid (resumable)
    python sparsity_interp_toy.py analyze --out-dir results     # CSV + plots
    python sparsity_interp_toy.py snapshots --out-dir results --probs 0.05   # weights during training
    python sparsity_interp_toy.py visualize --out-dir results --probs 0.05   # figures, GIF, index.html
    # quick smoke test:
    python sparsity_interp_toy.py run --out-dir smoke --steps 600 --seeds 1 \
        --probs 0.1 --max-jobs 6
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ToyNet(nn.Module):
    def __init__(self, n, m, topk=None):
        super().__init__()
        self.w1 = nn.Linear(n, m)
        self.w2 = nn.Linear(m, n)
        self.topk = topk
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def hidden(self, x):
        h = torch.relu(self.w1(x))
        if self.topk is not None and self.topk < h.shape[1]:
            idx = h.topk(self.topk, dim=1).indices
            keep = torch.zeros_like(h).scatter_(1, idx, 1.0)
            h = h * keep
        return h

    def forward(self, x):
        h = self.hidden(x)
        return self.w2(h), h


# ---------------------------------------------------------------------------
# GSE: always-sparse training with Guided Stochastic Exploration
# ---------------------------------------------------------------------------


class GSE:
    """Always-sparse training with unstructured Guided Stochastic Exploration (GSE),
    following Heddes, Srinivasa, Givargis and Nicolau (IJCNN 2025).

    The network is sparse from step 0.  The initial connections are drawn at random
    (Erdos-Renyi, equal density in the two layers of this toy) and the weights are
    initialized so that each unit's pre-activation has unit variance given its actual
    fan-in.  Every `update_freq` steps a prune-grow round runs:

      grow   Sample a random subset S of the inactive connections (uniform over
             (output, input) pairs, size gamma * |active|).  Compute the gradient on a
             fresh batch and activate the connections in S with the largest gradient
             magnitude, so that the number of active connections rises by a fraction
             alpha.  New weights start at zero.  (A layer that is at most 50% sparse
             offers all of its inactive connections instead, as in the authors' code.)
      prune  Remove the smallest-magnitude active weights, excluding the ones just
             grown, so the total returns to its starting value.

    Both steps rank connections over the two weight matrices together (scope="global")
    or inside each matrix (scope="local").  alpha follows a cosine from alpha0 to zero
    at anneal_frac of training, after which the topology is frozen.  Biases stay dense.
    With dynamic=False the initial masks never change (the static-mask control).

    Note on scale: this toy computes the dense gradient for the grow step because the
    network is tiny.  The selection rule is the same as with a sparse implementation
    that only evaluates the gradient of the sampled subset."""

    def __init__(self, layers, sparsity, total_steps, update_freq=100, alpha=0.2,
                 anneal_frac=0.6, gamma=1.0, scope="global", dynamic=True):
        assert scope in ("global", "local")
        self.layers = layers  # list of nn.Linear
        self.sparsity = sparsity
        self.total_steps = total_steps
        self.update_freq = update_freq
        self.alpha0 = alpha
        self.t_end = anneal_frac * total_steps
        self.gamma = gamma
        self.scope = scope
        self.dynamic = dynamic
        self.step = 0
        self.rounds = 0
        self.growing = False  # True while the dense gradient for a grow step is computed
        self.masks, self.hooks = [], []

        shapes = [tuple(l.weight.shape) for l in layers]  # (out, in)
        total = sum(o * i for o, i in shapes)
        scale = (1.0 - sparsity) * total / sum(o + i for o, i in shapes)  # Erdos-Renyi
        self.n_layer = [math.ceil(o * i * min(1.0, scale * (o + i) / (o * i))) for o, i in shapes]
        self.n_total = sum(self.n_layer)
        for lin, n_l in zip(layers, self.n_layer):
            w = lin.weight.data
            o, i = w.shape
            mk = torch.zeros(o * i, dtype=torch.bool, device=w.device)
            mk[torch.randperm(o * i, device=w.device)[:n_l]] = True
            mk = mk.view(o, i)
            fan = mk.sum(1, keepdim=True).clamp_min(1).float()
            bound = math.sqrt(3.0) / torch.sqrt(fan)
            w.copy_((torch.rand_like(w) * 2 - 1) * bound * mk)
            self.masks.append(mk)
        for i, lin in enumerate(layers):
            self.hooks.append(lin.weight.register_hook(self._make_hook(i)))

    def _make_hook(self, i):
        def hook(grad):
            return grad if self.growing else grad * self.masks[i]
        return hook

    def apply_masks(self):
        for lin, mk in zip(self.layers, self.masks):
            lin.weight.data.mul_(mk)

    def frozen(self):
        return self.step >= self.t_end

    def alpha_now(self):
        if self.step >= self.t_end:
            return 0.0
        return self.alpha0 / 2.0 * (1.0 + math.cos(math.pi * self.step / self.t_end))

    @torch.no_grad()
    def after_step(self, optimizer, grad_fn=None):
        self.step += 1
        self.apply_masks()
        if self.dynamic and self.step % self.update_freq == 0 and not self.frozen():
            return self.prune_grow(optimizer, grad_fn)
        return 0

    def _topk_masks(self, scores, ks):
        """Boolean masks selecting the ks largest scores, over all layers (global) or per layer."""
        if self.scope == "global":
            flat = torch.cat([s.flatten() for s in scores])
            k = min(sum(ks), flat.numel())
            keep = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
            if k > 0:
                keep[flat.topk(k).indices] = True
            out, start = [], 0
            for s in scores:
                out.append(keep[start:start + s.numel()].view_as(s))
                start += s.numel()
            return out
        out = []
        for s, k in zip(scores, ks):
            keep = torch.zeros(s.numel(), dtype=torch.bool, device=s.device)
            if k > 0:
                keep[s.flatten().topk(min(k, s.numel())).indices] = True
            out.append(keep.view_as(s))
        return out

    def prune_grow(self, optimizer, grad_fn):
        alpha = self.alpha_now()
        # 1. dense gradient on a fresh batch (masking of gradients switched off)
        for lin in self.layers:
            lin.weight.grad = None
            lin.bias.grad = None
        self.growing = True
        with torch.enable_grad():
            grad_fn()
        self.growing = False
        grads = [lin.weight.grad.detach().abs() for lin in self.layers]
        for lin in self.layers:
            lin.weight.grad = None
            lin.bias.grad = None
        with torch.no_grad():
            neg_inf = float("-inf")
            # 2. grow: random subset of inactive connections, keep those with the largest gradient
            grow_scores, n_cand = [], []
            for g, mk in zip(grads, self.masks):
                o, i = mk.shape
                nnz = int(mk.sum())
                if 1.0 - nnz / mk.numel() <= 0.5:
                    cand = ~mk
                else:
                    size = int(nnz * self.gamma)
                    r = torch.randint(o, (size,), device=mk.device)
                    c = torch.randint(i, (size,), device=mk.device)
                    cand = torch.zeros_like(mk)
                    cand[r, c] = True
                    cand &= ~mk
                n_cand.append(int(cand.sum()))
                grow_scores.append(g.masked_fill(~cand, neg_inf))
            if self.scope == "global":
                n_grow = min(int(self.n_total * (1.0 + alpha)) - self.n_total, sum(n_cand))
                ks = [n_grow]
            else:
                ks = [min(int(n * (1.0 + alpha)) - n, c) for n, c in zip(self.n_layer, n_cand)]
            if sum(ks) <= 0:
                return 0
            grown = self._topk_masks(grow_scores, ks)
            # 3. prune: smallest-magnitude active weights, never the ones just grown
            prune_scores = []
            for lin, mk, gr in zip(self.layers, self.masks, grown):
                s = lin.weight.data.abs().masked_fill(~(mk | gr), neg_inf)
                prune_scores.append(s.masked_fill(gr, float("inf")))
            keep = self._topk_masks(prune_scores, [self.n_total] if self.scope == "global" else self.n_layer)
            n_changed = 0
            for lin, mk, kp in zip(self.layers, self.masks, keep):
                changed = mk != kp
                n_changed += int(changed.sum())
                mk.copy_(kp)
                lin.weight.data.mul_(mk)
                state = optimizer.state.get(lin.weight, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    buf = state.get(key)
                    if buf is not None:
                        buf[changed] = 0.0
            self.rounds += 1
            return n_changed // 2

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def check(self):
        """(active connections, weights that are non-zero outside the mask,
        density of each matrix, overall mask sparsity)"""
        nnz = sum(int(mk.sum()) for mk in self.masks)
        stray = sum(int(((lin.weight.data != 0) & ~mk).sum()) for lin, mk in zip(self.layers, self.masks))
        dens = [float(mk.float().mean()) for mk in self.masks]
        return nnz, stray, dens, 1.0 - nnz / sum(mk.numel() for mk in self.masks)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def sample(n, p, batch, device, gen):
    active = (torch.rand(batch, n, device=device, generator=gen) < p).float()
    val = 2 * torch.rand(batch, n, device=device, generator=gen) - 1
    return active * val


def feature_var(p):
    """Variance of |x_i| (the quantity the mean predictor leaves unexplained)."""
    return p / 3.0 - (p / 2.0) ** 2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pr(v, dim=-1, eps=1e-12):
    """Participation ratio (sum v)^2 / sum v^2 of a non-negative vector."""
    s1 = v.sum(dim)
    s2 = (v ** 2).sum(dim)
    return s1 ** 2 / (s2 + eps)


@torch.no_grad()
def compute_metrics(model, p, n, device, args):
    gen = torch.Generator(device="cpu").manual_seed(12345)
    x = sample(n, p, args.eval_samples, "cpu", gen).to(device)
    t = x.abs()
    y, h = model(x)
    mse_f = ((y - t) ** 2).mean(0)  # per feature
    var_f = feature_var(p)
    res = {}
    res["loss"] = float(mse_f.mean())
    res["norm_loss"] = float(mse_f.mean() / var_f)
    res["n_represented"] = int((mse_f / var_f < args.rep_thresh).sum())

    # weight selectivity: participation ratio of squared input weights per neuron
    w1 = model.w1.weight.data
    w2 = model.w2.weight.data
    live = h.std(0) > 1e-6
    res["n_live_neurons"] = int(live.sum())
    res["frac_nonzero_hidden"] = float((h > 0).float().mean())
    w_pr = pr(w1.pow(2))  # (m,)
    res["w_pr"] = float(w_pr[live].mean()) if live.any() else float("nan")

    # activation selectivity: squared correlation of each neuron with each |x_i|
    ht = (h - h.mean(0)) / (h.std(0) + 1e-8)
    tt = (t - t.mean(0)) / (t.std(0) + 1e-8)
    r = (ht.T @ tt) / h.shape[0]  # (m, n)
    r2 = r.pow(2)
    r2 = torch.where(r2 < args.corr_floor, torch.zeros_like(r2), r2)
    act_pr = pr(r2)
    res["act_pr"] = float(act_pr[live].mean()) if live.any() else float("nan")

    # ablation specificity
    sub = slice(0, args.ablate_samples)
    hs, ts = h[sub], t[sub]
    ybase = model.w2(hs)
    base = ((ybase - ts) ** 2).mean(0)
    m = hs.shape[1]
    delta = torch.zeros(m, n, device=device)
    for j in range(m):
        yab = ybase - hs[:, j:j + 1] * w2[:, j].unsqueeze(0)
        delta[j] = ((yab - ts) ** 2).mean(0) - base
    delta = delta.clamp_min(0)
    tot = delta.sum(1)  # (m,)
    rel = tot / (n * var_f)
    important = (rel > args.important_thresh) & live
    abl_pr = pr(delta)
    res["n_important"] = int(important.sum())
    if important.any():
        wts = tot[important]
        res["abl_pr_w"] = float((abl_pr[important] * wts).sum() / wts.sum())
        res["abl_pr_mean"] = float(abl_pr[important].mean())
        res["abl_mono_frac"] = float((abl_pr[important] < args.mono_thresh).float().mean())
    else:
        res["abl_pr_w"] = res["abl_pr_mean"] = res["abl_mono_frac"] = float("nan")
    return res


# ---------------------------------------------------------------------------
# Arms / jobs
# ---------------------------------------------------------------------------

SPARSITIES = [0.5, 0.75, 0.875]
KS = [32, 16, 8]


def arm_list():
    """(tier, name, kind, sparsity, k).  Tier controls run order."""
    arms = [(1, "dense", "dense", None, None)]
    arms += [(1, f"gse_s{s}", "gse", s, None) for s in SPARSITIES]
    arms += [(1, f"topk_k{k}", "topk", None, k) for k in KS]
    arms += [(2, f"both_s{s}_k{k}", "both", s, k) for s in SPARSITIES for k in KS]
    arms += [(3, f"static_s{s}", "static", s, None) for s in SPARSITIES]
    return arms


def family(kind):
    return kind


def make_jobs(args):
    jobs = []
    wanted = set(a.strip() for a in args.arms.split(",")) if args.arms else None
    for pi, p in enumerate(args.probs):
        for tier, name, kind, s, k in arm_list():
            if wanted is not None and name not in wanted and kind not in wanted:
                continue
            for seed in range(args.seeds):
                t = tier if pi == 0 else 4  # primary p first, others later
                jobs.append((t, seed, p, name, kind, s, k))
    jobs.sort(key=lambda j: (j[0], j[1], j[2], j[3]))
    return jobs


def run_path(out_dir, p, name, seed):
    return Path(out_dir) / "runs" / f"p{p}_{name}_seed{seed}.json"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one(p, kind, s, k, seed, args, device, snap_dir=None, snap_steps=()):
    n, m = args.n_features, args.m_hidden
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(1000 + seed)
    model = ToyNet(n, m, topk=k if kind in ("topk", "both") else None).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.01)
    gse = None
    grad_fn = None
    if kind in ("gse", "both", "static"):
        gse = GSE([model.w1, model.w2], s, args.steps, update_freq=args.update_freq,
                  alpha=args.alpha, anneal_frac=args.anneal_frac, gamma=args.subset_gamma,
                  scope=args.scope, dynamic=(kind != "static"))
        gb = args.grow_batch or args.batch

        def grad_fn():
            xg = sample(n, p, gb, device, gen)
            yg, _ = model(xg)
            ((yg - xg.abs()) ** 2).mean().backward()
    swaps = 0
    snap_set = set(snap_steps)
    kk = k if kind in ("topk", "both") else 0
    if snap_dir is not None:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        save_snapshot(model, gse, kk, snap_dir, 0)
    t0 = time.time()
    for step in range(args.steps):
        x = sample(n, p, args.batch, device, gen)
        y, _ = model(x)
        loss = ((y - x.abs()) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if gse is not None:
            swaps += gse.after_step(opt, grad_fn)
        if snap_dir is not None and (step + 1) in snap_set:
            save_snapshot(model, gse, kk, snap_dir, step + 1)
    train_time = time.time() - t0
    res = compute_metrics(model, p, n, device, args)
    res["train_time_s"] = train_time
    res["total_swaps"] = swaps
    if gse is not None:
        nnz, stray, dens, sp = gse.check()
        res["mask_sparsity"] = sp
        res["density_w1"], res["density_w2"] = dens
        res["gse_rounds"] = gse.rounds
        res["weight_sparsity"] = float((torch.cat([model.w1.weight.flatten(),
                                                   model.w2.weight.flatten()]) == 0).float().mean())
        assert stray == 0, f"{stray} weights are non-zero outside the mask"
        if gse.scope == "global":
            assert nnz == gse.n_total, f"active connections drifted: {nnz} != {gse.n_total}"
        else:
            assert [int(mk.sum()) for mk in gse.masks] == gse.n_layer, "per-layer connection count drifted"
        gse.remove_hooks()
    else:
        res["weight_sparsity"] = float((torch.cat([model.w1.weight.flatten(),
                                                   model.w2.weight.flatten()]) == 0).float().mean())
    return res


def pick_device(arg):
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_run(args):
    device = pick_device(args.device)
    out = Path(args.out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    jobs = make_jobs(args)
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    print(f"device={device}  jobs={len(jobs)}  steps={args.steps}")
    done, t_start = 0, time.time()
    for ji, (tier, seed, p, name, kind, s, k) in enumerate(jobs, 1):
        path = run_path(out, p, name, seed)
        if path.exists():
            continue
        res = train_one(p, kind, s, k, seed, args, device)
        res.update(dict(p=p, arm=name, kind=kind, sparsity=s, k=k, seed=seed, tier=tier))
        path.write_text(json.dumps(res, indent=2))
        done += 1
        el = time.time() - t_start
        eta = el / done * (len(jobs) - ji)
        print(f"[{ji}/{len(jobs)}] p={p} {name} seed={seed} "
              f"norm_loss={res['norm_loss']:.3f} rep={res['n_represented']} "
              f"abl_pr={res['abl_pr_w']:.2f} w_pr={res['w_pr']:.2f} "
              f"({res['train_time_s']:.0f}s)  eta {eta/60:.0f} min", flush=True)
    print("done.")


def cmd_pilot(args):
    device = pick_device(args.device)
    p = args.probs[0]
    jobs = make_jobs(args)
    counts = {}
    for _, _, _, _, kind, _, _ in jobs:
        counts[kind] = counts.get(kind, 0) + 1
    probes = [("dense", "dense", None, None), ("gse", "gse", 0.75, None),
              ("topk", "topk", None, 16), ("both", "both", 0.75, 16),
              ("static", "static", 0.75, None)]
    total, rows = 0.0, []
    for _, kind, s, k in probes:
        t0 = time.time()
        res = train_one(p, kind, s, k, 0, args, device)
        dt = time.time() - t0
        total += dt * counts.get(kind, 0)
        rows.append((kind, dt, res["norm_loss"], res["abl_pr_w"]))
        print(f"  {kind:7s} {dt:6.1f}s  norm_loss={res['norm_loss']:.3f}  rep={res['n_represented']}  "
              f"important={res['n_important']}  abl_pr_w={res['abl_pr_w']:.2f}  act_pr={res['act_pr']:.2f}  w_pr={res['w_pr']:.2f}")
    print(f"\ndevice={device}  jobs={len(jobs)}  estimated total = {total/3600:.2f} h "
          f"({total/60:.0f} min)")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

METRICS = ["norm_loss", "n_represented", "w_pr", "act_pr", "abl_pr_w", "abl_mono_frac",
           "weight_sparsity", "frac_nonzero_hidden"]


def load_runs(out_dir):
    return [json.loads(f.read_text()) for f in sorted((Path(out_dir) / "runs").glob("*.json"))]


def cmd_analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out_dir)
    runs = load_runs(out)
    if not runs:
        raise SystemExit("no runs found")
    with open(out / "all_runs.csv", "w", newline="") as f:
        keys = sorted({k for r in runs for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(runs)

    # per-(p, arm) mean +- sd
    groups = {}
    for r in runs:
        groups.setdefault((r["p"], r["arm"]), []).append(r)
    summary = []
    for (p, arm), rs in sorted(groups.items()):
        row = dict(p=p, arm=arm, kind=rs[0]["kind"], sparsity=rs[0]["sparsity"], k=rs[0]["k"], n=len(rs))
        for mname in METRICS:
            vals = np.array([r[mname] for r in rs], dtype=float)
            row[mname + "_mean"] = float(np.nanmean(vals))
            row[mname + "_sd"] = float(np.nanstd(vals))
        summary.append(row)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    colors = {"dense": "#444444", "gse": "#1f77b4", "topk": "#d62728",
              "both": "#9467bd", "static": "#8c8c8c"}
    markers = {"dense": "s", "gse": "o", "topk": "^", "both": "D", "static": "x"}
    ps = sorted({r["p"] for r in runs})
    for yname, ylabel, fname in [("abl_pr_w", "features affected per neuron (ablation, lower = more monosemantic)", "pareto_ablation.png"),
                                 ("act_pr", "features per neuron (activation correlation)", "pareto_activation.png"),
                                 ("w_pr", "features per neuron (input weights)", "pareto_weights.png")]:
        fig, axes = plt.subplots(1, len(ps), figsize=(5.2 * len(ps), 4.4), squeeze=False)
        for ax, p in zip(axes[0], ps):
            for kind in ["dense", "gse", "topk", "both", "static"]:
                rows = [r for r in summary if r["p"] == p and r["kind"] == kind]
                if not rows:
                    continue
                ax.errorbar([r["norm_loss_mean"] for r in rows], [r[yname + "_mean"] for r in rows],
                            xerr=[r["norm_loss_sd"] for r in rows], yerr=[r[yname + "_sd"] for r in rows],
                            fmt=markers[kind], color=colors[kind], label=kind, alpha=0.85,
                            capsize=2, ms=6, ls="none")
            ax.set_xscale("log")
            ax.set_title(f"feature probability p={p}")
            ax.set_xlabel("normalized loss (lower = better)")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)
    print(f"wrote {out/'summary.csv'}, {out/'all_runs.csv'} and pareto_*.png")
    print("\nsummary (p, arm, norm_loss, n_represented, abl_pr_w, act_pr, w_pr):")
    for r in summary:
        print(f"  p={r['p']:<5} {r['arm']:18s} n={r['n']} loss={r['norm_loss_mean']:.3f} "
              f"rep={r['n_represented_mean']:.1f} abl={r['abl_pr_w_mean']:.2f} "
              f"act={r['act_pr_mean']:.2f} w={r['w_pr_mean']:.2f}")


# ---------------------------------------------------------------------------
# Snapshots + visualization (inspired by https://distill.pub/2017/feature-visualization/)
# ---------------------------------------------------------------------------

DEFAULT_VIZ_ARMS = "dense,gse_s0.75,topk_k16,both_s0.75_k16,static_s0.75"


def save_snapshot(model, gse, k, snap_dir, step):
    w1 = model.w1.weight.detach().cpu().numpy()
    w2 = model.w2.weight.detach().cpu().numpy()
    if gse is not None:
        m1 = gse.masks[0].cpu().numpy().astype(np.uint8)
        m2 = gse.masks[1].cpu().numpy().astype(np.uint8)
    else:
        m1, m2 = np.ones_like(w1, dtype=np.uint8), np.ones_like(w2, dtype=np.uint8)
    np.savez_compressed(Path(snap_dir) / f"step{step:06d}.npz", step=step, topk=k,
                        w1=w1, b1=model.w1.bias.detach().cpu().numpy(),
                        w2=w2, b2=model.w2.bias.detach().cpu().numpy(), m1=m1, m2=m2)


def cmd_snapshots(args):
    device = pick_device(args.device)
    p = args.probs[0]
    names = [a.strip() for a in (args.arms or DEFAULT_VIZ_ARMS).split(",")]
    table = {name: (kind, s, k) for _, name, kind, s, k in arm_list()}
    fracs = [float(f) for f in args.snap_fracs.split(",")]
    steps = sorted({0} | {max(1, int(args.steps * f)) for f in fracs})
    print(f"device={device} p={p} seed={args.seed} snapshot steps={steps}")
    for name in names:
        kind, s, k = table[name]
        d = Path(args.out_dir) / "snapshots" / f"p{p}_{name}_seed{args.seed}"
        res = train_one(p, kind, s, k, args.seed, args, device, snap_dir=d, snap_steps=steps)
        (d / "metrics.json").write_text(json.dumps(res, indent=2))
        print(f"  {name:18s} norm_loss={res['norm_loss']:.3f} rep={res['n_represented']} "
              f"abl_pr_w={res['abl_pr_w']:.2f} ({res['train_time_s']:.0f}s)", flush=True)
    print("snapshots saved; now run: visualize")


def load_snaps(d):
    return [dict(np.load(f)) for f in sorted(Path(d).glob("step*.npz"))]


def model_from_snap(sn, device):
    m, n = sn["w1"].shape
    k = int(sn["topk"]) or None
    net = ToyNet(n, m, topk=k).to(device)
    with torch.no_grad():
        net.w1.weight.copy_(torch.tensor(sn["w1"]))
        net.w1.bias.copy_(torch.tensor(sn["b1"]))
        net.w2.weight.copy_(torch.tensor(sn["w2"]))
        net.w2.bias.copy_(torch.tensor(sn["b2"]))
    return net.eval()


@torch.no_grad()
def neuron_stats(model, p, n, device, nsamp=50000, nablate=10000):
    gen = torch.Generator(device="cpu").manual_seed(777)
    x = sample(n, p, nsamp, "cpu", gen).to(device)
    t = x.abs()
    y, h = model(x)
    sub = slice(0, nablate)
    hs, ts = h[sub], t[sub]
    yb = model.w2(hs)
    base = ((yb - ts) ** 2).mean(0)
    m = hs.shape[1]
    delta = torch.zeros(m, n, device=device)
    for j in range(m):
        yab = yb - hs[:, j:j + 1] * model.w2.weight[:, j].unsqueeze(0)
        delta[j] = ((yab - ts) ** 2).mean(0) - base
    delta = delta.clamp_min(0)
    return x, h, delta.sum(1), pr(delta)


def optimize_input(model, j, n, device, l1, restarts=8, iters=300):
    """Feature visualization by optimization: find sparse inputs in [-1,1]^n that
    maximize neuron j.  First half maximizes the pre-activation (so dead ReLUs still
    get a gradient), second half the real post-ReLU/top-k activation."""
    torch.manual_seed(j)
    z = (0.1 * torch.randn(restarts, n, device=device)).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    for it in range(iters):
        x = z.clamp(-1, 1)
        if it < iters // 2:
            act = model.w1(x)[:, j]
        else:
            act = model.hidden(x)[:, j]
        loss = -(act.sum()) + l1 * x.abs().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = z.clamp(-1, 1)
        return x.mean(0).cpu().numpy(), float(model.hidden(x)[:, j].mean())


def _orders(w1_final):
    a = np.abs(w1_final)
    feat_of_neuron = a.argmax(1)
    neuron_order = np.lexsort((-a.max(1), feat_of_neuron))
    pos = np.empty_like(neuron_order)
    pos[neuron_order] = np.arange(len(neuron_order))
    feat_order = np.lexsort((-a.max(0), pos[a.argmax(0)]))
    return neuron_order, feat_order


def _bar(ax, vals, title, color="#1f77b4", k=8, fmt="f{}"):
    idx = np.argsort(-np.abs(vals))[:k]
    ax.barh(range(len(idx)), vals[idx][::-1], color=color)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([fmt.format(i) for i in idx[::-1]], fontsize=6)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title(title, fontsize=7)


def cmd_visualize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap

    device = pick_device(args.device)
    p = args.probs[0]
    n = args.n_features
    out = Path(args.out_dir)
    viz = out / "viz"
    viz.mkdir(exist_ok=True)
    names = [a.strip() for a in (args.arms or DEFAULT_VIZ_ARMS).split(",")]
    snaps = {}
    for name in names:
        d = out / "snapshots" / f"p{p}_{name}_seed{args.seed}"
        if d.exists():
            snaps[name] = load_snaps(d)
        else:
            print(f"missing snapshots for {name}; skipping")
    if not snaps:
        raise SystemExit("no snapshots found - run 'snapshots' first")
    names = list(snaps)
    steps = [int(s["step"]) for s in snaps[names[0]]]
    orders = {nm: _orders(snaps[nm][-1]["w1"]) for nm in names}

    # 1. weight heatmaps over training
    fig, axes = plt.subplots(len(names), len(steps), figsize=(2.3 * len(steps), 1.7 * len(names)),
                             squeeze=False)
    for r, nm in enumerate(names):
        no, fo = orders[nm]
        vmax = np.abs(snaps[nm][-1]["w1"]).max()
        for c, sn in enumerate(snaps[nm]):
            ax = axes[r][c]
            ax.imshow(np.abs(sn["w1"])[no][:, fo], aspect="auto", cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"step {int(sn['step'])}", fontsize=8)
            if c == 0:
                ax.set_ylabel(nm, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"|W1| (neurons x features), p={p}.  Neurons and features are sorted so each neuron's main "
                 f"feature lies on the diagonal; weight OFF the diagonal means the neuron also reads other features.", fontsize=9)
    fig.tight_layout()
    fig.savefig(viz / f"heatmaps_p{p}.png", dpi=140)
    plt.close(fig)

    # 2. animation of the same
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 2.8), squeeze=False)
    ims = []
    for ax, nm in zip(axes[0], names):
        no, fo = orders[nm]
        vmax = np.abs(snaps[nm][-1]["w1"]).max()
        ims.append(ax.imshow(np.abs(snaps[nm][0]["w1"])[no][:, fo], aspect="auto", cmap="magma",
                             vmin=0, vmax=vmax))
        ax.set_title(nm, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    ttl = fig.suptitle("", fontsize=9)

    def upd(i):
        for im, nm in zip(ims, names):
            no, fo = orders[nm]
            im.set_data(np.abs(snaps[nm][i]["w1"])[no][:, fo])
        ttl.set_text(f"|W1| during training, p={p}, step {steps[i]}")
        return ims
    ani = FuncAnimation(fig, upd, frames=len(steps), interval=800)
    ani.save(viz / f"training_p{p}.gif", writer=PillowWriter(fps=1.2))
    plt.close(fig)

    # 3. GSE topology (which connections were added/dropped relative to init)
    gnames = [nm for nm in names if nm != "dense" and not nm.startswith("topk")]
    if gnames:
        cmap = ListedColormap(["#ffffff", "#9aa5b1", "#ff7f0e", "#d62728"])
        fig, axes = plt.subplots(len(gnames), len(steps), figsize=(2.3 * len(steps), 1.7 * len(gnames)),
                                 squeeze=False)
        for r, nm in enumerate(gnames):
            no, fo = orders[nm]
            m0 = snaps[nm][0]["m1"].astype(bool)
            for c, sn in enumerate(snaps[nm]):
                m = sn["m1"].astype(bool)
                code = np.zeros(m.shape, dtype=int)
                code[m & m0] = 1       # kept from init
                code[m & ~m0] = 2      # added
                code[~m & m0] = 3      # dropped
                ax = axes[r][c]
                ax.imshow(code[no][:, fo], aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"step {int(sn['step'])}", fontsize=8)
                if c == 0:
                    ax.set_ylabel(nm, fontsize=7, rotation=0, ha="right", va="center")
                if c == len(steps) - 1:
                    ax.set_xlabel(f"{(code == 2).sum() / max(1, m0.sum()):.0%} rewired", fontsize=6)
        fig.suptitle("W1 connections vs initial mask: gray = kept, orange = added, red = dropped. "
                     "The static control never changes.", fontsize=9)
        fig.tight_layout()
        fig.savefig(viz / f"topology_p{p}.png", dpi=140)
        plt.close(fig)

    # 4. per-neuron pages
    for nm in names:
        sn = snaps[nm][-1]
        model = model_from_snap(sn, device)
        x, h, tot, abl = neuron_stats(model, p, n, device)
        ranked = torch.argsort(tot, descending=True)
        rel = (tot / (n * feature_var(p)))[ranked]
        pool = ranked[rel > 0.001].tolist() or ranked.tolist()
        V = min(args.viz_neurons, len(pool))
        head = pool[:min(2, V)]
        rest = pool[len(head):]
        if V > len(head) and rest:
            pick = np.linspace(0, len(rest) - 1, V - len(head)).round().astype(int)
            head += [rest[i] for i in pick]
        order = head
        ncols = 4 if args.actmax else 3
        fig, axes = plt.subplots(len(order), ncols, figsize=(2.4 * ncols + 0.3, 1.55 * len(order)), squeeze=False)
        for r, j in enumerate(order):
            reads = sn["w1"][j]
            writes = sn["w2"][:, j]
            top = torch.topk(h[:, j], max(1, int(0.005 * h.shape[0]))).indices
            enrich = ((x[top] != 0).float().mean(0) / p).cpu().numpy()
            _bar(axes[r][0], reads, f"n{j} reads (W1)\nablation PR={float(abl[j]):.1f}", "#1f77b4")
            _bar(axes[r][1], writes, "writes (W2)", "#2ca02c")
            _bar(axes[r][2], enrich, "top examples: feature enrichment", "#d62728")
            if args.actmax:
                opt_x, opt_act = optimize_input(model, j, n, device, args.actmax_l1)
                _bar(axes[r][3], opt_x, f"optimized input (act {opt_act:.2f})", "#9467bd")
        fig.suptitle(f"{nm}: top-2 most important neurons, then neurons spread across the importance ranking "
                     f"(p={p}).  A monosemantic neuron has one dominant bar in every column.", fontsize=8)
        fig.tight_layout()
        fig.savefig(viz / f"neurons_{nm}_p{p}.png", dpi=130)
        plt.close(fig)
        print(f"  wrote neurons_{nm}_p{p}.png")

    # 5. index page
    parts = ["<html><head><meta charset='utf-8'><title>Sparsity and monosemanticity (toy)</title>",
             "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}"
             "img{max-width:100%;border:1px solid #ddd;margin:.5em 0}</style></head><body>",
             f"<h1>Sparsity and monosemanticity: toy model, p={p}</h1>",
             "<p>Weight heatmaps, rewiring and per-neuron views for each arm, in the spirit of "
             "<a href='https://distill.pub/2017/feature-visualization/'>Feature Visualization</a> "
             "(optimized inputs and dataset examples). For first-layer neurons the optimized input is "
             "close to the thresholded weight row (so it is omitted by default), and the dataset-example and "
             "write-weight views add the independent evidence.</p>",
             f"<h2>Training dynamics</h2><img src='training_p{p}.gif'><img src='heatmaps_p{p}.png'>"]
    if gnames:
        parts.append(f"<h2>Rewiring</h2><img src='topology_p{p}.png'>")
    parts.append("<h2>Neurons</h2>")
    parts += [f"<h3>{nm}</h3><img src='neurons_{nm}_p{p}.png'>" for nm in names]
    parts.append("</body></html>")
    (viz / "index.html").write_text("\n".join(parts))
    print(f"wrote {viz/'index.html'}")


# ---------------------------------------------------------------------------


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "pilot", "analyze", "snapshots", "visualize"])
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-features", type=int, default=128)
    ap.add_argument("--m-hidden", type=int, default=64)
    ap.add_argument("--probs", type=float, nargs="+", default=[0.1, 0.05, 0.2],
                    help="feature probabilities; the first is the primary one and runs first")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--update-freq", type=int, default=100, help="steps between prune-grow rounds")
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="initial fraction of connections replaced per round (cosine to 0)")
    ap.add_argument("--anneal-frac", type=float, default=0.6,
                    help="fraction of training at which alpha reaches 0 and the topology freezes")
    ap.add_argument("--subset-gamma", type=float, default=1.0,
                    help="size of the sampled subset of inactive connections, relative to the number of active ones")
    ap.add_argument("--scope", choices=["global", "local"], default="global",
                    help="rank connections over both weight matrices together (paper) or within each")
    ap.add_argument("--grow-batch", type=int, default=0, help="batch for the grow-step gradient (0 = --batch)")
    ap.add_argument("--eval-samples", type=int, default=50000)
    ap.add_argument("--ablate-samples", type=int, default=10000)
    ap.add_argument("--rep-thresh", type=float, default=0.25,
                    help="a feature counts as represented if its relative error is below this")
    ap.add_argument("--corr-floor", type=float, default=0.005,
                    help="squared correlations below this are treated as noise")
    ap.add_argument("--important-thresh", type=float, default=0.001,
                    help="neuron is 'important' if ablating it raises total error by more than "
                         "this fraction of total variance")
    ap.add_argument("--mono-thresh", type=float, default=1.5,
                    help="ablation PR below this counts as monosemantic")
    ap.add_argument("--max-jobs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="seed for snapshots/visualize")
    ap.add_argument("--snap-fracs", default="0.0025,0.01,0.05,0.25,1.0",
                    help="fractions of --steps at which to save weight snapshots")
    ap.add_argument("--viz-neurons", type=int, default=6, help="neurons per arm in the neuron pages")
    ap.add_argument("--actmax-l1", type=float, default=0.1, help="L1 penalty for optimized inputs")
    ap.add_argument("--actmax", action="store_true",
                    help="add the optimized-input panel (for first-layer neurons it mostly repeats the "
                         "thresholded weight row and saturates at +-1, so it is off by default)")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm names or kinds to run, e.g. 'dense,gse_s0.75,static_s0.75' "
                         "or 'dense,gse' (default: all)")
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    {"run": cmd_run, "pilot": cmd_pilot, "analyze": cmd_analyze,
     "snapshots": cmd_snapshots, "visualize": cmd_visualize}[a.cmd](a)
