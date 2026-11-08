#!/usr/bin/env python3
"""
clustered_features_toy.py - Does sparsity clean neurons by removing unrelated mixing
(superposition) or by splitting groups of related features (generalization)?

Companion to sparsity_interp_toy.py and pretrained_then_sparse.py (same GSE class, same
local per-matrix ranking, same recipe).  It changes the data, not the method.

Data
    n features, grouped into G = n / fam_size families of related features.
    --conds "cooc:famout,..."  e.g. "0.5:1,0:1"
      cooc > 0  a family is active with probability q = p / cooc and, when it is, each of its
                members is active with probability cooc, so members co-occur.  The marginal
                probability of every feature is still p.  cooc = 0 gives independent features
                (family active with probability 1, each member with probability p).
      famout 1  the network also has one output per family, the sum of |x_i| over its members,
                so that a neuron which responds to any member of a family has a job to do
                (a reason for related features to share a neuron).  famout 0 omits these.
    Targets: |x_i| for every feature (plus the family sums).  The loss weights every output by
    1 / its variance, so features and families count equally.

Neuron types (important neurons only; ablation of one neuron at a time)
    clean    the features whose error rises number about one          (feature PR  < 1.5)
    grouped  several features, but all of them in one family          (family PR   < 1.5)
    mixed    features from more than one family                       (family PR  >= 1.5)
    Superposition predicts "mixed" neurons, generalization predicts "grouped" ones.

Arms
    from scratch     dense | gse_s{S} | static_s{S} | topk_k{K} | both_s{S}_k{K}
    from the trained dense network (the "dense" arm above), each for --ft-steps more steps:
        pre_dense_ft | pre_oneshot_s{S} | pre_static_s{S} | pre_gse_s{S}
        pre_topk_oneshot_k{K} | pre_topk_k{K} | pre_both_s{S}_k{K}
    The pruned arms use the magnitude mask inside each matrix as the starting topology.

Usage
    python clustered_features_toy.py run --out-dir results_cluster --probs 0.01 0.02 --seeds 5
    python clustered_features_toy.py run ... --shard 0/2   # and --shard 1/2 in a second process
    python clustered_features_toy.py analyze --out-dir results_cluster
    # smoke test (about a minute on CPU):
    python clustered_features_toy.py run --out-dir smoke_cl --probs 0.05 --seeds 1 --conds 0.5:1 \
        --sparsities 0.75 --ks 16 --steps 300 --ft-steps 200 --eval-samples 5000 --ablate-samples 1000
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sparsity_interp_toy as base
from sparsity_interp_toy import GSE, pr
from pretrained_then_sparse import make_gse


# ---------------------------------------------------------------------------
# Model and data
# ---------------------------------------------------------------------------


class Net(nn.Module):
    def __init__(self, n, m, out, topk=None):
        super().__init__()
        self.w1 = nn.Linear(n, m)
        self.w2 = nn.Linear(m, out)
        self.topk = topk
        for lin in (self.w1, self.w2):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        h = torch.relu(self.w1(x))
        if self.topk is not None and self.topk < h.shape[1]:
            idx = h.topk(self.topk, dim=1).indices
            h = h * torch.zeros_like(h).scatter_(1, idx, 1.0)
        return self.w2(h), h


class Data:
    def __init__(self, n, fam_size, p, cooc, fam_out, device):
        assert n % fam_size == 0
        self.n, self.s, self.G = n, fam_size, n // fam_size
        self.fam_out, self.device = fam_out, device
        if cooc > 0:
            self.r, self.q = cooc, min(1.0, p / cooc)
        else:
            self.r, self.q = p, 1.0
        self.out = n + (self.G if fam_out else 0)
        g = torch.Generator().manual_seed(777)
        self.var = self.targets(self.sample(400000, g, "cpu")).var(0).to(device)

    def sample(self, batch, gen, device=None):
        dev = device or self.device
        fam = torch.rand(batch, self.G, device=dev, generator=gen) < self.q
        mem = torch.rand(batch, self.n, device=dev, generator=gen) < self.r
        act = mem & fam.repeat_interleave(self.s, dim=1)
        val = 2 * torch.rand(batch, self.n, device=dev, generator=gen) - 1
        return act.float() * val

    def targets(self, x):
        t = x.abs()
        if self.fam_out:
            t = torch.cat([t, t.reshape(-1, self.G, self.s).sum(2)], 1)
        return t

    def loss(self, y, t):
        return (((y - t) ** 2) / self.var).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def metrics(model, data, args, device):
    n, G, s = data.n, data.G, data.s
    gen = torch.Generator().manual_seed(12345)
    x = data.sample(args.eval_samples, gen, "cpu").to(device)
    t = data.targets(x)
    y, h = model(x)
    nerr = ((y - t) ** 2).mean(0) / data.var
    res = {"norm_loss": float(nerr[:n].mean()), "n_represented": int((nerr[:n] < args.rep_thresh).sum())}
    if data.fam_out:
        res["norm_loss_fam"] = float(nerr[n:].mean())
        res["n_fam_represented"] = int((nerr[n:] < args.rep_thresh).sum())
    live = h.std(0) > 1e-6
    res["n_live_neurons"] = int(live.sum())
    sub = slice(0, args.ablate_samples)
    hs, ts = h[sub], t[sub]
    ybase = model.w2(hs)
    base_err = ((ybase - ts) ** 2).mean(0)
    w2 = model.w2.weight.data
    m = hs.shape[1]
    delta = torch.zeros(m, data.out, device=device)
    for j in range(m):
        yab = ybase - hs[:, j:j + 1] * w2[:, j].unsqueeze(0)
        delta[j] = ((yab - ts) ** 2).mean(0) - base_err
    dn = (delta / data.var).clamp_min(0)
    important = (dn.sum(1) / data.out > args.important_thresh) & live
    res["n_important"] = int(important.sum())
    nan = float("nan")
    if not important.any():
        res.update(abl_pr_w=nan, fam_pr_w=nan, top_fam_share_w=nan, frac_clean=nan, frac_grouped=nan, frac_mixed=nan)
        return res
    df = dn[:, :n]
    feat_pr = pr(df)
    fam = df.reshape(m, G, s).sum(2)
    fam_pr = pr(fam)
    top = fam.max(1).values / (fam.sum(1) + 1e-12)
    wts = df.sum(1)[important]
    for name, v in (("abl_pr_w", feat_pr), ("fam_pr_w", fam_pr), ("top_fam_share_w", top)):
        res[name] = float((v[important] * wts).sum() / wts.sum())
    clean = feat_pr < args.mono_thresh
    grouped = (~clean) & (fam_pr < args.mono_thresh)
    mixed = (~clean) & (fam_pr >= args.mono_thresh)
    k = important.sum()
    res["frac_clean"] = float((clean & important).sum() / k)
    res["frac_grouped"] = float((grouped & important).sum() / k)
    res["frac_mixed"] = float((mixed & important).sum() / k)
    return res


METRICS = ["norm_loss", "n_represented", "norm_loss_fam", "n_fam_represented", "n_important", "abl_pr_w", "fam_pr_w",
           "top_fam_share_w", "frac_clean", "frac_grouped", "frac_mixed", "weight_sparsity", "frac_initial_mask_kept"]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(model, gse, data, args, device, steps, lr, gen):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    grad_fn = None
    if gse is not None:
        gb = args.grow_batch or args.batch

        def grad_fn():
            xg = data.sample(gb, gen)
            yg, _ = model(xg)
            data.loss(yg, data.targets(xg)).backward()
    swaps = 0
    for _ in range(steps):
        x = data.sample(args.batch, gen)
        y, _ = model(x)
        loss = data.loss(y, data.targets(x))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if gse is not None:
            swaps += gse.after_step(opt, grad_fn)
    return swaps


def finish(model, gse, data, args, device, init_masks=None):
    res = metrics(model, data, args, device)
    res["weight_sparsity"] = float((torch.cat([model.w1.weight.flatten(), model.w2.weight.flatten()]) == 0).float().mean())
    if gse is not None:
        nnz, stray, dens, sp = gse.check()
        assert stray == 0, f"{stray} weights outside the mask"
        assert [int(mk.sum()) for mk in gse.masks] == gse.n_layer, "per-layer connection count drifted"
        res.update(mask_sparsity=sp, density_w1=dens[0], density_w2=dens[1], gse_rounds=gse.rounds)
        if init_masks is not None:
            res["frac_initial_mask_kept"] = sum(int((a & b).sum()) for a, b in zip(init_masks, gse.masks)) / sum(gse.n_layer)
        gse.remove_hooks()
    return res


def arm_specs(args):
    """(name, family, sparsity, k) in run order; the first one is the dense network."""
    a = [("dense", "dense", None, None)]
    a += [(f"gse_s{s}", "gse", s, None) for s in args.sparsities]
    a += [(f"static_s{s}", "static", s, None) for s in args.sparsities]
    a += [(f"topk_k{k}", "topk", None, k) for k in args.ks]
    a += [(f"both_s{s}_k{k}", "both", s, k) for s in args.sparsities for k in args.ks]
    a += [("pre_dense_ft", "pre_dense_ft", None, None)]
    a += [(f"pre_oneshot_s{s}", "pre_oneshot", s, None) for s in args.sparsities]
    a += [(f"pre_static_s{s}", "pre_static", s, None) for s in args.sparsities]
    a += [(f"pre_gse_s{s}", "pre_gse", s, None) for s in args.sparsities]
    a += [(f"pre_topk_oneshot_k{k}", "pre_topk_oneshot", None, k) for k in args.ks]
    a += [(f"pre_topk_k{k}", "pre_topk", None, k) for k in args.ks]
    a += [(f"pre_both_s{s}_k{k}", "pre_both", s, k) for s in args.sparsities for k in args.ks]
    return a


def cond_name(cooc, fam_out):
    return f"cooc{cooc}_fam{int(fam_out)}"


def run_path(out_dir, cond, p, m, name, seed):
    return Path(out_dir) / "runs" / f"{cond_name(*cond)}_p{p}_m{m}_{name}_seed{seed}.json"


def run_job(cond, p, m, seed, args, device):
    cooc, fam_out = cond
    n = args.n_features
    data = Data(n, args.fam_size, p, cooc, fam_out, device)
    specs = arm_specs(args)
    todo = [sp for sp in specs if not run_path(args.out_dir, cond, p, m, sp[0], seed).exists()]
    if not todo:
        return 0

    def gen_for(tag):
        return torch.Generator(device=device).manual_seed(1000 * (tag + 1) + seed)

    def save(name, family, s, k, res, t0):
        res.update(cooc=cooc, fam_out=int(fam_out), p=p, m=m, seed=seed, arm=name, kind=family, sparsity=s, k=k,
                   train_time_s=time.time() - t0, q=data.q, r=data.r)
        path = run_path(args.out_dir, cond, p, m, name, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res))

    # the dense network trained from scratch is both an arm and the start of the "pre_" arms
    cache = Path(args.out_dir) / "pre" / f"{cond_name(*cond)}_p{p}_m{m}_seed{seed}.pt"
    t0 = time.time()
    if cache.exists():
        sd = torch.load(cache, map_location=device)
    else:
        torch.manual_seed(seed)
        net = Net(n, m, data.out).to(device)
        train(net, None, data, args, device, args.steps, args.lr, gen_for(0))
        sd = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, cache)
    if any(sp[0] == "dense" for sp in todo):
        net = Net(n, m, data.out).to(device)
        net.load_state_dict(sd)
        save("dense", "dense", None, None, finish(net, None, data, args, device), t0)

    for ti, (name, fam, s, k) in enumerate(todo):
        if name == "dense":
            continue
        t0 = time.time()
        tag = ti + 1
        init_masks = None
        if fam in ("gse", "static", "topk", "both"):
            torch.manual_seed(seed)
            net = Net(n, m, data.out, topk=k if fam in ("topk", "both") else None).to(device)
            g = None
            if fam in ("gse", "static", "both"):
                g = GSE([net.w1, net.w2], s, args.steps, update_freq=args.update_freq, alpha=args.alpha,
                        anneal_frac=args.anneal_frac, gamma=args.subset_gamma, scope="local", dynamic=(fam != "static"))
            train(net, g, data, args, device, args.steps, args.lr, gen_for(tag))
            res = finish(net, g, data, args, device)
        else:
            net = Net(n, m, data.out, topk=k if fam in ("pre_topk", "pre_topk_oneshot", "pre_both") else None).to(device)
            net.load_state_dict(sd)
            g = None
            if fam in ("pre_oneshot", "pre_static", "pre_gse", "pre_both"):
                g = make_gse(net, sd, s, args, dynamic=(fam in ("pre_gse", "pre_both")))
                init_masks = [mk.clone() for mk in g.masks]
            if fam in ("pre_oneshot", "pre_topk_oneshot"):
                if g is not None:
                    g.remove_hooks()
                    g = None
            else:
                train(net, g, data, args, device, args.ft_steps, args.ft_lr, gen_for(tag))
            res = finish(net, g, data, args, device, init_masks)
        save(name, fam, s, k, res, t0)
    return len(todo)


def parse_conds(text):
    out = []
    for item in text.split(","):
        c, f = item.split(":")
        out.append((float(c), bool(int(f))))
    return out


def cmd_run(args):
    device = base.pick_device(args.device)
    conds = parse_conds(args.conds)
    jobs = [(seed, c, p) for seed in range(args.seeds) for c in conds for p in args.probs]
    if args.shard:
        i, nsh = (int(v) for v in args.shard.split("/"))
        jobs = jobs[i::nsh]
    print(f"device={device} conds={conds} probs={args.probs} m={args.m_hidden} seeds={args.seeds} "
          f"arms={len(arm_specs(args))} jobs={len(jobs)}", flush=True)
    for ji, (seed, c, p) in enumerate(jobs, 1):
        t0 = time.time()
        k = run_job(c, p, args.m_hidden, seed, args, device)
        print(f"[{ji}/{len(jobs)}] {cond_name(*c)} p={p} seed={seed}: {k} arms in {time.time() - t0:.0f}s", flush=True)
    print("done")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def cmd_analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out_dir)
    runs = [json.loads(f.read_text()) for f in sorted((out / "runs").glob("*.json"))]
    if not runs:
        raise SystemExit("no runs found")
    keys = sorted({k for r in runs for k in r})
    with open(out / "all_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(runs)
    order = []
    for r in runs:
        if r["arm"] not in order:
            order.append(r["arm"])
    rank = {nm: i for i, nm in enumerate(arm_specs(args))}
    names = [sp[0] for sp in arm_specs(args)]
    groups = {}
    for r in runs:
        groups.setdefault((r["cooc"], r["fam_out"], r["p"], r["m"], r["arm"]), []).append(r)
    summary = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3], names.index(k[4]) if k[4] in names else 99)):
        rs = groups[key]
        row = dict(cooc=key[0], fam_out=key[1], p=key[2], m=key[3], arm=key[4], kind=rs[0]["kind"], n=len(rs))
        for mn in METRICS:
            vals = np.array([r.get(mn, np.nan) for r in rs], dtype=float)
            ok = not np.all(np.isnan(vals))
            row[mn + "_mean"] = float(np.nanmean(vals)) if ok else float("nan")
            row[mn + "_sd"] = float(np.nanstd(vals)) if ok else float("nan")
        summary.append(row)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    last = None
    for r in summary:
        c = (r["cooc"], r["fam_out"], r["p"], r["m"])
        if c != last:
            last = c
            print(f"\ncooc={c[0]} famout={c[1]} p={c[2]} m={c[3]}")
            print(f"  {'arm':24s} n  feats fams  featPR famPR top%  clean grouped mixed")
        print(f"  {r['arm']:24s} {r['n']}  {r['n_represented_mean']:5.1f} {r['n_fam_represented_mean']:4.1f}  "
              f"{r['abl_pr_w_mean']:5.2f}  {r['fam_pr_w_mean']:5.2f} {100 * r['top_fam_share_w_mean']:4.0f}  "
              f"{r['frac_clean_mean']:5.2f} {r['frac_grouped_mean']:7.2f} {r['frac_mixed_mean']:5.2f}")
    # stacked bars of neuron types, one panel per (condition, p)
    panels = sorted({(r["cooc"], r["fam_out"], r["p"], r["m"]) for r in summary})
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(8, 0.5 * len(names)), 3.2 * len(panels)), squeeze=False)
    cols = {"clean": "#2ca02c", "grouped": "#1f77b4", "mixed": "#d62728"}
    for ax, c in zip(axes[:, 0], panels):
        rows = [r for r in summary if (r["cooc"], r["fam_out"], r["p"], r["m"]) == c]
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for kind in ("clean", "grouped", "mixed"):
            v = np.array([r[f"frac_{kind}_mean"] for r in rows])
            ax.bar(x, v, bottom=bottom, color=cols[kind], label=kind)
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels([r["arm"] for r in rows], rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("fraction of neurons")
        ax.set_title(f"co-occurrence {c[0]}, family outputs {c[1]}, p={c[2]}, m={c[3]}", fontsize=9)
    axes[0, 0].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "neuron_types.png", dpi=150)
    print(f"\nwrote {out/'summary.csv'}, {out/'all_runs.csv'} and neuron_types.png")


def build_parser():
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--out-dir", default="results_cluster")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-features", type=int, default=128)
    ap.add_argument("--fam-size", type=int, default=8)
    ap.add_argument("--m-hidden", type=int, default=64)
    ap.add_argument("--probs", type=float, nargs="+", default=[0.01, 0.02])
    ap.add_argument("--conds", default="0.5:1,0:1", help="comma list of cooc:famout")
    ap.add_argument("--sparsities", type=float, nargs="+", default=[0.5, 0.75])
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 8])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--shard", default="", help="i/n: run every n-th job starting at i")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--ft-steps", type=int, default=10000)
    ap.add_argument("--ft-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--update-freq", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--anneal-frac", type=float, default=0.6)
    ap.add_argument("--subset-gamma", type=float, default=1.0)
    ap.add_argument("--grow-batch", type=int, default=0)
    ap.add_argument("--eval-samples", type=int, default=50000)
    ap.add_argument("--ablate-samples", type=int, default=10000)
    ap.add_argument("--rep-thresh", type=float, default=0.25)
    ap.add_argument("--important-thresh", type=float, default=0.001)
    ap.add_argument("--mono-thresh", type=float, default=1.5)
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)
