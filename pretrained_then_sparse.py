#!/usr/bin/env python3
"""
pretrained_then_sparse.py - Does sparsifying an already-trained dense network make its
neurons easier to read?

Companion to sparsity_interp_toy.py (same toy model, same metrics, same GSE class).
There, sparse networks are trained from scratch.  Here every run starts from a trained
dense network and asks what happens when sparsity is imposed afterwards.

For each (feature probability p, hidden width m, seed):

  1. Train a dense network (same recipe as the main experiment: Adam, lr 3e-3, cosine
     schedule, --steps steps).  Cached in <out>/pre/.
  2. From that one network, run these arms (S = target sparsity):

       dense_pre          the trained dense network, no further training (starting point)
       dense_ft           keep training the dense network for --ft-steps (control for the
                          extra training time that the sparse arms receive)
       mag_s{S}_oneshot   prune to S by weight magnitude (inside each matrix), no retraining
       mag_s{S}_static    the same magnitude mask, then fine-tune with the mask fixed
       mag_s{S}_gse       the same magnitude mask as the starting topology, then fine-tune
                          with unstructured GSE rewiring (grow by gradient, prune by
                          magnitude; alpha cosine --alpha -> 0 at --anneal-frac)

     The mask is the same for the oneshot, static and GSE arms, so the comparison between
     them isolates what fine-tuning and rewiring add.

  The from-scratch results of the main experiment (static_s{S}, gse_s{S}) are the natural
  reference; `analyze --ref-summary results_u_local/summary.csv` prints them next to these.
  Note the budgets differ: from scratch = --steps steps, here = --steps + --ft-steps.

Usage
    python pretrained_then_sparse.py run --out-dir results_pre --probs 0.01 0.02 0.05 \
        --m-hidden 64 --seeds 5
    python pretrained_then_sparse.py run --out-dir results_pre --probs 0.01 --m-hidden 128 --seeds 5
    python pretrained_then_sparse.py analyze --out-dir results_pre \
        --ref-summary results_u_local/summary.csv
    # smoke test (about a minute on CPU):
    python pretrained_then_sparse.py run --out-dir smoke_pre --probs 0.05 --seeds 1 \
        --steps 600 --ft-steps 400 --ablate-samples 2000 --eval-samples 5000
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import sparsity_interp_toy as base
from sparsity_interp_toy import GSE, ToyNet, compute_metrics, sample

SPARSITIES = [0.5, 0.75]


def arm_names(sparsities):
    names = ["dense_pre", "dense_ft"]
    for s in sparsities:
        names += [f"mag_s{s}_oneshot", f"mag_s{s}_static", f"mag_s{s}_gse"]
    return names


def run_path(out_dir, p, m, name, seed):
    return Path(out_dir) / "runs" / f"p{p}_m{m}_{name}_seed{seed}.json"


def train_loop(model, gse, p, args, device, steps, lr, seed_gen):
    """Adam + cosine schedule; if gse is given, its masks are enforced and (when dynamic) rewired."""
    n = args.n_features
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    grad_fn = None
    if gse is not None:
        gb = args.grow_batch or args.batch

        def grad_fn():
            xg = sample(n, p, gb, device, seed_gen)
            yg, _ = model(xg)
            ((yg - xg.abs()) ** 2).mean().backward()
    swaps = 0
    for _ in range(steps):
        x = sample(n, p, args.batch, device, seed_gen)
        y, _ = model(x)
        loss = ((y - x.abs()) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if gse is not None:
            swaps += gse.after_step(opt, grad_fn)
    return swaps


def pretrain(p, m, seed, args, device):
    cache = Path(args.out_dir) / "pre" / f"p{p}_m{m}_seed{seed}.pt"
    if cache.exists():
        return torch.load(cache, map_location=device)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(1000 + seed)
    model = ToyNet(args.n_features, m).to(device)
    train_loop(model, None, p, args, device, args.steps, args.lr, gen)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, cache)
    return sd


def make_gse(model, pre_sd, sparsity, args, dynamic):
    """GSE object whose initial masks are the per-matrix magnitude masks of the pretrained
    network and whose weights are the surviving pretrained weights."""
    gse = GSE([model.w1, model.w2], sparsity, args.ft_steps, update_freq=args.update_freq,
              alpha=args.alpha, anneal_frac=args.anneal_frac, gamma=args.subset_gamma,
              scope="local", dynamic=dynamic)
    for i, (lin, key) in enumerate(zip(gse.layers, ("w1.weight", "w2.weight"))):
        w = pre_sd[key].to(lin.weight.device)
        keep = torch.zeros(w.numel(), dtype=torch.bool, device=w.device)
        keep[w.abs().flatten().topk(gse.n_layer[i]).indices] = True
        keep = keep.view_as(w)
        gse.masks[i].copy_(keep)
        lin.weight.data.copy_(w * keep)
    return gse


def evaluate(model, p, args, device):
    res = compute_metrics(model, p, args.n_features, device, args)
    res["weight_sparsity"] = float((torch.cat([model.w1.weight.flatten(),
                                               model.w2.weight.flatten()]) == 0).float().mean())
    return res


def run_seed(p, m, seed, args, device):
    out = {}
    pre_sd = pretrain(p, m, seed, args, device)

    def fresh():
        model = ToyNet(args.n_features, m).to(device)
        model.load_state_dict(pre_sd)
        return model

    def gen_for(tag):
        return torch.Generator(device=device).manual_seed(5000 + 17 * seed + tag)

    def record(name, kind, s, res, t0):
        res.update(p=p, m=m, seed=seed, arm=name, kind=kind, sparsity=s, train_time_s=time.time() - t0)
        out[name] = res

    t0 = time.time()
    record("dense_pre", "dense_pre", None, evaluate(fresh(), p, args, device), t0)

    t0 = time.time()
    model = fresh()
    train_loop(model, None, p, args, device, args.ft_steps, args.ft_lr, gen_for(0))
    record("dense_ft", "dense_ft", None, evaluate(model, p, args, device), t0)

    for si, s in enumerate(args.sparsities):
        # one-shot magnitude pruning, no retraining
        t0 = time.time()
        model = fresh()
        g = make_gse(model, pre_sd, s, args, dynamic=False)
        g.remove_hooks()
        record(f"mag_s{s}_oneshot", "oneshot", s, evaluate(model, p, args, device), t0)

        for kind, dynamic in (("static", False), ("gse", True)):
            t0 = time.time()
            model = fresh()
            g = make_gse(model, pre_sd, s, args, dynamic=dynamic)
            init_masks = [mk.clone() for mk in g.masks]
            swaps = train_loop(model, g, p, args, device, args.ft_steps, args.ft_lr, gen_for(1 + 2 * si + dynamic))
            res = evaluate(model, p, args, device)
            nnz, stray, dens, sp = g.check()
            assert stray == 0, f"{stray} weights outside the mask"
            assert [int(mk.sum()) for mk in g.masks] == g.n_layer, "per-layer connection count drifted"
            kept = sum(int((a & b).sum()) for a, b in zip(init_masks, g.masks)) / sum(g.n_layer)
            res.update(mask_sparsity=sp, density_w1=dens[0], density_w2=dens[1],
                       frac_initial_mask_kept=kept, total_swaps=swaps, gse_rounds=g.rounds)
            g.remove_hooks()
            record(f"mag_s{s}_{kind}", kind, s, res, t0)
    return out


def cmd_run(args):
    device = base.pick_device(args.device)
    print(f"device={device} probs={args.probs} m={args.m_hidden} seeds={args.seeds} "
          f"steps={args.steps} ft_steps={args.ft_steps} sparsities={args.sparsities}", flush=True)
    names = arm_names(args.sparsities)
    jobs = [(p, m, seed) for seed in range(args.seeds) for p in args.probs for m in [args.m_hidden]]
    done = 0
    for p, m, seed in jobs:
        if all(run_path(args.out_dir, p, m, nm, seed).exists() for nm in names):
            continue
        t0 = time.time()
        results = run_seed(p, m, seed, args, device)
        for nm, res in results.items():
            path = run_path(args.out_dir, p, m, nm, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res))
        done += 1
        r = results
        print(f"[{done}] p={p} m={m} seed={seed} {time.time() - t0:.0f}s | "
              f"dense_ft rep={r['dense_ft']['n_represented']} abl={r['dense_ft']['abl_pr_w']:.2f} | "
              + " | ".join(f"s{s}: static {r[f'mag_s{s}_static']['n_represented']}/"
                           f"{r[f'mag_s{s}_static']['abl_pr_w']:.2f} gse {r[f'mag_s{s}_gse']['n_represented']}/"
                           f"{r[f'mag_s{s}_gse']['abl_pr_w']:.2f}" for s in args.sparsities), flush=True)
    print("done")


def cmd_analyze(args):
    out = Path(args.out_dir)
    runs = [json.loads(f.read_text()) for f in sorted((out / "runs").glob("*.json"))]
    if not runs:
        raise SystemExit("no runs found")
    keys = sorted({k for r in runs for k in r})
    with open(out / "all_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(runs)
    metrics = list(base.METRICS) + ["frac_initial_mask_kept"]
    groups = {}
    for r in runs:
        groups.setdefault((r["p"], r["m"], r["arm"]), []).append(r)
    order = {nm: i for i, nm in enumerate(arm_names(sorted({r["sparsity"] for r in runs if r["sparsity"]})))}
    summary = []
    for (p, m, arm), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], order.get(kv[0][2], 99))):
        row = dict(p=p, m=m, arm=arm, kind=rs[0]["kind"], sparsity=rs[0]["sparsity"], n=len(rs))
        for mn in metrics:
            vals = np.array([r.get(mn, np.nan) for r in rs], dtype=float)
            row[mn + "_mean"] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
            row[mn + "_sd"] = float(np.nanstd(vals)) if not np.all(np.isnan(vals)) else float("nan")
        summary.append(row)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    ref = {}
    if args.ref_summary:
        with open(args.ref_summary) as f:
            for row in csv.DictReader(f):
                ref[(float(row["p"]), row["arm"])] = row
    print("\nfeatures represented / features disturbed per neuron / clean-neuron fraction (mean over seeds)")
    last = None
    for r in summary:
        if (r["p"], r["m"]) != last:
            last = (r["p"], r["m"])
            print(f"\np={r['p']} m={r['m']}")
        extra = ""
        if r["kind"] in ("static", "gse") and not np.isnan(r["frac_initial_mask_kept_mean"]):
            extra = f"  initial mask kept {r['frac_initial_mask_kept_mean']:.2f}"
        print(f"  {r['arm']:18s} n={r['n']} norm_loss={r['norm_loss_mean']:.3f} rep={r['n_represented_mean']:.1f}"
              f" abl={r['abl_pr_w_mean']:.2f} clean={r['abl_mono_frac_mean']:.2f}{extra}")
        if r["sparsity"] and r["kind"] in ("static", "gse"):
            key = (r["p"], f"{r['kind']}_s{r['sparsity']}")
            if key in ref:
                q = ref[key]
                print(f"  {'  from scratch (ref)':18s}         norm_loss={float(q['norm_loss_mean']):.3f} "
                      f"rep={float(q['n_represented_mean']):.1f} abl={float(q['abl_pr_w_mean']):.2f} "
                      f"clean={float(q['abl_mono_frac_mean']):.2f}")
    print(f"\nwrote {out/'summary.csv'} and {out/'all_runs.csv'}")


def build_parser():
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--out-dir", default="results_pre")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--probs", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    ap.add_argument("--m-hidden", type=int, default=64)
    ap.add_argument("--sparsities", type=float, nargs="+", default=SPARSITIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000, help="steps for the dense pretraining")
    ap.add_argument("--ft-steps", type=int, default=10000, help="fine-tuning steps for every arm")
    ap.add_argument("--ft-lr", type=float, default=1e-3, help="peak lr of fine-tuning (cosine to 1%%)")
    ap.add_argument("--ref-summary", default="", help="summary.csv of the from-scratch run, for comparison")
    return ap


def parse_args():
    args, _ = build_parser().parse_known_args()
    # every metric / GSE option not defined here takes the main script's default
    defaults = base.build_parser().parse_args(["run"])
    for k, v in vars(defaults).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    # allow overriding those defaults on the command line (e.g. --lr, --batch, --ablate-samples)
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for k, v in vars(defaults).items():
        if k in ("cmd", "out_dir", "device", "probs", "seeds", "steps", "m_hidden", "scope"):
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            continue
        extra.add_argument(flag, type=type(v) if v is not None else str, default=v)
    ov, _ = extra.parse_known_args()
    for k, v in vars(ov).items():
        setattr(args, k, v)
    return args


if __name__ == "__main__":
    a = parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)
