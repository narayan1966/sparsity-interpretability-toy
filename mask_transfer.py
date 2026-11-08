#!/usr/bin/env python3
"""
mask_transfer.py - Is it the connectivity pattern that GSE finds, or the process of finding it?

Companion to sparsity_interp_toy.py (same toy model, same metrics, same GSE class).

For each (feature probability p, hidden width m, sparsity S, seed):

  1. Train a network with GSE from scratch (the main experiment's `gse_s{S}` arm).
     Its final connectivity pattern (the two masks) is saved.
  2. Re-train a NEW network from scratch, with the same budget, on a fixed mask:

       gse_s{S}                 the source run (dynamic rewiring; reference)
       gsemask_rewind_s{S}      static training on the final GSE mask, starting from the source
                                run's initial weights (lottery-ticket style).  Connections that GSE
                                added later start at zero, as they did in the source run.
       gsemask_fresh_s{S}       static training on the final GSE mask, with new random weights
       gsemask_degshuf_s{S}     static training on a random mask with the same number of
                                connections at every hidden unit and every feature as the GSE
                                mask, but otherwise random (double-edge-swap shuffle of each
                                matrix).  Isolates the specific wiring from the degree profile.
       static_s{S}              static training on the random mask that GSE started from
                                (the main experiment's static control)

  How to read it.
       gsemask_* ~ gse_s{S} and clearly better than static_s{S}
            -> the readability comes from where the connections are, not from how they were found
       gsemask_* ~ static_s{S}
            -> the search process matters; the final mask alone is not enough
       gsemask_degshuf_s{S} ~ gsemask_fresh_s{S}
            -> only the degree profile of the mask matters, not the specific wiring

  Mask diagnostics are stored with every run: fan-in of the hidden units, fraction of connections
  shared with the initial mask, and the fraction of W1 connections that have a matching W2 connection
  (feature -> unit -> feature loops), which random masks have at rate equal to the W2 density.

Usage
    python mask_transfer.py run --out-dir results_mask --probs 0.01 0.02 0.05 \
        --m-hidden 64 96 128 --seeds 5
    python mask_transfer.py analyze --out-dir results_mask
    # smoke test (about a minute on CPU):
    python mask_transfer.py run --out-dir smoke_mask --probs 0.05 --seeds 1 --steps 600 \
        --ablate-samples 2000 --eval-samples 5000
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import sparsity_interp_toy as base
from sparsity_interp_toy import GSE, ToyNet, compute_metrics, sample

SPARSITIES = [0.5, 0.75]
KINDS = ["gse", "gsemask_rewind", "gsemask_fresh", "gsemask_degshuf", "static"]


def arm_names(sparsities):
    return [f"{k}_s{s}" for s in sparsities for k in KINDS]


def run_path(out_dir, p, m, name, seed):
    return Path(out_dir) / "runs" / f"p{p}_m{m}_{name}_seed{seed}.json"


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def degree_preserving_shuffle(mask, rng, sweeps=20):
    """Randomize a boolean matrix keeping every row sum and column sum (double-edge swaps)."""
    mk = mask.copy()
    rows, cols = np.nonzero(mk)
    n_edges = len(rows)
    if n_edges < 2:
        return mk
    for _ in range(sweeps * n_edges):
        a, b = rng.integers(n_edges, size=2)
        r1, c1, r2, c2 = rows[a], cols[a], rows[b], cols[b]
        if r1 == r2 or c1 == c2 or mk[r1, c2] or mk[r2, c1]:
            continue
        mk[r1, c1] = mk[r2, c2] = False
        mk[r1, c2] = mk[r2, c1] = True
        cols[a], cols[b] = c2, c1
    return mk


def mask_stats(masks, init_masks=None):
    """Diagnostics of (W1 mask [m, n], W2 mask [n, m])."""
    m1, m2 = masks[0].float(), masks[1].float()
    fan_in = m1.sum(1)  # features read by each hidden unit
    fan_out = m1.sum(0)  # hidden units that read each feature
    loops = (m1 * m2.t()).sum()  # W1[j, i] and W2[i, j] both active
    out = dict(
        w1_fanin_mean=float(fan_in.mean()), w1_fanin_sd=float(fan_in.std()),
        w1_fanout_sd=float(fan_out.std()),
        w1_features_uncovered=float((fan_out == 0).float().mean()),
        loop_frac=float(loops / m1.sum().clamp_min(1)),
    )
    if init_masks is not None:
        out["frac_initial_mask_kept"] = sum(int((a & b).sum()) for a, b in zip(init_masks, masks)) \
            / sum(int(a.sum()) for a in init_masks)
    return out


def impose_mask(gse, masks, weights=None, generator=None):
    """Put `masks` into a static GSE object (in place, so its gradient hooks stay valid) and set the
    weights: `weights` if given, else the GSE initialization (uniform, unit pre-activation variance
    for the actual fan-in)."""
    for i, (lin, mk_old, mk) in enumerate(zip(gse.layers, gse.masks, masks)):
        mk_old.copy_(mk)
        if weights is None:
            fan = mk.sum(1, keepdim=True).clamp_min(1).float()
            bound = math.sqrt(3.0) / torch.sqrt(fan)
            w = (torch.rand(lin.weight.shape, generator=generator, device=mk.device) * 2 - 1) * bound
        else:
            w = weights[i]
        lin.weight.data.copy_(w * mk)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_loop(model, gse, p, args, device, gen):
    n = args.n_features
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.01)
    gb = args.grow_batch or args.batch

    def grad_fn():
        xg = sample(n, p, gb, device, gen)
        yg, _ = model(xg)
        ((yg - xg.abs()) ** 2).mean().backward()

    swaps = 0
    for _ in range(args.steps):
        x = sample(n, p, args.batch, device, gen)
        y, _ = model(x)
        loss = ((y - x.abs()) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        swaps += gse.after_step(opt, grad_fn)
    return swaps


def new_net(p, m, seed, s, args, device, dynamic, tag=0):
    """Model + GSE object.  tag=0 reproduces the main experiment's seeding exactly, so the source
    gse run and the static control are the same runs as `gse_s{S}` / `static_s{S}` there."""
    torch.manual_seed(seed + 10_000 * tag)
    gen = torch.Generator(device=device).manual_seed(1000 + seed + 10_000 * tag)
    model = ToyNet(args.n_features, m).to(device)
    gse = GSE([model.w1, model.w2], s, args.steps, update_freq=args.update_freq, alpha=args.alpha,
              anneal_frac=args.anneal_frac, gamma=args.subset_gamma, scope=args.scope, dynamic=dynamic)
    return model, gse, gen


def finish(model, gse, p, args, device, init_masks, swaps, extra=None):
    res = compute_metrics(model, p, args.n_features, device, args)
    nnz, stray, dens, sp = gse.check()
    assert stray == 0, f"{stray} weights outside the mask"
    if gse.scope == "global":
        assert nnz == gse.n_total, "active connections drifted"
    else:
        assert [int(mk.sum()) for mk in gse.masks] == gse.n_layer, "per-layer connection count drifted"
    res.update(mask_sparsity=sp, density_w1=dens[0], density_w2=dens[1], total_swaps=swaps,
               gse_rounds=gse.rounds,
               weight_sparsity=float((torch.cat([model.w1.weight.flatten(),
                                                 model.w2.weight.flatten()]) == 0).float().mean()))
    res.update(mask_stats(gse.masks, init_masks))
    if extra:
        res.update(extra)
    gse.remove_hooks()
    return res


def run_cell(p, m, seed, s, args, device):
    """All five arms for one (p, m, sparsity, seed).  Returns {arm name: result}."""
    out = {}
    rng = np.random.default_rng(777 + seed)

    def record(kind, res, t0):
        name = f"{kind}_s{s}"
        res.update(p=p, m=m, seed=seed, arm=name, kind=kind, sparsity=s, train_time_s=time.time() - t0)
        out[name] = res

    # 1. source: GSE from scratch (identical to the main experiment's gse arm)
    t0 = time.time()
    model, gse, gen = new_net(p, m, seed, s, args, device, dynamic=True)
    init_masks = [mk.clone() for mk in gse.masks]
    init_w = [lin.weight.data.clone() for lin in gse.layers]
    swaps = train_loop(model, gse, p, args, device, gen)
    final_masks = [mk.clone() for mk in gse.masks]
    mdir = Path(args.out_dir) / "masks"
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save({"final": [mk.cpu() for mk in final_masks], "initial": [mk.cpu() for mk in init_masks]},
               mdir / f"p{p}_m{m}_s{s}_seed{seed}.pt")
    record("gse", finish(model, gse, p, args, device, init_masks, swaps), t0)

    # 2. rewind: static training on the final mask from the source run's initial weights
    t0 = time.time()
    model, gse, gen = new_net(p, m, seed, s, args, device, dynamic=False)
    impose_mask(gse, final_masks, weights=init_w)
    swaps = train_loop(model, gse, p, args, device, gen)
    record("gsemask_rewind", finish(model, gse, p, args, device, init_masks, swaps), t0)

    # 3. fresh: static training on the final mask with new random weights
    t0 = time.time()
    model, gse, gen = new_net(p, m, seed, s, args, device, dynamic=False, tag=1)
    impose_mask(gse, final_masks, generator=gen)
    swaps = train_loop(model, gse, p, args, device, gen)
    record("gsemask_fresh", finish(model, gse, p, args, device, init_masks, swaps), t0)

    # 4. degree-preserving shuffle of the final mask, new random weights
    t0 = time.time()
    model, gse, gen = new_net(p, m, seed, s, args, device, dynamic=False, tag=2)
    shuffled = [torch.from_numpy(degree_preserving_shuffle(mk.cpu().numpy(), rng)).to(device)
                for mk in final_masks]
    impose_mask(gse, shuffled, generator=gen)
    swaps = train_loop(model, gse, p, args, device, gen)
    record("gsemask_degshuf", finish(model, gse, p, args, device, init_masks, swaps), t0)

    # 5. static control: the random mask GSE started from (identical to the main static arm)
    t0 = time.time()
    model, gse, gen = new_net(p, m, seed, s, args, device, dynamic=False)
    swaps = train_loop(model, gse, p, args, device, gen)
    record("static", finish(model, gse, p, args, device, init_masks, swaps), t0)
    return out


def cmd_run(args):
    device = base.pick_device(args.device)
    print(f"device={device} probs={args.probs} m={args.m_hidden} sparsities={args.sparsities} "
          f"seeds={args.seeds} steps={args.steps} scope={args.scope}", flush=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    names = arm_names(args.sparsities)
    jobs = [(p, m, s, seed) for seed in range(args.seeds) for p in args.probs
            for m in args.m_hidden for s in args.sparsities]
    done = 0
    for p, m, s, seed in jobs:
        if all(run_path(args.out_dir, p, m, nm, seed).exists() for nm in names if nm.endswith(f"_s{s}")):
            continue
        t0 = time.time()
        r = run_cell(p, m, seed, s, args, device)
        for nm, res in r.items():
            path = run_path(args.out_dir, p, m, nm, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res))
        done += 1
        print(f"[{done}/{len(jobs)}] p={p} m={m} s={s} seed={seed} {time.time() - t0:.0f}s | "
              + " | ".join(f"{k} {r[f'{k}_s{s}']['n_represented']}/{r[f'{k}_s{s}']['abl_pr_w']:.2f}"
                           for k in KINDS), flush=True)
    print("done")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

DIAG = ["frac_initial_mask_kept", "loop_frac", "w1_fanin_sd", "w1_fanout_sd", "w1_features_uncovered"]


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
    metrics = list(base.METRICS) + DIAG
    groups = {}
    for r in runs:
        groups.setdefault((r["p"], r["m"], r["sparsity"], r["kind"]), []).append(r)
    order = {k: i for i, k in enumerate(KINDS)}
    summary = []
    for (p, m, s, kind), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], order[kv[0][3]])):
        row = dict(p=p, m=m, sparsity=s, arm=f"{kind}_s{s}", kind=kind, n=len(rs))
        for mn in metrics:
            vals = np.array([r.get(mn, np.nan) for r in rs], dtype=float)
            ok = not np.all(np.isnan(vals))
            row[mn + "_mean"] = float(np.nanmean(vals)) if ok else float("nan")
            row[mn + "_sd"] = float(np.nanstd(vals)) if ok else float("nan")
        summary.append(row)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\nfeatures represented / features disturbed per neuron / clean-neuron fraction (mean over seeds)")
    print("d_* = paired difference to the source gse run (same seed); loop = W1 connections with a W2 partner")
    by_seed = {(r["p"], r["m"], r["sparsity"], r["kind"], r["seed"]): r for r in runs}
    last = None
    for r in summary:
        key = (r["p"], r["m"], r["sparsity"])
        if key != last:
            last = key
            print(f"\np={r['p']} m={r['m']} sparsity={r['sparsity']}")
        diffs = []
        for mn in ("n_represented", "abl_pr_w", "abl_mono_frac"):
            d = [by_seed[(r["p"], r["m"], r["sparsity"], r["kind"], sd)][mn]
                 - by_seed[(r["p"], r["m"], r["sparsity"], "gse", sd)][mn]
                 for sd in range(10_000)
                 if (r["p"], r["m"], r["sparsity"], r["kind"], sd) in by_seed
                 and (r["p"], r["m"], r["sparsity"], "gse", sd) in by_seed]
            diffs.append(np.mean(d) if d else float("nan"))
        print(f"  {r['kind']:16s} n={r['n']} norm_loss={r['norm_loss_mean']:.3f} rep={r['n_represented_mean']:.1f}"
              f" abl={r['abl_pr_w_mean']:.2f} clean={r['abl_mono_frac_mean']:.2f}"
              f" | d_rep={diffs[0]:+.1f} d_abl={diffs[1]:+.2f} d_clean={diffs[2]:+.2f}"
              f" | loop={r['loop_frac_mean']:.2f} kept={r['frac_initial_mask_kept_mean']:.2f}")
    print(f"\nwrote {out/'summary.csv'} and {out/'all_runs.csv'}")


def build_parser():
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--out-dir", default="results_mask")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--probs", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    ap.add_argument("--m-hidden", type=int, nargs="+", default=[64])
    ap.add_argument("--sparsities", type=float, nargs="+", default=SPARSITIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000, help="training steps of every arm")
    ap.add_argument("--scope", choices=["global", "local"], default="local",
                    help="GSE ranking: per matrix (main results) or across both matrices")
    return ap


def parse_args():
    args, _ = build_parser().parse_known_args()
    # every metric / GSE option not defined here takes the main script's default
    defaults = base.build_parser().parse_args(["run"])
    for k, v in vars(defaults).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    # ... and can still be overridden on the command line (e.g. --lr, --batch, --ablate-samples)
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for k, v in vars(defaults).items():
        if k in ("cmd", "out_dir", "device", "probs", "seeds", "steps", "m_hidden", "scope"):
            continue
        if isinstance(v, bool):
            continue
        extra.add_argument("--" + k.replace("_", "-"), type=type(v) if v is not None else str, default=v)
    ov, _ = extra.parse_known_args()
    for k, v in vars(ov).items():
        setattr(args, k, v)
    return args


if __name__ == "__main__":
    a = parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)
