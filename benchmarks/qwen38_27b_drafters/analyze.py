"""Aggregate collected tables into small JSON files for the report.

    python analyze.py --tables tables --results results --out report/data.json

Acceptance = accepted draft tokens / proposed draft tokens (bonus token excluded).
Every average is over requests (each request weighted equally inside a bin),
with a 95% bootstrap interval over requests.
"""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CTX_EDGES = [0, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000]
GEN_STEP = 2_048
CYCLE_STEP = 1_024  # both detectors run at every checkpoint
rng = np.random.default_rng(0)


def boot(values, n=1000):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None, None, None, 0
    if len(v) == 1:
        return float(v[0]), None, None, 1
    means = rng.choice(v, size=(n, len(v))).mean(axis=1)
    return float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), len(v)


def curve(rounds, by, xcol, edges=None, step=None):
    """Per (group, x-bin): mean over requests of per-request acceptance in that bin."""
    r = rounds[rounds.proposed > 0].copy()
    if edges is not None:
        r["x"] = pd.cut(r[xcol], edges, right=False, labels=edges[:-1]).astype(float)
    else:
        r["x"] = (r[xcol] // step) * step
    per_req = r.groupby(by + ["x", "id"]).agg(a=("accepted", "sum"), p=("proposed", "sum"),
                                              n=("accepted", "size"))
    per_req["acc"] = per_req.a / per_req.p
    per_req["tau"] = per_req.a / per_req.n
    out = []
    for key, g in per_req.groupby(level=list(range(len(by) + 1))):
        m, lo, hi, n = boot(g.acc)
        t, tlo, thi, _ = boot(g.tau)
        out.append(dict(zip(by + ["x"], [k if isinstance(k, str) else float(k) for k in key]),
                        acc=m, lo=lo, hi=hi, tau=t, tau_lo=tlo, tau_hi=thi, n=n,
                        rounds=int(g.n.sum())))
    return out


# ---------- repetition / EOS (noeos panel) ----------

def periodic(window, periods, threshold=0.99):
    best = None
    for p in periods:
        if p >= len(window):
            break
        if (window[p:] == window[:-p]).mean() >= threshold:
            best = p
            break
    return best


def cycle_profile(args):
    """Cycle period at checkpoints every CYCLE_STEP generated tokens (issue11-style detectors)."""
    path, = args
    with gzip.open(path, "rt") as handle:
        d = json.load(handle)
    x = np.array(d["token_ids"], dtype=np.int64)
    points, short, long_ = [], [], []
    for t in range(2048, len(x) + 1, CYCLE_STEP):
        points.append(t)
        short.append(periodic(x[t - 2048:t], range(1, 129)))
        long_.append(periodic(x[max(0, t - 8192):t], range(129, 2049)) if short[-1] is None else None)
    return d["id"], dict(points=points, short=short, long=long_)


def noeos_analysis(tables, results, methods):
    rounds = pd.read_parquet(tables / "rounds.parquet")
    reqs = pd.read_parquet(tables / "requests.parquet")
    out = dict(n_traj=reqs.groupby("method").size().to_dict(),
               curves=curve(rounds, ["method"], "gen", step=GEN_STEP),
               curves_ds=curve(rounds, ["method", "dataset"], "gen", step=GEN_STEP))
    # cycles per method output
    cycles = {}
    for m in methods:
        files = sorted((results / m / "outputs").glob("*.json.gz"))
        with ProcessPoolExecutor(32) as pool:
            cycles[m] = dict(pool.map(cycle_profile, [(f,) for f in files]))
    prev = []
    for m, per in cycles.items():
        for rid, c in per.items():
            for t, s, l in zip(c["points"], c["short"], c["long"]):
                prev.append(dict(method=m, id=rid, t=t, cyc=(s is not None) or (l is not None), period=s or l))
    if not prev:  # no trajectory long enough for the detectors
        prev = [dict(method="ar", id="", t=0, cyc=False, period=None)]
    prev = pd.DataFrame(prev).merge(reqs[["id", "dataset"]].drop_duplicates(), on="id")
    prev["x"] = (prev.t // GEN_STEP) * GEN_STEP
    out["cycle_prevalence"] = (prev.groupby(["method", "x"]).cyc.mean().reset_index()
                               .to_dict(orient="records"))
    out["cycle_prevalence_ds"] = (prev[prev.method == "ar"].groupby(["dataset", "x"]).cyc.mean()
                                  .reset_index().to_dict(orient="records"))
    onset = prev[prev.cyc].groupby(["method", "id"]).t.min().reset_index()
    out["cycle_onset"] = onset.merge(reqs[["id", "dataset"]].drop_duplicates(), on="id").to_dict(orient="records")
    out["cycle_period"] = prev[prev.cyc & (prev.method == "ar")].groupby("id").period.median().dropna().tolist()
    # acceptance inside / outside a detected cycle (round -> latest checkpoint)
    r = rounds[rounds.proposed > 0].copy()
    r["t"] = ((r.gen // CYCLE_STEP) * CYCLE_STEP).clip(lower=2048)
    r = r.merge(prev[["method", "id", "t", "cyc"]], on=["method", "id", "t"], how="left")
    r["cyc"] = r.cyc.fillna(False).astype(bool)
    g = r.groupby(["method", "id", "cyc"]).agg(a=("accepted", "sum"), p=("proposed", "sum")).reset_index()
    g["acc"] = g.a / g.p
    out["acc_by_cycle"] = [dict(method=m, cyc=bool(c), **dict(zip(["acc", "lo", "hi", "n"], boot(x.acc))))
                           for (m, c), x in g.groupby(["method", "cyc"])]
    # AR target logprobs: where the model wanted EOS, and entropy
    eos_rows, first_eos = [], []
    for f in sorted((tables / "ar_logprobs").glob("*.npz")):
        z = np.load(f)
        rank = z["eos_rank"]
        wanted = rank == 1
        idx = np.arange(len(rank))
        b = (idx // GEN_STEP) * GEN_STEP
        df = pd.DataFrame(dict(x=b, wanted=wanted, eos_lp=z["eos"], ent=z["entropy_topk"],
                               margin=z["top1"] - z["top2"]))
        agg = df.groupby("x").agg(wanted=("wanted", "mean"), ent=("ent", "mean"),
                                  eos_p=("eos_lp", lambda s: float(np.nanmean(np.exp(s))))).reset_index()
        agg["id"] = f.stem
        eos_rows.append(agg)
        first_eos.append(dict(id=f.stem, first=int(idx[wanted][0]) if wanted.any() else None,
                              n_wanted=int(wanted.sum()), n=len(rank)))
    if eos_rows:
        e = pd.concat(eos_rows).merge(reqs[["id", "dataset"]].drop_duplicates(), on="id")
        out["eos_wanted"] = e.groupby("x").agg(wanted=("wanted", "mean"), ent=("ent", "mean"),
                                               eos_p=("eos_p", "mean")).reset_index().to_dict(orient="records")
        out["eos_wanted_ds"] = e.groupby(["dataset", "x"]).wanted.mean().reset_index().to_dict(orient="records")
        fe = pd.DataFrame(first_eos).merge(reqs[["id", "dataset"]].drop_duplicates(), on="id")
        out["first_eos"] = fe.to_dict(orient="records")
        # acceptance before / after the first position where the target wanted EOS
        r2 = rounds[rounds.proposed > 0].merge(fe[["id", "first"]], on="id")
        r2["after"] = r2.gen >= r2["first"].fillna(1e12)
        g = r2.groupby(["method", "id", "after"]).agg(a=("accepted", "sum"), p=("proposed", "sum")).reset_index()
        g["acc"] = g.a / g.p
        out["acc_by_eos"] = [dict(method=m, after=bool(a), **dict(zip(["acc", "lo", "hi", "n"], boot(x.acc))))
                             for (m, a), x in g.groupby(["method", "after"])]
        # acceptance vs target entropy (AR text == drafter text when outputs match)
        ent_rows = []
        for f in sorted((tables / "ar_logprobs").glob("*.npz")):
            z = np.load(f)
            ent_rows.append(pd.DataFrame(dict(id=f.stem, gen=np.arange(len(z["entropy_topk"])),
                                              ent=z["entropy_topk"])))
        ent = pd.concat(ent_rows)
        rr = rounds[rounds.proposed > 0].merge(ent, on=["id", "gen"])
        rr["ent_bin"] = pd.cut(rr.ent, [0, 0.01, 0.1, 0.3, 1, 2, 10], right=False).astype(str)
        out["acc_by_entropy"] = (rr.groupby(["method", "ent_bin"]).apply(
            lambda x: x.accepted.sum() / x.proposed.sum()).reset_index(name="acc").to_dict(orient="records"))
    # example tails for the report: one AR trajectory per dataset
    out["examples"] = []
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    except Exception:
        tok = None
    first = {r["id"]: int(r["first"]) for r in out.get("first_eos", []) if r["first"] is not None and r["first"] == r["first"]}
    ar_reqs = reqs[reqs.method == "ar"].sort_values("id")
    for ds, g in ar_reqs.groupby("dataset"):
        if tok is None:
            break
        rid = g.id.iloc[0]
        with gzip.open(results / "ar" / "outputs" / f"{rid}.json.gz", "rt") as handle:
            ids = json.load(handle)["token_ids"]
        fe = first.get(rid)
        starts = ([max(0, fe - 120)] if fe is not None else []) + [8192, 24576, len(ids) - 300]
        tails = [dict(at=s, len=300, text=tok.decode(ids[s:s + 300])) for s in starts if 0 <= s < len(ids)]
        out["examples"].append(dict(id=rid, dataset=ds, first_eos=fe, tails=tails))
    return out



# ---------- within one trace ----------
LONG_SETS = ["arxiv", "govreport", "multilexsum", "narrativeqa", "mrcr", "swebench_live", "longbench_v2"]
HALF_OCT = [0, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
            65536, 98304, 131072, 196608, 262144, 393216]
BASE_GEN = 2_048     # baseline of a trace: acceptance over its first 2K generated tokens
MIN_ROUNDS = 16      # a (trace, window) cell needs this many rounds


def cycle_flags(results, methods, same_text=False, source=None):
    """(method, id, checkpoint t) -> in cycle, from the detectors in cycle_profile.
    same_text: all methods share one text, detected once (on `source`, e.g. the reference, if given)."""
    rows = []
    src = [source] if source else (methods[:1] if same_text else methods)
    for m in src:
        files = sorted((results / m / "outputs").glob("*.json.gz"))
        with ProcessPoolExecutor(32) as pool:
            for rid, c in pool.map(cycle_profile, [(f,) for f in files]):
                for t, s_, l_ in zip(c["points"], c["short"], c["long"]):
                    cyc = (s_ is not None) or (l_ is not None)
                    for mm in (methods if same_text else [m]):
                        rows.append((mm, rid, t, cyc))
    return pd.DataFrame(rows, columns=["method", "id", "t", "cyc"])


def wls_slope(x, y, w):
    if len(x) < 3:
        return None
    x, y, w = map(np.asarray, (x, y, w))
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    var = np.average((x - xm) ** 2, weights=w)
    return None if var == 0 else float(np.average((x - xm) * (y - ym), weights=w) / var)


def within_trace(rounds, flags, heat=True):
    """Paired change of acceptance inside each trace, per-trace slopes and a per-trace heatmap.
    Baseline: the trace's first BASE_GEN generated tokens from `pstart` (column, default 0 = start of the answer)."""
    r = rounds[rounds.proposed > 0].copy()
    r["t"] = ((r.gen // CYCLE_STEP) * CYCLE_STEP).clip(lower=2048)
    r = r.merge(flags, on=["method", "id", "t"], how="left")
    r["cyc"] = r.cyc.fillna(False).astype(bool)
    r["a_nc"] = np.where(r.cyc, 0, r.accepted)
    r["p_nc"] = np.where(r.cyc, 0, r.proposed)
    pstart = r.pstart if "pstart" in r else 0
    base = r[r.gen < pstart + BASE_GEN].groupby(["method", "id"]).agg(ba=("accepted", "sum"), bp=("proposed", "sum"))
    base["base"] = base.ba / base.bp
    long_ = set(LONG_SETS)
    heat_on = heat
    out = dict(edges=HALF_OCT, base_gen=BASE_GEN, min_rounds=MIN_ROUNDS)
    for axis, col in (("gen", "gen"), ("ctx", "ctx")):
        r["b"] = np.searchsorted(HALF_OCT, r[col], side="right") - 1
        g = r.groupby(["method", "id", "b"]).agg(a=("accepted", "sum"), p=("proposed", "sum"), n=("accepted", "size"),
                                                 a_nc=("a_nc", "sum"), p_nc=("p_nc", "sum"),
                                                 cyc=("cyc", "mean")).reset_index()
        g = g.merge(base[["base"]].reset_index(), on=["method", "id"])
        g = g.merge(r.drop_duplicates("id")[["id", "dataset", "prompt_tokens"]], on="id")
        g["acc"] = g.a / g.p
        g["acc_nc"] = np.where(g.p_nc >= 7 * MIN_ROUNDS, g.a_nc / g.p_nc.where(g.p_nc > 0), np.nan)
        g = g[g.n >= MIN_ROUNDS]
        g["d"] = g.acc - g.base
        g["d_nc"] = g.acc_nc - g.base
        g["grp"] = np.where(g.dataset.isin(long_), "long", "short")
        g["xc"] = [float(np.sqrt(max(HALF_OCT[b], 128) * HALF_OCT[b + 1])) for b in g.b]  # geometric bin centre
        curves = []
        for grp_name, gg in [("all", g)] + list(g.groupby("grp")) + list(g.groupby("dataset")):
            for (m, b), x in gg.groupby(["method", "b"]):
                for key, acc in (("d", "acc"), ("d_nc", "acc_nc")):
                    mu, lo, hi, n = boot(x[key])
                    if n:
                        y = x[x[key].notna()]  # tooltip acceptance on the same traces as the plotted change
                        curves.append(dict(group=grp_name, method=m, x=HALF_OCT[b], x1=HALF_OCT[b + 1], kind=key,
                                           mean=mu, lo=lo, hi=hi, n=n, acc=float(y[acc].mean()), cyc=float(y.cyc.mean())))
        slopes = []
        for (m, rid), x in g.groupby(["method", "id"]):
            row = dict(method=m, id=rid, dataset=x.dataset.iloc[0], grp=x.grp.iloc[0])
            row["s"] = wls_slope(np.log2(x.xc), x.acc, x.n)
            y = x.dropna(subset=["acc_nc"])
            row["s_nc"] = wls_slope(np.log2(y.xc), y.acc_nc, y.n)
            slopes.append(row)
        sl = pd.DataFrame(slopes)
        slope_sum = []
        for grp_name, ss in [("all", sl)] + list(sl.groupby("grp")) + list(sl.groupby("dataset")):
            for m, x in ss.groupby("method"):
                for key in ("s", "s_nc"):
                    v = x[key].dropna()
                    mu, lo, hi, n = boot(v)
                    if n:
                        slope_sum.append(dict(group=grp_name, method=m, kind=key, mean=mu, lo=lo, hi=hi, n=n,
                                              neg=float((v < 0).mean()), median=float(v.median())))
        heat = {}
        for (m, rid), x in (g.groupby(["method", "id"]) if heat_on else []):
            heat.setdefault(m, []).append(dict(id=rid, ds=x.dataset.iloc[0], p=int(x.prompt_tokens.iloc[0]),
                                               base=float(x.base.iloc[0]),
                                               s=next((s_["s"] for s_ in slopes if s_["method"] == m and s_["id"] == rid), None),
                                               c=[[int(b), round(float(a), 4), None if a2 != a2 else round(float(a2), 4), round(float(c), 3), int(n)]
                                                  for b, a, a2, c, n in zip(x.b, x.acc, x.acc_nc, x.cyc, x.n)]))
        out[axis] = dict(curves=curves, slopes=slope_sum, heat=heat)
    return out

IN_EDGES = [0, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 1 << 30]
IN_HEAD = 2_048  # input-length view: acceptance over the first 2K answer tokens of each trace


def fe_slope(tr):
    """Slope of acceptance per doubling of the input within datasets (fixed effects), bootstrap over traces in datasets."""
    groups = [(np.log2(d.prompt_tokens.to_numpy()), d.acc.to_numpy()) for _, d in tr.groupby("dataset")]

    def est(parts):
        num = sum(((l - l.mean()) * (a - a.mean())).sum() for l, a in parts)
        den = sum(((l - l.mean()) ** 2).sum() for l, a in parts)
        return float(num / den) if den > 0 else float("nan")

    bs = []
    for _ in range(1000):
        parts = []
        for l, a in groups:
            i = rng.integers(0, len(l), len(l))
            parts.append((l[i], a[i]))
        bs.append(est(parts))
    bs = np.asarray(bs)
    bs = bs[~np.isnan(bs)]
    return est(groups), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def input_view(rounds):
    """Acceptance over the first IN_HEAD answer tokens of long-input traces vs input length.
    A fixed answer window keeps the position in the answer equal across documents."""
    r = rounds[(rounds.proposed > 0) & (rounds.gen < IN_HEAD) & rounds.dataset.isin(LONG_SETS)]
    tr = r.groupby(["method", "id", "dataset", "prompt_tokens"]).agg(a=("accepted", "sum"), p=("proposed", "sum"),
                                                                   n=("accepted", "size")).reset_index()
    tr = tr[tr.n >= MIN_ROUNDS]
    tr["acc"] = tr.a / tr.p
    tr["bin"] = np.searchsorted(IN_EDGES, tr.prompt_tokens, side="right") - 1
    out = dict(edges=IN_EDGES[:-1], head=IN_HEAD, bins=[], slopes=[])
    for grp_name, gg in [("long", tr)] + list(tr.groupby("dataset")):
        for (m, b), x in gg.groupby(["method", "bin"]):
            mu, lo, hi, n = boot(x.acc)
            out["bins"].append(dict(group=grp_name, method=m, x=IN_EDGES[b], x1=IN_EDGES[b + 1], acc=mu, lo=lo, hi=hi, n=n,
                                    p=float(np.exp(np.log(x.prompt_tokens).mean()))))
    for m, x in tr.groupby("method"):
        est, lo, hi = fe_slope(x)
        out["slopes"].append(dict(method=m, slope=est, lo=lo, hi=hi, n=len(x)))
    return out


GRID_IN = [0, 1_024, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 1 << 30]
GRID_GEN = [0, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 1 << 30]


def grid_view(rounds, flags):
    """Paired change of acceptance by input length x generated tokens: per-trace acceptance in each cell
    (>= MIN_ROUNDS rounds) minus the same trace's first GRID_GEN[1] tokens (the first row), then mean over traces
    with a bootstrap CI. `nc` drops rounds inside a cycle from the cell (the baseline keeps all rounds)."""
    r = rounds[rounds.proposed > 0].copy()
    r["t"] = ((r.gen // CYCLE_STEP) * CYCLE_STEP).clip(lower=2048)
    r = r.merge(flags, on=["method", "id", "t"], how="left")
    r["cyc"] = r.cyc.fillna(False).astype(bool)
    r["gi"] = np.searchsorted(GRID_IN, r.prompt_tokens, side="right") - 1
    r["gg"] = np.searchsorted(GRID_GEN, r.gen, side="right") - 1
    b0 = r[r.gen < GRID_GEN[1]].groupby(["method", "id"]).agg(ba=("accepted", "sum"), bp=("proposed", "sum"),
                                                              n=("accepted", "size"))
    b0 = b0[b0.n >= MIN_ROUNDS]
    base = (b0.ba / b0.bp).rename("base").reset_index()
    out = dict(in_edges=GRID_IN[:-1], gen_edges=GRID_GEN[:-1], base_gen=GRID_GEN[1], cells=[])
    for kind, sub in (("all", r), ("nc", r[~r.cyc])):
        tr = sub.groupby(["method", "id", "gi", "gg"]).agg(a=("accepted", "sum"), p=("proposed", "sum"),
                                                            n=("accepted", "size")).reset_index()
        tr = tr[tr.n >= MIN_ROUNDS].merge(base, on=["method", "id"])
        tr["acc"] = tr.a / tr.p
        tr["d"] = tr.acc - tr.base
        for (m, i, g), x in tr.groupby(["method", "gi", "gg"]):
            mu, lo, hi, n = boot(x.d, 300)
            out["cells"].append(dict(kind=kind, method=m, i=int(i), g=int(g), d=mu, lo=lo, hi=hi, n=n,
                                     acc=float(x.acc.mean()), base=float(x.base.mean())))
    return out


def think_ends(outputs, end_id):
    """Position of the first </think> in each reference output (None-like large value if absent)."""
    ends = {}
    for f in outputs.glob("*.json.gz"):
        with gzip.open(f, "rt") as h:
            ids = json.load(h)["token_ids"]
        ends[f.name.removesuffix(".json.gz")] = ids.index(end_id) if end_id in ids else 1 << 30
    return ends


def position_view(rounds, flags, ends):
    """Same text for every drafter: acceptance by answer position (per trace and half-octave window, mean over traces),
    split by phase (reasoning before </think> / answer after it), and the share of traces still outside a cycle
    and still reasoning."""
    r = rounds[rounds.proposed > 0].copy()
    r["think"] = r.gen < r.id.map(ends).fillna(1 << 30)
    r["t"] = ((r.gen // CYCLE_STEP) * CYCLE_STEP).clip(lower=2048)
    r = r.merge(flags, on=["method", "id", "t"], how="left")
    r["cyc"] = r.cyc.fillna(False).astype(bool)
    r["grp"] = np.where(r.dataset.isin(LONG_SETS), "long", "short")
    r["b"] = np.searchsorted(HALF_OCT, r.gen, side="right") - 1
    pos = []
    phases = (("all", r), ("think", r[r.think]), ("answer", r[~r.think]))
    for phase, rp in phases:
        for kind, sub in (("all", rp), ("nc", rp[~rp.cyc])):
            g = sub.groupby(["method", "id", "b", "grp", "dataset"]).agg(a=("accepted", "sum"), p=("proposed", "sum"),
                                                                         n=("accepted", "size")).reset_index()
            g = g[g.n >= MIN_ROUNDS]
            g["acc"] = g.a / g.p
            for grp_name, gg in [("all", g)] + list(g.groupby("grp")) + list(g.groupby("dataset")):
                for (m, b), x in gg.groupby(["method", "b"]):
                    mu, lo, hi, n = boot(x.acc)
                    pos.append(dict(group=grp_name, method=m, kind=kind, phase=phase, x=HALF_OCT[b], x1=HALF_OCT[b + 1],
                                    acc=mu, lo=lo, hi=hi, n=n))
    # composition of the traces that reach position t (the forced text ends at cycle start + 4K, so traces drop out):
    # share still reasoning and share already inside a detected cycle, among the same traces
    length = rounds.groupby("id").gen.max()
    cyc = flags[flags.method == flags.method.iloc[0]] if len(flags) else flags
    cyc = cyc.sort_values("t")
    comp = []
    for t in HALF_OCT[1:]:
        ids = length.index[length.to_numpy() >= t]
        if len(ids) < 3:
            break
        inc = cyc[(cyc.t <= t) & cyc.id.isin(ids)].groupby("id").cyc.last().reindex(ids).fillna(False)
        comp.append(dict(x=int(t), n=int(len(ids)), think=float(np.mean([ends.get(i, 1 << 30) > t for i in ids])),
                         cyc=float(inc.mean())))
    return dict(edges=HALF_OCT, min_rounds=MIN_ROUNDS, pos=pos, comp=comp)


ENT_MIN_TOKENS = 64  # a (trace, window) cell needs this many tokens


def entropy_view(tables, flags, ends, meta, limit=None):
    """Target entropy (top-20, from the AR trajectory) inside each trace, by answer position and by full context.
    Situations "<phase>:<rounds>": phase all / think / answer (first </think>), rounds all / nc (outside cycles).
    Value: mean over traces of the per-(trace, window) mean, and the paired change from the trace's first
    BASE_GEN tokens of the same situation. flags: AR cycle flags (id, t, cyc); meta: id -> (dataset, prompt_tokens);
    limit: id -> number of tokens to keep (forced: the teacher-forced span)."""
    fl = {(i, t): c for i, t, c in zip(flags.id, flags.t, flags.cyc)}
    rows = []
    for f in sorted((tables / "ar_logprobs").glob("*.npz")):
        rid = f.stem
        if rid not in meta:
            continue
        ent = np.load(f)["entropy_topk"].astype(float)
        if limit is not None:
            ent = ent[:limit.get(rid, len(ent))]
        gen = np.arange(len(ent))
        t = np.maximum((gen // CYCLE_STEP) * CYCLE_STEP, 2048)
        cyc = np.array([fl.get((rid, int(x)), False) for x in t])
        ds, ptok = meta[rid]
        rows.append(pd.DataFrame(dict(id=rid, dataset=ds, gen=gen, ctx=gen + ptok, ent=ent, cyc=cyc,
                                      think=gen < ends.get(rid, 1 << 30))))
    if not rows:
        return None
    r = pd.concat(rows, ignore_index=True)
    r = r[np.isfinite(r.ent)]
    r["grp"] = np.where(r.dataset.isin(set(LONG_SETS)), "long", "short")
    # same split as the acceptance curves: phase (all / reasoning / answer) x rounds (all / outside cycles)
    sits = [(f"{ph}:{k}", rp[~rp.cyc] if k == "nc" else rp)
            for ph, rp in (("all", r), ("think", r[r.think]), ("answer", r[~r.think])) for k in ("all", "nc")]
    out = dict(edges=HALF_OCT, base_gen=BASE_GEN, min_tokens=ENT_MIN_TOKENS)
    for axis in ("gen", "ctx"):
        curves = []
        for sit, rs in sits:
            rs = rs.copy()
            # baseline: the first BASE_GEN tokens of this situation in the trace
            rs["k"] = rs.groupby("id").cumcount()
            base = rs[rs.k < BASE_GEN].groupby("id").ent.mean().rename("base")
            rs["b"] = np.searchsorted(HALF_OCT, rs[axis], side="right") - 1
            g = rs.groupby(["id", "b"]).agg(e=("ent", "mean"), n=("ent", "size"), cyc=("cyc", "mean"),
                                           think=("think", "mean"), dataset=("dataset", "first"),
                                           grp=("grp", "first")).reset_index()
            g = g[g.n >= ENT_MIN_TOKENS].merge(base.reset_index(), on="id")
            g["d"] = g.e - g.base
            for grp_name, gg in [("all", g)] + list(g.groupby("grp")) + list(g.groupby("dataset")):
                for b, x in gg.groupby("b"):
                    for kind in ("e", "d"):
                        mu, lo, hi, n = boot(x[kind])
                        if n:
                            curves.append(dict(group=grp_name, sit=sit, kind=kind, x=HALF_OCT[b], x1=HALF_OCT[b + 1],
                                               mean=mu, lo=lo, hi=hi, n=n, e=float(x.e.mean()),
                                               cyc=float(x.cyc.mean()), think=float(x.think.mean())))
        out[axis] = curves
    return out


def natural_analysis(tables):
    rounds = pd.read_parquet(tables / "rounds.parquet")
    reqs = pd.read_parquet(tables / "requests.parquet")
    out = dict(
        curves=curve(rounds, ["method"], "ctx", edges=CTX_EDGES),
        curves_ds=curve(rounds, ["method", "dataset"], "ctx", edges=CTX_EDGES),
        curves_domain=curve(rounds, ["method", "domain"], "ctx", edges=CTX_EDGES),
    )
    # long inputs: acceptance over the whole answer vs input length (one point per request)
    r = rounds[rounds.proposed > 0].groupby(["method", "id"]).agg(a=("accepted", "sum"), p=("proposed", "sum"),
                                                                  n=("accepted", "size")).reset_index()
    r["acc"] = r.a / r.p
    r["tau"] = r.a / r.n
    r = r.merge(reqs[["id", "method", "dataset", "domain", "prompt_tokens", "generated_tokens",
                      "matches_ar", "finish_reason"]], on=["id", "method"])
    out["per_request"] = r[["method", "id", "dataset", "domain", "prompt_tokens", "generated_tokens",
                            "acc", "tau", "matches_ar", "finish_reason"]].to_dict(orient="records")
    # input-length bins (answer-only acceptance) -> input effect separated from generation effect
    r["xin"] = pd.cut(r.prompt_tokens, CTX_EDGES, right=False, labels=CTX_EDGES[:-1]).astype(float)
    out["by_input"] = [dict(method=m, domain=d, x=float(x), **dict(zip(["acc", "lo", "hi", "n"], boot(g.acc))))
                       for (m, d, x), g in r.groupby(["method", "domain", "xin"])]
    out["by_input_all"] = [dict(method=m, x=float(x), **dict(zip(["acc", "lo", "hi", "n"], boot(g.acc))))
                           for (m, x), g in r.groupby(["method", "xin"])]
    out["overall_ds"] = [dict(method=m, dataset=d, **dict(zip(["acc", "lo", "hi", "n"], boot(g.acc))))
                         for (m, d), g in r.groupby(["method", "dataset"])]
    out["match_ar"] = reqs.groupby("method").matches_ar.mean().to_dict()
    out["counts"] = reqs.groupby(["method", "dataset"]).size().reset_index(name="n").to_dict(orient="records")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", default="tables")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="report/data.json")
    p.add_argument("--config", default="configs/natural.json", help="run config naming the target (for the </think> token)")
    p.add_argument("--panels", nargs="*", help="limit the within-trace section to these panels")
    p.add_argument("--prompts", default="data/prompts", help="prompt files (forced: force_tokens per prompt)")
    args = p.parse_args()
    data = {}
    t = Path(args.tables)
    if (t / "natural/rounds.parquet").exists():
        data["natural"] = natural_analysis(t / "natural")
    if (t / "noeos/rounds.parquet").exists():
        methods = [d.name for d in (Path(args.results) / "noeos").iterdir() if d.is_dir()]
        data["noeos"] = noeos_analysis(t / "noeos", Path(args.results) / "noeos", methods)
    cfg = json.loads(Path(args.config).read_text())
    from transformers import AutoTokenizer
    end_id = AutoTokenizer.from_pretrained(cfg["target"], revision=cfg.get("target_revision")).convert_tokens_to_ids("</think>")
    for panel, same_text in (("natural", False), ("noeos", False), ("forced", True)):
        if not (t / panel / "rounds.parquet").exists() or (args.panels and panel not in args.panels):
            continue
        res = Path(args.results) / panel
        methods = sorted(d.name for d in res.iterdir() if d.is_dir() and d.name != "ar")
        flags = cycle_flags(res, methods, same_text=same_text, source="ar" if same_text and (res / "ar").exists() else None)
        rounds = pd.read_parquet(t / panel / "rounds.parquet")
        rounds = rounds[rounds.method != "ar"].copy()
        # end of reasoning (first </think>) per trace: on the shared reference, or on each method's own text
        if same_text:
            e = think_ends(res / "ar" / "outputs", end_id)
            rounds["end"] = rounds.id.map(e).fillna(1 << 30)
        else:
            e = {m: think_ends(res / m / "outputs", end_id) for m in methods}
            rounds["end"] = [e.get(m, {}).get(i, 1 << 30) for m, i in zip(rounds.method, rounds.id)]
        w = within_trace(rounds, flags, heat=False)
        for phase, sub in (("think", rounds[rounds.gen < rounds.end].assign(pstart=0)),
                           ("answer", rounds[rounds.gen >= rounds.end].assign(pstart=lambda x: x.end))):
            wp = within_trace(sub, flags, heat=False)
            for axis in ("gen", "ctx"):
                for key in ("curves", "slopes"):
                    w[axis][key] += [dict(c, phase=phase) for c in wp[axis][key]]
        data.setdefault(panel, {})["within"] = w
        if panel in ("natural", "forced"):
            data[panel]["input"] = input_view(rounds)
        data[panel]["grid"] = grid_view(rounds, flags)
        if panel == "forced":
            data[panel]["position"] = position_view(rounds, flags, think_ends(res / "ar" / "outputs", end_id))
        # target entropy along the AR text (shared text in forced; AR's own text otherwise)
        if (res / "ar" / "outputs").is_dir() and (t / panel / "ar_logprobs").is_dir():
            if same_text:
                ar_flags = flags[flags.method == methods[0]]
            else:
                ar_flags = cycle_flags(res, ["ar"])
            meta = {i: (d, int(p)) for i, d, p in rounds.drop_duplicates("id")[["id", "dataset", "prompt_tokens"]].itertuples(index=False)}
            limit = None
            if same_text:
                fp = Path(args.prompts) / f"{panel}.jsonl"
                limit = {j["id"]: int(j["force_tokens"]) for j in map(json.loads, fp.read_text().splitlines())}
            data[panel]["entropy"] = entropy_view(t / panel, ar_flags, think_ends(res / "ar" / "outputs", end_id), meta, limit)
        print(panel, "within-trace done", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, default=lambda o: None if o != o else o))
    print("wrote", args.out, {k: list(v) for k, v in data.items()})


if __name__ == "__main__":
    main()
