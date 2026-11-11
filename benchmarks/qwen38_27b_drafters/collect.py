"""Collect raw run outputs into compact tables for analysis.

    python collect.py --results results --prompts data/prompts --out tables

Writes per panel:
  tables/<panel>/rounds.parquet    one row per verification round of a finished request
                                   (method, id, ctx = context before the round, gen = generated
                                   before the round, proposed, accepted, emitted)
  tables/<panel>/requests.parquet  one row per finished request (+ prompt metadata, AR match)
  tables/<panel>/ar_logprobs/<id>.npz   AR target stats per generated token (if logged)
Only rounds from the attempt that finished a request are used (retries never double count).
Forced panel (teacher forcing, results/forced/<method>, prompts data/prompts/forced.jsonl from
prepare_forced.py): only requests that reproduced the reference count, and rounds are cut at the
prompt's force_tokens so every method covers the same span; results/forced/ar may link to the reference.
"""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def load_method(method_dir):
    reqs = []
    for f in sorted(method_dir.glob("requests-*.jsonl")):
        reqs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    reqs = [r for r in reqs if not r.get("forced") or r.get("forced_ok")]  # forced: only exact reproductions
    reqs = list({r["id"]: r for r in reqs}.values())  # last finish wins
    by_attempt = {}
    for r in reqs:
        by_attempt.setdefault(r["attempt"], {})[r["engine_request_id"]] = r["id"]
    cols = {k: [] for k in ["id", "ctx", "gen", "proposed", "accepted", "emitted"]}
    for attempt, wanted in by_attempt.items():
        path = method_dir / f"rounds-{attempt}.jsonl.gz"
        try:
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:  # truncated tail of a killed attempt
                        break
                    rid = wanted.get(r["request_id"])
                    if rid is None or r.get("is_prefill") and r["proposed"] == 0 and r["emitted"] == 0:
                        continue
                    cols["id"].append(rid)
                    cols["ctx"].append(r["prefix_tokens"])
                    cols["gen"].append(r["generated_before"])
                    cols["proposed"].append(r["proposed"])
                    cols["accepted"].append(r["accepted"])
                    cols["emitted"].append(r["emitted"])
        except (EOFError, OSError):
            pass  # unfinished gzip stream: keep what was read
    rounds = pd.DataFrame(cols)
    rounds["method"] = method_dir.name
    return method_dir.name, pd.DataFrame(reqs), rounds


def ar_stats(args):
    path, out = args
    with gzip.open(path, "rt") as handle:
        d = json.load(handle)
    lp = d.get("logprobs")
    if not lp:
        return
    arrays = {k: np.array([np.nan if x is None else x for x in v], dtype=np.float32)
              for k, v in lp.items() if k != "top1_id"}
    arrays["token_ids"] = np.array(d["token_ids"], dtype=np.int32)
    np.savez_compressed(out / f"{d['id']}.npz", **arrays)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--prompts", default="data/prompts")
    p.add_argument("--out", default="tables")
    p.add_argument("--panels", nargs="+", default=["natural", "noeos"])
    args = p.parse_args()
    for panel in args.panels:
        root = Path(args.results) / panel
        if not root.exists():
            continue
        out = Path(args.out) / panel
        out.mkdir(parents=True, exist_ok=True)
        prompts = pd.read_json(Path(args.prompts) / f"{panel}.jsonl", lines=True)
        meta = prompts[["id", "dataset", "domain", "prompt_tokens", "length_bin"]]
        with ProcessPoolExecutor(16) as pool:
            results = list(pool.map(load_method, sorted(d for d in root.iterdir() if d.is_dir())))
        reqs = pd.concat([r.assign(method=m) for m, r, _ in results if len(r)], ignore_index=True)
        rounds = pd.concat([x for _, _, x in results if len(x)], ignore_index=True)
        ar = reqs[reqs.method == "ar"].set_index("id")["output_sha256"]
        reqs["matches_ar"] = reqs.apply(lambda r: ar.get(r["id"]) == r["output_sha256"], axis=1)
        reqs = reqs.drop(columns=["dataset", "domain", "prompt_tokens"]).merge(meta, on="id")
        reqs.to_parquet(out / "requests.parquet")
        rounds = rounds.merge(meta[["id", "dataset", "domain", "prompt_tokens"]], on="id")
        if "force_tokens" in prompts:  # forced panel: one common span per prompt for every method
            limit = rounds.id.map(prompts.set_index("id")["force_tokens"])
            rounds = rounds[rounds.gen < limit]
        rounds.to_parquet(out / "rounds.parquet")
        lp_dir = out / "ar_logprobs"
        lp_dir.mkdir(exist_ok=True)
        files = sorted((root / "ar" / "outputs").glob("*.json.gz")) if (root / "ar").exists() else []
        with ProcessPoolExecutor(32) as pool:
            list(pool.map(ar_stats, [(f, lp_dir) for f in files]))
        print(panel, "requests", reqs.groupby("method").size().to_dict(), "rounds", len(rounds), flush=True)


if __name__ == "__main__":
    main()
