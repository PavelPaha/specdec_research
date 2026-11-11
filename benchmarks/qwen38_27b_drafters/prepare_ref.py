"""Prompts for the reference (teacher-forcing) panel: one greedy target trajectory per prompt, EOS masked.

Budget per prompt:
  short-input sets  -> 49 152 generated tokens
  long-input sets   -> generate until the full context reaches 128K (at least 49 152 new tokens),
                       never past the model window
Long-input sets are subsampled: REF_PER_LONG prompts per set, evenly spread over input length.

Usage: python prepare_ref.py --prompts data/prompts/natural.jsonl --out data/prompts/ref.jsonl
(also writes ref-short.jsonl / ref-long.jsonl next to it, so batch sizes can differ)
"""

import argparse
import json
from pathlib import Path

LONG = ["arxiv", "govreport", "multilexsum", "narrativeqa", "mrcr", "swebench_live", "longbench_v2"]
SHORT_NEW, LONG_CONTEXT, WINDOW = 49_152, 131_072, 262_144
REF_PER_LONG = 10


def budget(row):
    p = row["prompt_tokens"]
    new = SHORT_NEW if row["dataset"] not in LONG else max(SHORT_NEW, LONG_CONTEXT - p)
    return min(new, WINDOW - p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts/natural.jsonl")
    ap.add_argument("--out", default="data/prompts/ref.jsonl")
    ap.add_argument("--per-long", type=int, default=REF_PER_LONG)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    keep = [r for r in rows if r["dataset"] not in LONG]
    for ds in LONG:
        group = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: (r["prompt_tokens"], r["id"]))
        if len(group) <= a.per_long:
            keep += group
            continue
        step = (len(group) - 1) / (a.per_long - 1)
        keep += [group[round(i * step)] for i in range(a.per_long)]
    for r in keep:
        r["max_new_tokens"] = budget(r)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in keep))
    out = Path(a.out)
    for part, rows_ in (("short", [r for r in keep if r["dataset"] not in LONG]), ("long", [r for r in keep if r["dataset"] in LONG])):
        out.with_name(f"{out.stem}-{part}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_))
    total = sum(r["max_new_tokens"] for r in keep)
    by = {}
    for r in keep:
        by.setdefault(r["dataset"], [0, 0])
        by[r["dataset"]][0] += 1
        by[r["dataset"]][1] += r["max_new_tokens"]
    print(len(keep), "prompts,", f"{total / 1e6:.1f}M tokens to generate")
    for ds, (n, t) in sorted(by.items()):
        print(f"  {ds:14s} {n:4d} prompts {t / 1e6:6.2f}M tokens")


if __name__ == "__main__":
    main()
