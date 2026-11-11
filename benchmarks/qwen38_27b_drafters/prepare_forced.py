"""Forced-panel prompts: the reference prompts with a per-prompt forcing length.

A reference trajectory (EOS masked) usually falls into an exact cycle; past the cycle the target
just repeats and every drafter is accepted almost always, so it says nothing about long context.
force_tokens = min(len(reference), cycle_start + MARGIN): the whole clean part plus MARGIN tokens of
the cycle (enough to see the switch). cycle_start uses the same detectors as analyze.py.

Usage: python prepare_forced.py --prompts data/prompts/ref.jsonl --ref results/ref/ar \
         --out data/prompts/forced.jsonl
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from analyze import CYCLE_STEP, periodic

MARGIN = 4096


def cycle_start(ids):
    x = np.asarray(ids, dtype=np.int64)
    for t in range(2048, len(x) + 1, CYCLE_STEP):
        if periodic(x[t - 2048:t], range(1, 129)) or periodic(x[max(0, t - 8192):t], range(129, 2049)):
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts/ref.jsonl")
    ap.add_argument("--ref", default="results/ref/ar")
    ap.add_argument("--out", default="data/prompts/forced.jsonl")
    ap.add_argument("--margin", type=int, default=MARGIN)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    full = kept = 0
    for r in rows:
        with gzip.open(Path(a.ref) / "outputs" / f"{r['id']}.json.gz", "rt") as h:
            n_ids = json.load(h)["token_ids"]
        c = cycle_start(n_ids)
        r["ref_tokens"] = len(n_ids)
        r["cycle_at"] = c
        r["force_tokens"] = min(len(n_ids), (c if c is not None else len(n_ids)) + a.margin)
        full += len(n_ids)
        kept += r["force_tokens"]
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} prompts, forced {kept / 1e6:.2f}M of {full / 1e6:.2f}M reference tokens "
          f"({100 * kept / full:.0f}%); cycles in {sum(r['cycle_at'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
