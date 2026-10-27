"""Audit actual emitted IDs for EOS and short periodic tails, without inference.

Each prefix bin is classified using its last 2048 generated tokens. A cycle
means a period <=128 with >=99% lag agreement, not semantic quality.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path

import msgspec
import numpy as np
import pandas as pd


class Tokens(msgspec.Struct):
    token_ids: list[int]


def periodicity(tokens):
    if len(tokens) < 256:
        return None, 0.0
    matches = [(float(np.mean(tokens[lag:] == tokens[:-lag])), lag)
               for lag in range(1, 129)]
    score, lag = max(matches, key=lambda x: (x[0], -x[1]))
    return lag, score


def audit_request(task):
    directory, summary, eos_ids, bin_size = task
    decoder = msgspec.json.Decoder(Tokens)
    ids = []
    with gzip.open(Path(directory) / f"{summary['id']}.tokens.jsonl.gz", "rb") as handle:
        for line in handle:
            ids.extend(decoder.decode(line).token_ids)
    assert len(ids) == summary["generated_tokens"]
    values = np.asarray(ids, dtype=np.int32)
    eos_count = int(np.isin(values, eos_ids).sum())
    assert eos_count == 0, (directory, summary["id"], eos_count)
    rows = []
    for start in range(0, summary["final_context"], bin_size):
        end = min(start + bin_size, summary["final_context"])
        output_end = end - summary["prompt_tokens"]
        if output_end <= 0:
            continue
        tail = values[max(0, output_end - 2048):output_end]
        period, score = periodicity(tail)
        rows.append(dict(run=Path(directory).name, id=summary["id"], domain=summary["domain"],
            prefix_bin_start=start, prefix_bin_end=start + bin_size,
            audited_tokens=len(tail), best_period=period, lag_agreement=score,
            exact_cycle=score == 1.0, near_cycle=score >= .99,
            unique_tokens=len(np.unique(tail)), eos_count=eos_count))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins-csv")
    parser.add_argument("--bin-size", type=int, default=2048)
    args = parser.parse_args()
    tasks = []
    for directory in args.runs:
        path = Path(directory)
        meta = json.loads((path / "metadata.json").read_text())
        assert meta["status"] == "complete"
        for line in (path / "requests.jsonl").read_text().splitlines():
            tasks.append((directory, json.loads(line), meta["eos_token_ids"], args.bin_size))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [row for result in pool.map(audit_request, tasks) for row in result]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "cycles_by_prompt_bin.csv", index=False)
    tail = frame.sort_values("prefix_bin_start").groupby(["run", "id"]).tail(1)
    summary = []
    for run, group in tail.groupby("run"):
        summary.append(dict(run=run, requests=len(group), eos_tokens=int(group.eos_count.sum()),
            tail_exact_cycles=int(group.exact_cycle.sum()), tail_near_cycles=int(group.near_cycle.sum())))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.bins_csv:
        from benchmarks.long_context.plot import summarize
        bins = pd.read_csv(args.bins_csv)
        merged = bins.merge(frame, on=["run", "id", "domain", "prefix_bin_start", "prefix_bin_end"],
                            how="left", validate="many_to_one")
        assert merged.near_cycle.notna().all()
        merged.to_csv(output / "acceptance_with_cycle_flags.csv", index=False)
        for name, selected in [("nonperiodic", ~merged.near_cycle), ("periodic", merged.near_cycle)]:
            subset = merged[selected]
            if len(subset):
                summarize(subset).to_csv(output / f"curves_{name}.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
