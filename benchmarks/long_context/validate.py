"""Verify trace accounting against actual emitted tokens before trusting curves."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def validate(path):
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["status"] == "complete", path
    summaries = {r["engine_request_id"]: r for r in
                 map(json.loads, (path / "requests.jsonl").read_text().splitlines())}
    emitted = defaultdict(int)
    proposed = defaultdict(int)
    eos_index = {}
    eos_ids = set(metadata["eos_token_ids"])
    with gzip.open(path / "rounds.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            rid = row["request_id"]
            assert row["generated_before"] == emitted[rid], (rid, "non-contiguous output")
            assert row["prefix_tokens"] == summaries[rid]["prompt_tokens"] + emitted[rid]
            assert 0 <= row["accepted"] <= row["proposed"]
            assert row["emitted"] == len(row["token_ids"])
            if metadata["config"].get("eos_policy") == "bad_words":
                assert not eos_ids.intersection(row["token_ids"]), (rid, "EOS emitted despite mask")
            assert not row.get("num_nans_in_logits"), (rid, "NaN logits")
            if rid not in eos_index:
                for i, token in enumerate(row["token_ids"]):
                    if token in eos_ids:
                        eos_index[rid] = emitted[rid] + i
                        break
            assert row["first_eos_output_index"] == eos_index.get(rid)
            emitted[rid] += row["emitted"]
            proposed[rid] += row["proposed"]
    for rid, summary in summaries.items():
        assert emitted[rid] == summary["generated_tokens"], (rid, "lost output")
        assert summary["completed_requested_length"]
        assert summary["first_eos_output_index"] == eos_index.get(rid)
        if metadata["args"]["method"] != "ar":
            assert proposed[rid] > 0, (rid, "speculation never enabled")
    progress = json.loads((path / "progress.json").read_text())
    assert progress["preemptions"] == 0, "KV capacity was insufficient"
    print(f"PASS {path}: {len(summaries)} requests, {sum(emitted.values())} emitted tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    for path in args.runs:
        validate(Path(path))


if __name__ == "__main__":
    main()
