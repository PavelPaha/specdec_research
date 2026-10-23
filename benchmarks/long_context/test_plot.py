"""Guard against fabricated acceptance, EOS mixing and round-weighted averages."""

import gzip
import json

import pandas as pd

from benchmarks.long_context.plot import aggregate, summarize


def test_acceptance_excludes_bonus_and_keeps_eos_phases_separate(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"args": {"method": "mtp", "budget": 3}}))
    (tmp_path / "requests.started.jsonl").write_text(json.dumps({
        "engine_request_id": "internal", "id": "math-000", "domain": "math"}) + "\n")
    (tmp_path / "requests.jsonl").write_text(json.dumps({
        "engine_request_id": "internal", "completed_requested_length": True}) + "\n")
    rows = [dict(request_id="internal", prefix_tokens=100, proposed=3, accepted=2,
                 emitted=3, step_seconds=0.1, phase="pre_eos", is_prefill=False),
            dict(request_id="internal", prefix_tokens=103, proposed=3, accepted=0,
                 emitted=1, step_seconds=0.1, phase="post_eos", is_prefill=False)]
    with gzip.open(tmp_path / "rounds.jsonl.gz", "wt") as handle:
        handle.write("\n".join(map(json.dumps, rows)) + "\n")
    bins = {r["phase"]: r for r in aggregate(tmp_path)}
    assert bins["all"]["acceptance_rate"] == 2 / 6
    assert bins["all"]["accepted_per_round"] == 1
    assert bins["pre_eos"]["acceptance_rate"] == 2 / 3
    assert bins["post_eos"]["acceptance_rate"] == 0


def test_prompt_average_does_not_overweight_many_rounds():
    rows = [dict(method="mtp", domain="math", phase="all", prefix_bin_start=0,
                 prefix_bin_end=8192, id="a", acceptance_rate=0.0,
                 accepted_per_round=0.0, decode_tokens_per_second=10),
            dict(method="mtp", domain="math", phase="all", prefix_bin_start=0,
                 prefix_bin_end=8192, id="b", acceptance_rate=1.0,
                 accepted_per_round=3.0, decode_tokens_per_second=30)]
    result = summarize(pd.DataFrame(rows)).set_index("metric")
    assert result.loc["acceptance_rate", "mean"] == 0.5
    assert result.loc["acceptance_rate", "n_prompts"] == 2
