"""Summarize completed EOS-exclusion runs; never modifies inference or raw logs."""

import argparse
import gzip
from itertools import combinations
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.long_context.plot import aggregate, summarize


def window_change(frame, left, right):
    def window(bounds, denominator="proposed"):
        rows = frame[(frame.prefix_bin_start >= bounds[0]) &
                     (frame.prefix_bin_start < bounds[1])]
        sums = rows.groupby("id")[["accepted", denominator]].sum()
        return sums.accepted / sums[denominator]
    paired = pd.concat([window(left), window(right)], axis=1, keys=["early", "late"]).dropna()
    differences = (paired.late - paired.early).to_numpy() * 100
    rng = np.random.default_rng(20261028)
    bootstrap = rng.choice(differences, (2000, len(differences)), replace=True).mean(axis=1)
    return dict(n=len(paired), early=paired.early.mean(), late=paired.late.mean(),
                early_accepted_per_round=window(left, "rounds").mean(),
                late_accepted_per_round=window(right, "rounds").mean(),
                change_pp=differences.mean(), low_pp=np.quantile(bootstrap, .025),
                high_pp=np.quantile(bootstrap, .975))


def plot_nonperiodic(stats, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, domain in zip(axes.flat, ["math", "code", "chat", "reasoning"]):
        selected = stats[(stats.domain == domain) & (stats.phase == "all") &
                         (stats.metric == "acceptance_rate")]
        for method, group in selected.groupby("method"):
            group = group.sort_values("prefix_bin_start")
            x = (group.prefix_bin_start + group.prefix_bin_end) / 2048
            ax.plot(x, group["mean"], marker=".", label=f"{method}; n={group.n_prompts.min()}–{group.n_prompts.max()}")
            ax.fill_between(x, group.low, group.high, alpha=.15)
        ax.axvline(32, color="black", linestyle="--")
        ax.set(title=domain, xlabel="Prefix (Ki tokens)", ylabel="Accepted / proposed", xlim=(0, 48), ylim=(0, 1.02))
        ax.grid(alpha=.2)
        if len(selected):
            ax.legend(fontsize=8)
    fig.suptitle("Qwen3-8B · EOS banned · bins without detected short cycles\n"
                 "Period ≤128; ≥99% lag agreement in last 2048 tokens; 95% prompt bootstrap CI")
    fig.tight_layout()
    fig.savefig(output / "acceptance_without_short_cycles.png", dpi=160)
    fig.savefig(output / "acceptance_without_short_cycles.pdf")
    plt.close(fig)


def plot_cycle_fraction(flags, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, domain in zip(axes.flat, ["math", "code", "chat", "reasoning"]):
        for run, group in flags[flags.domain == domain].groupby("run"):
            rates = group.groupby("prefix_bin_start").near_cycle.mean()
            ax.plot((rates.index + 1024) / 1024, rates, label=run)
        ax.axvline(32, color="black", linestyle="--")
        ax.set(title=domain, xlabel="Prefix (Ki tokens)", ylabel="Fraction of prompts with short cycles",
               xlim=(0, 48), ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Qwen3-8B · EOS banned · short-cycle audit\n"
                 "Period ≤128; ≥99% lag agreement in last 2048 tokens; 50 prompts per domain")
    fig.tight_layout()
    fig.savefig(output / "short_cycle_fraction.png", dpi=160)
    plt.close(fig)


def audit_long_tails(directory, flags):
    """Inspect final tails missed by the primary short-cycle detector."""
    last = flags.sort_values("prefix_bin_start").groupby("id").tail(1)
    records = []
    for row in last[~last.near_cycle].itertuples():
        tokens = []
        with gzip.open(directory / f"{row.id}.tokens.jsonl.gz", "rt") as stream:
            for line in stream:
                tokens.extend(json.loads(line)["token_ids"])
        tail = np.asarray(tokens[-8192:], dtype=np.int32)
        score, period = max(((float(np.mean(tail[lag:] == tail[:-lag])), lag)
                             for lag in range(129, 2049)), key=lambda x: (x[0], -x[1]))
        records.append(dict(id=row.id, domain=row.domain, audited_tokens=len(tail),
                            best_period=period, lag_agreement=score, near_cycle=score >= .99))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    full = root / "experiment/full"
    frame = pd.read_csv(root / "experiment/plots/per_prompt_bins.csv")
    frame = frame[frame.phase == "all"]
    assert frame.complete.all()
    totals = []
    output_hashes = {}
    for p in sorted(full.glob("*/metadata.json")):
        meta = json.loads(p.read_text())
        assert meta["status"] == "complete"
        requests = [json.loads(x) for x in p.with_name("requests.jsonl").read_text().splitlines()]
        output_hashes[meta["args"]["method"]] = {r["id"]: r["output_sha256"] for r in requests}
        assert len(requests) == 200
        assert all(r["final_context"] == 49152 and r["first_eos_output_index"] is None for r in requests)
        progress = json.loads(p.with_name("progress.json").read_text())
        assert progress["preemptions"] == 0
        assert all(r["nan_logits"] == 0 for r in progress["requests"].values())
        work = json.loads(p.with_name("workload.json").read_text())
        totals.append(dict(method=meta["args"]["method"], budget=meta["args"]["budget"],
                           **work, run_directory=str(p.parent)))
    baseline = next(r["aggregate_tokens_per_second"] for r in totals if r["method"] == "ar")
    for row in totals:
        row["throughput_ratio_vs_ar"] = row["aggregate_tokens_per_second"] / baseline
    identical_outputs = {}
    for left, right in combinations(sorted(output_hashes), 2):
        assert output_hashes[left].keys() == output_hashes[right].keys()
        identical_outputs[f"{left}_vs_{right}"] = sum(
            output_hashes[left][rid] == output_hashes[right][rid] for rid in output_hashes[left])
    changes = []
    for method, group in frame[frame.proposed > 0].groupby("method"):
        for domain in ["all", "math", "code", "chat", "reasoning"]:
            data = group if domain == "all" else group[group.domain == domain]
            for name, left, right in [("early_vs_late", (4096, 8192), (45056, 49152)),
                                       ("native_boundary", (28672, 32768), (32768, 36864))]:
                changes.append(dict(method=method, domain=domain, comparison=name,
                                    early_window=left, late_window=right, **window_change(data, left, right)))
    changes = pd.DataFrame(changes)
    changes.to_csv(output / "paired_changes.csv", index=False)
    identical_ids = {rid for rid in output_hashes["dflash"]
                     if output_hashes["dflash"][rid] == output_hashes["eagle3"][rid]}
    identical_changes = [dict(method=method, **window_change(group, (4096, 8192), (45056, 49152)))
                         for method, group in frame[(frame.proposed > 0) & frame.id.isin(identical_ids)].groupby("method")]
    pd.DataFrame(identical_changes).to_csv(output / "identical_trajectory_changes.csv", index=False)
    speculative = frame[frame.proposed > 0]
    keys = ["method", "domain", "id"]
    def per_prompt_window(low, high):
        grouped = speculative[(speculative.prefix_bin_start >= low) &
                              (speculative.prefix_bin_start < high)].groupby(keys)[["accepted", "proposed"]].sum()
        return grouped.accepted / grouped.proposed
    individual = pd.concat([per_prompt_window(4096, 8192), per_prompt_window(45056, 49152)],
                           axis=1, keys=["early", "late"]).dropna().reset_index()
    individual["change_pp"] = (individual.late - individual.early) * 100
    individual.to_csv(output / "per_prompt_changes.csv", index=False)
    strong = individual[individual.change_pp < -50]
    prompts = {r["id"]: r for r in map(json.loads, Path("data/long_context_qwen3_8b_prompts.jsonl").read_text().splitlines())}
    examples = []
    for method, group in strong.groupby("method"):
        run = "dflash-k7" if method == "dflash k=7" else "eagle3-k7"
        for row in group.nsmallest(8, "change_pp").to_dict("records"):
            prompt = prompts[row["id"]]
            with gzip.open(full / run / f"{row['id']}.text.txt.gz", "rt") as handle:
                tail = handle.read()[-1200:]
            examples.append(dict(**row, source=prompt["source"], source_id=prompt["source_id"],
                                 messages=prompt["messages"], last_1200_characters=tail))
    (output / "hard_cases.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    pd.DataFrame(totals).to_csv(output / "throughput.csv", index=False)
    audit = json.loads((root / "experiment/cycle-audit/summary.json").read_text())
    assert all(r["eos_tokens"] == 0 and r["requests"] == 200 for r in audit)
    flags = pd.read_csv(root / "experiment/cycle-audit/cycles_by_prompt_bin.csv")
    plot_cycle_fraction(flags, output)
    long_records, long_summary = [], []
    for run, group in flags.groupby("run"):
        extra_tails = audit_long_tails(full / run, group)
        long_records.extend(dict(run=run, **r) for r in extra_tails)
        short = next(r["tail_near_cycles"] for r in audit if r["run"] == run)
        added = sum(r["near_cycle"] for r in extra_tails)
        long_summary.append(dict(run=run, checked=len(extra_tails),
                                 extra_exact_cycles=sum(r["lag_agreement"] == 1.0 for r in extra_tails),
                                 extra_near_cycles=added, total_detected_tail_cycles=short + added,
                                 remaining_unclassified=len(extra_tails) - added))
    pd.DataFrame(long_records).to_csv(output / "long_tail_audit.csv", index=False)
    nonperiodic = pd.read_csv(root / "experiment/cycle-audit/curves_nonperiodic.csv")
    plot_nonperiodic(nonperiodic, output)
    control = root / "native-block-control/dflash-k15-control28"
    meta = json.loads((control / "metadata.json").read_text())
    assert meta["status"] == "complete"
    control_audit_path = root / "native-block-control/cycle-audit"
    control_audit = json.loads((control_audit_path / "summary.json").read_text())
    assert len(control_audit) == 1 and control_audit[0]["requests"] == 28
    assert control_audit[0]["eos_tokens"] == 0
    control_flags = pd.read_csv(control_audit_path / "cycles_by_prompt_bin.csv")
    long_tails = audit_long_tails(control, control_flags)
    (output / "control_long_tail_audit.json").write_text(json.dumps(long_tails, indent=2))
    control_requests = [json.loads(line) for line in (control / "requests.jsonl").read_text().splitlines()]
    assert len(control_requests) == 28 and all(r["final_context"] == 49152 for r in control_requests)
    control_identical = sum(r["output_sha256"] == output_hashes["dflash"][r["id"]] for r in control_requests)
    extra = pd.DataFrame(aggregate(control, bin_size=2048))
    extra = extra[extra.phase == "all"]
    ids = set(extra.id)
    assert len(ids) == 28
    matched = pd.concat([frame[(frame.method == "dflash k=7") & frame.id.isin(ids)], extra])
    matched = matched.copy()
    matched["domain"] = "all"
    stats = summarize(matched)
    stats = stats[stats.metric.isin(["acceptance_rate", "accepted_per_round"])]
    stats.to_csv(output / "dflash_block_budget_control.csv", index=False)
    control_changes = [dict(method=m, **window_change(g, (4096,8192), (45056,49152)))
                       for m, g in matched.groupby("method")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["acceptance_rate", "accepted_per_round"]):
        for method, group in stats[stats.metric == metric].groupby("method"):
            group = group.sort_values("prefix_bin_start")
            x = (group.prefix_bin_start + group.prefix_bin_end) / 2048
            ax.plot(x, group["mean"], label=method)
            ax.fill_between(x, group.low, group.high, alpha=.15)
        ax.axvline(32, color="black", linestyle="--")
        ax.set(xlabel="Prefix (Ki tokens)", ylabel=metric, xlim=(0,48))
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("DFlash · same first 28 prompts · k=7 vs published block-16 setting k=15")
    fig.tight_layout()
    fig.savefig(output / "dflash_block_budget_control.png", dpi=160)
    plt.close(fig)
    for name in ["acceptance_rate_all.png", "acceptance_rate_all.pdf", "accepted_per_round_all.png",
                 "speedup_vs_ar_all.png", "curves.csv", "per_prompt_bins.csv"]:
        shutil.copy2(root / "experiment/plots" / name, output / name)
    for name in ["summary.json", "curves_nonperiodic.csv", "cycles_by_prompt_bin.csv"]:
        shutil.copy2(root / "experiment/cycle-audit" / name, output / ("cycles_" + name))
    for name in ["summary.json", "cycles_by_prompt_bin.csv"]:
        shutil.copy2(control_audit_path / name, output / ("control_cycles_" + name))
    metadata_output = output / "metadata"
    for run in [*(p.parent for p in full.glob("*/metadata.json")), control]:
        destination = metadata_output / run.name
        destination.mkdir(parents=True, exist_ok=True)
        for name in ["metadata.json", "capacity.json", "workload.json", "requests.jsonl",
                     "progress.json", "throughput.jsonl", "effective_target_hf_config.json"]:
            shutil.copy2(run / name, destination / name)
        # Direct run.py invocations have no parent GPU monitor; AR has no draft config.
        for name in ["gpu.jsonl", "effective_draft_hf_config.json"]:
            if (run / name).exists():
                shutil.copy2(run / name, destination / name)
    for name in ["source-integrity.json", "runtime-integrity.json", "smoke-summary.json", "boundary-summary.json"]:
        # These supplementary snapshots belong to the original orchestration.
        # Replays using the documented CLI still have each run's own metadata.
        if (root / name).exists():
            shutil.copy2(root / name, metadata_output / name)
    summary = dict(main_runs=totals, cycle_audit=audit, dflash_budget_control=control_changes,
                   long_tail_audit=long_summary,
                   identical_outputs_out_of_200=identical_outputs,
                   identical_trajectory_acceptance=identical_changes,
                   control_cycle_audit=control_audit,
                   control_missed_short_cycles=long_tails,
                   dflash_control_identical_outputs_out_of_28=control_identical,
                   drops_over_50pp=strong.groupby("method").size().to_dict(),
                   inference_commit=meta["git_commit"], raw_root=str(root))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(changes[(changes.domain == "all")].to_string(index=False))


if __name__ == "__main__":
    main()
