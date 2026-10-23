"""Stream raw rounds into request-level bins, then plot domain-specific curves."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_jsonl(path, compressed=False):
    if not path.exists():
        return
    opener = gzip.open if compressed else open
    try:
        with opener(path, "rt") as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Only the last, still-being-written line may be incomplete.
                    if not line.endswith("\n"):
                        return
                    raise
    except EOFError:
        # Active gzip streams do not have their final trailer yet.
        return


def aggregate(run_dir, bin_size=8192, include_partial=False):
    metadata = json.loads((run_dir / "metadata.json").read_text())
    label = metadata["args"]["method"]
    if label == "eagle3":
        label = "eagle3.1"
    if label != "ar":
        label += f" k={metadata['args']['budget']}"
    requests = {r["engine_request_id"]: r for r in read_jsonl(run_dir / "requests.started.jsonl")}
    completed = {r["engine_request_id"] for r in read_jsonl(run_dir / "requests.jsonl")
                 if r["completed_requested_length"]}
    bins = defaultdict(lambda: dict(proposed=0, accepted=0, rounds=0, emitted=0, seconds=0.0))
    for row in read_jsonl(run_dir / "rounds.jsonl.gz", compressed=True):
        rid = row["request_id"]
        if rid not in requests or (rid not in completed and not include_partial):
            continue
        if row["is_prefill"]:
            continue
        assert 0 <= row["accepted"] <= row["proposed"], row
        phases = ["all"]
        if row["phase"] in ("pre_eos", "post_eos"):
            phases.append(row["phase"])
        for phase in phases:
            key = (rid, row["prefix_tokens"] // bin_size, phase)
            b = bins[key]
            b["proposed"] += row["proposed"]
            b["accepted"] += row["accepted"]
            b["rounds"] += int(row["proposed"] > 0)
            b["emitted"] += row["emitted"]
            b["seconds"] += row["step_seconds"]
    rows = []
    for (rid, index, phase), values in bins.items():
        rows.append(dict(
            run=run_dir.name, method=label, id=requests[rid]["id"],
            domain=requests[rid]["domain"], phase=phase,
            prefix_bin_start=index * bin_size, prefix_bin_end=(index + 1) * bin_size,
            acceptance_rate=values["accepted"] / values["proposed"] if values["proposed"] else np.nan,
            accepted_per_round=values["accepted"] / values["rounds"] if values["rounds"] else np.nan,
            decode_tokens_per_second=values["emitted"] / values["seconds"] if values["seconds"] else np.nan,
            complete=rid in completed, **values,
        ))
    return rows


def summarize(frame):
    """Equal weight per prompt, with bootstrap over prompts (not over rounds)."""
    rng = np.random.default_rng(20261024)
    rows = []
    keys = ["method", "domain", "phase", "prefix_bin_start", "prefix_bin_end"]
    for key, group in frame.groupby(keys):
        base = dict(zip(keys, key))
        metrics = ["acceptance_rate", "accepted_per_round", "decode_tokens_per_second"]
        if "speedup_vs_ar" in group:
            metrics.append("speedup_vs_ar")
        for metric in metrics:
            measured = group[group[metric].notna()]
            values = measured[metric].to_numpy()
            if not len(values):
                continue
            means = rng.choice(values, (500, len(values)), replace=True).mean(axis=1) if len(values) > 1 else None
            rows.append(dict(base, metric=metric, n_prompts=measured["id"].nunique(), mean=values.mean(),
                             low=np.quantile(means, 0.025) if means is not None else np.nan,
                             high=np.quantile(means, 0.975) if means is not None else np.nan))
    return pd.DataFrame(rows)


def write_report(frame, output, native, limit, partial):
    lines = ["# Измеренные срезы acceptance", "",
             "Промежуточный прогон; учитываются только уже полученные токены." if partial
             else "Срезы завершённых запросов; число промптов показано для каждого среза.", "",
             f"Нативный контекст: {native}; запрошенный предел: {limit}; YaRN выключен.", "",
             "Числа ниже не доказывают причину деградации: после EOS модель принудительно продолжает ответ.", "",
             "| Домен | Метод | Фаза | Префикс | Промптов | Accepted / proposed |",
             "| --- | --- | --- | --- | ---: | ---: |"]
    windows = [(0, 8192), (32768, 65536), (native - 16384, native), (native, limit)]
    for low, high in windows:
        subset = frame[(frame.prefix_bin_start >= low) & (frame.prefix_bin_start < high)]
        subset = subset[subset.proposed > 0]
        for (domain, method, phase), group in subset.groupby(["domain", "method", "phase"]):
            prompts = group.groupby("id")[["accepted", "proposed"]].sum()
            rate = (prompts.accepted / prompts.proposed).mean()
            lines.append(f"| {domain} | {method} | {phase} | {low}–{high} | {len(prompts)} | {rate:.3f} |")
    lines.extend(["", "Отсутствующая строка означает отсутствие измерений, а не нулевой acceptance.",
                  "Полные кривые, интервалы и число наблюдений: `curves.csv`; сырые суммы: `per_prompt_bins.csv`.", ""])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bin-size", type=int, default=8192)
    parser.add_argument("--include-partial", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    comparison_signature = None
    for name in args.runs:
        path = Path(name)
        if (path / "metadata.json").exists():
            meta = json.loads((path / "metadata.json").read_text())
            signature = tuple(meta["config"][key] for key in [
                "target_revision", "max_model_len", "max_num_seqs", "temperature", "dtype"])
            if comparison_signature is None:
                comparison_signature = signature
            elif signature != comparison_signature:
                raise ValueError("Cannot combine different targets, lengths, concurrency or sampling settings")
            rows.extend(aggregate(path, args.bin_size, args.include_partial))
    if not rows:
        print("No measured rounds available yet; no plots fabricated.")
        return
    frame = pd.DataFrame(rows)
    pair_keys = ["id", "domain", "phase", "prefix_bin_start", "prefix_bin_end"]
    baseline = frame[frame.method == "ar"][pair_keys + ["decode_tokens_per_second"]]
    if not baseline.empty:
        baseline = baseline.rename(columns={"decode_tokens_per_second": "ar_tokens_per_second"})
        frame = frame.merge(baseline, how="left", on=pair_keys, validate="many_to_one")
        frame["speedup_vs_ar"] = frame.decode_tokens_per_second / frame.ar_tokens_per_second
    frame.to_csv(output / "per_prompt_bins.csv", index=False)
    stats = summarize(frame)
    stats.to_csv(output / "curves.csv", index=False)
    first_meta = json.loads((Path(args.runs[0]) / "metadata.json").read_text())
    native = first_meta["config"]["native_context"]
    limit = first_meta["config"]["max_model_len"]
    write_report(frame, output, native, limit, args.include_partial)
    domains = ["math", "code", "chat", "reasoning"]
    labels = {"acceptance_rate": "Accepted / proposed draft tokens",
              "accepted_per_round": "Accepted draft tokens / verification round (no bonus)",
              "decode_tokens_per_second": "Per-request decode tokens/s (scheduler wall time)",
              "speedup_vs_ar": "Paired decode speedup vs AR"}
    for phase in ["all", "pre_eos", "post_eos"]:
        for metric, ylabel in labels.items():
            subset = stats[(stats.phase == phase) & (stats.metric == metric)]
            if subset.empty:
                continue
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
            for ax, domain in zip(axes.flat, domains):
                for method, group in subset[subset.domain == domain].groupby("method"):
                    group = group.sort_values("prefix_bin_start")
                    x = (group.prefix_bin_start + group.prefix_bin_end) / 2 / 1024
                    n_low, n_high = group.n_prompts.min(), group.n_prompts.max()
                    count_label = str(n_low) if n_low == n_high else f"{n_low}–{n_high}"
                    line, = ax.plot(x, group["mean"], marker=".", markersize=3,
                                    label=f"{method} (n={count_label})")
                    ax.fill_between(x, group.low, group.high, alpha=0.15)
                    if len(group) == 1 and group.low.notna().all():
                        ax.errorbar(x, group["mean"], fmt="none", capsize=3,
                                    color=line.get_color(),
                                    yerr=[group["mean"] - group.low, group.high - group["mean"]])
                ax.axvline(native / 1024, color="black", linestyle="--", label="Native context")
                ax.set_xlim(0, limit / 1024)
                if metric == "acceptance_rate":
                    ax.set_ylim(0, 1.02)
                ax.set(title=domain, xlabel="Prefix before verification (Ki tokens)", ylabel=ylabel)
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
            partial = " (partial run)" if args.include_partial else ""
            smoke = " · SHORT SMOKE TEST" if first_meta["args"].get("max_new_tokens") else ""
            concurrency = first_meta["config"]["max_num_seqs"]
            fig.suptitle(f"Qwen3.6-35B-A3B · TP1 · C≤{concurrency} · {phase} · 95% bootstrap CI for n≥2{partial}{smoke}")
            fig.tight_layout()
            fig.savefig(output / f"{metric}_{phase}.png", dpi=160)
            fig.savefig(output / f"{metric}_{phase}.pdf")
            plt.close(fig)
    print(f"Saved {len(frame)} prompt bins to {output}")


if __name__ == "__main__":
    main()
