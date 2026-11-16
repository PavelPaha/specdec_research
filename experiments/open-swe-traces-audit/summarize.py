"""Summaries for the Open-SWE-Traces v1.2 audit.

usage: summarize.py <audit_dir> <light.parquet> <out_dir>
  audit_dir: trajectories.csv, calls.csv.gz, reasoning_share.csv, [signals.csv]
writes: trajectories.csv (with flags, config, context_mode, L_max_best_estimate), calls.csv.gz,
        summary_by_config.csv, validation_by_config.csv, tools_full_corpus.csv,
        greedy_check.csv, strict_summary_ref_hist.csv
"""
import json, os, sys
import numpy as np, pandas as pd

aud, light, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
WINDOW = 262144
# empirical thresholds of hidden compaction (input at the first strict reference to an unseen summary
# jumps here on the full corpus: CC 0 refs in 200-210K vs 47 in 210-230K; OpenCode 14 in 125-130K vs 579 in 130-135K)
COMPACT_THR = {"claudecode": 210_000, "opencode": 130_000}
TH = [("ge32K", 32768), ("ge64K", 65536), ("ge100K", 100000), ("ge128K", 131072),
      ("ge200K", 200000), ("ge250K", 250000), ("ge256K_window", 262144)]
LABEL = {"claudecode": "Claude Code", "opencode": "OpenCode", "openhands": "OpenHands",
         "sweagent": "SWE-agent", "minisweagent": "mini-SWE-agent"}


def config_of(h, src):
    # SWE-agent behaves as two different run configurations depending on the task source (see README §4)
    return f"sweagent/{src}" if h == "sweagent" else h


# context mode inferred from the window wall test (README §4):
#   "preserve" = past reasoning was in the context (template default), "no_past_reasoning" = it was not
MODE = {"claudecode": "preserve", "opencode": "preserve", "openhands": "preserve",
        "sweagent/scale-swe": "preserve", "sweagent/swe-rebench-v2": "no_past_reasoning",
        "minisweagent": "no_past_reasoning"}
CLABEL = {"claudecode": "Claude Code", "opencode": "OpenCode", "openhands": "OpenHands",
          "sweagent/scale-swe": "SWE-agent · scale-swe", "sweagent/swe-rebench-v2": "SWE-agent · swe-rebench-v2",
          "minisweagent": "mini-SWE-agent"}

t = pd.read_csv(f"{aud}/trajectories.csv")
c = pd.read_csv(f"{aud}/calls.csv.gz")
rs = pd.read_csv(f"{aud}/reasoning_share.csv")[["trajectory_id", "L_max_no_past_reasoning", "past_reasoning_at_end"]]
t = t.merge(rs, on="trajectory_id", how="left")
t["config"] = [config_of(h, s) for h, s in zip(t.harness, t.source)]
t["context_mode"] = t.config.map(MODE)

# flags: calls after the first crossing of the compaction threshold cannot be reconstructed
t["compaction_threshold"] = t.harness.map(COMPACT_THR).fillna(WINDOW)
c = c.merge(t[["trajectory_id", "compaction_threshold", "config", "context_mode"]], on="trajectory_id")
first = c[(c.input_visible_template > c.compaction_threshold) & (c.context_mode == "preserve")] \
    .groupby("trajectory_id").call.min().rename("first_call_over_thr")
c = c.merge(first, on="trajectory_id", how="left")
# compacting harnesses: the crossing call itself precedes compaction (keep it);
# other "preserve" configs: a call >= window is impossible, so it is excluded too.
# "no_past_reasoning" configs: visible length with reasoning is not the context; not flagged here.
c["after_suspected_compaction"] = np.where(c.harness.isin(list(COMPACT_THR)),
                                           c.call > c.first_call_over_thr, c.call >= c.first_call_over_thr)
t = t.merge(first, on="trajectory_id", how="left")
t["suspected_hidden_compaction"] = t.first_call_over_thr.notna() & (t.harness.isin(COMPACT_THR))
t["over_window_visible"] = t.L_max_visible_template >= WINDOW
t["over_window_no_past_reasoning"] = t.L_max_no_past_reasoning >= WINDOW
# L_max over calls that are still reconstructible (before first threshold crossing, inclusive)
ok = c[~c.after_suspected_compaction.fillna(False)]
t = t.merge(ok.groupby("trajectory_id").input_visible_template.max().rename("L_max_preserve_reconstructible"),
            on="trajectory_id")
# best estimate of the real L_max under the inferred context mode
t["L_max_best_estimate"] = np.where(t.context_mode == "preserve", t.L_max_preserve_reconstructible,
                                    t.L_max_no_past_reasoning)
t = t.drop(columns=["compaction_threshold"])
t.to_csv(f"{out}/trajectories.csv", index=False)
c.drop(columns=["compaction_threshold", "first_call_over_thr"]).to_csv(f"{out}/calls.csv.gz", index=False, compression="gzip")


def stats(x):
    q = np.percentile(x, [25, 50, 75, 90, 95])
    d = dict(n=len(x), p25=q[0], median=q[1], p75=q[2], p90=q[3], p95=q[4], max=x.max())
    d.update({k: int((x >= v).sum()) for k, v in TH})
    return d


rows = []
for k in CLABEL:
    g = t[t.config == k]
    for metric in ["L_max_best_estimate", "L_max_visible_template", "L_max_no_past_reasoning",
                   "L_max_preserve_reconstructible", "session_prompt_tokens_sum"]:
        rows.append(dict(config=f"Qwen3.8-27B · {CLABEL[k]} · xhigh", key=k, context_mode=MODE[k], metric=metric,
                         **stats(g[metric])))
for metric in ["L_max_best_estimate", "L_max_visible_template"]:
    rows.append(dict(config="All Qwen3.8-27B v1.2", key="all", context_mode="mixed", metric=metric, **stats(t[metric])))
pd.DataFrame(rows).round(0).to_csv(f"{out}/summary_by_config.csv", index=False)

v = []
for k in CLABEL:
    g = t[t.config == k]
    cc = c[c.config == k]
    v.append(dict(
        config=k, context_mode=MODE[k], n_traj=len(g), n_calls=len(cc),
        traj_over_window_no_past_reasoning=int(g.over_window_no_past_reasoning.sum()),
        template_render_ok=int(g.template_ok.sum()), bad_tool_args=int(g.n_bad_tool_args.sum()),
        exact_ids_checked=int(g.exact_checked.sum()), exact_ids_ok=int(g.exact_ok.sum()),
        system_prompt_variants=g.system_sha.nunique(),
        prefix_tokens_median=g.prefix_tokens_system_tools.median(),
        prefix_tokens_min=g.prefix_tokens_system_tools.min(), prefix_tokens_max=g.prefix_tokens_system_tools.max(),
        redaction_placeholders=int(g.n_redaction_placeholders.sum()), user_msgs_mid=int(g.n_user_mid.sum()),
        traj_over_window_visible=int(g.over_window_visible.sum()),
        traj_suspected_hidden_compaction=int(g.suspected_hidden_compaction.sum()),
        calls_after_suspected_compaction=int(cc.after_suspected_compaction.fillna(False).sum()),
        share_calls_reconstructible=round(1 - cc.after_suspected_compaction.fillna(False).mean(), 4),
        output_tokens_max=int(g.output_tokens_max.max()), output_tokens_p99_of_traj_max=g.output_tokens_max.quantile(.99),
        tool_output_tokens_p95_of_traj_max=g.tool_output_tokens_max.quantile(.95),
        tool_output_tokens_median_of_traj_median=g.tool_output_tokens_median.median(),
    ))
pd.DataFrame(v).to_csv(f"{out}/validation_by_config.csv", index=False)

# greedy check + compaction-threshold evidence from signals.csv (if present)
if os.path.exists(f"{aud}/signals.csv"):
    s = pd.read_csv(f"{aud}/signals.csv")
    gr = []
    for h, g in s.groupby("harness"):
        pairs = same = same_r = 0
        for _, gg in g.groupby(["instance_id", "first_prefix_sha"]):
            n = len(gg)
            if n < 2:
                continue
            pairs += n * (n - 1) // 2
            same += sum(k * (k - 1) // 2 for k in gg.first_output_sha.value_counts())
            same_r += sum(k * (k - 1) // 2 for k in gg.first_reasoning_sha.value_counts())
        gr.append(dict(harness=h, runs=len(g), tasks=g.instance_id.nunique(),
                       pairs_same_prefix=pairs, pairs_identical_first_output=same,
                       pairs_identical_first_reasoning=same_r))
    pd.DataFrame(gr).to_csv(f"{out}/greedy_check.csv", index=False)
    print(pd.DataFrame(gr).to_string())
    # visible input at the first strict reference to an unseen summary
    ss = s[s.first_strict_summary_ref_call.notna()][["trajectory_id", "harness", "first_strict_summary_ref_call"]]
    ss = ss.merge(c[["trajectory_id", "call", "input_visible_template", "config"]],
                  left_on=["trajectory_id", "first_strict_summary_ref_call"], right_on=["trajectory_id", "call"])
    bins = [0, 50e3, 100e3, 120e3, 125e3, 130e3, 135e3, 140e3, 150e3, 175e3, 190e3, 200e3, 210e3, 230e3, 262144, 1e7]
    hist = ss.groupby("config").input_visible_template.apply(
        lambda x: pd.Series(np.histogram(x, bins=bins)[0], index=[f"{int(a/1e3)}-{int(b/1e3)}K" for a, b in zip(bins, bins[1:])]))
    hist.unstack().to_csv(f"{out}/strict_summary_ref_hist.csv")
    print(hist.unstack().to_string())

# tool distribution over the FULL corpus (light survey, all 235,391 trajectories)
L = pd.read_parquet(light, columns=["harness", "tool_names"])
tr = []
for h, g in L.groupby("harness"):
    cnt = {}
    for s in g.tool_names:
        for k, n in json.loads(s).items():
            cnt[k] = cnt.get(k, 0) + n
    tot = sum(cnt.values())
    for k, n in sorted(cnt.items(), key=lambda x: -x[1]):
        tr.append(dict(harness=h, tool=k, calls=n, share=round(n / tot, 4)))
pd.DataFrame(tr).to_csv(f"{out}/tools_full_corpus.csv", index=False)
print(pd.DataFrame(v).T.to_string())
