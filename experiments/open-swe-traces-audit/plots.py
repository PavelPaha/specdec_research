"""Figures for the Open-SWE-Traces v1.2 audit (full corpus). usage: plots.py <results_dir> <out_dir>"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["claudecode", "opencode", "openhands", "sweagent/scale-swe", "sweagent/swe-rebench-v2", "minisweagent"]
LABEL = {"claudecode": "Claude Code", "opencode": "OpenCode", "openhands": "OpenHands",
         "sweagent/scale-swe": "SWE-agent · scale-swe", "sweagent/swe-rebench-v2": "SWE-agent · swe-rebench-v2",
         "minisweagent": "mini-SWE-agent"}
COL = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
WINDOW = 262144
THR = {"claudecode": 210_000, "opencode": 130_000}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
t = pd.read_csv(f"{res}/trajectories.csv")
kfmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K")
XMAX = 400_000

# 1. ECDF of L_max: as stored (with past reasoning) vs best estimate under the inferred context mode
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
for ax, col, title in [(axs[0], "L_max_visible_template", "As stored: all messages incl. past reasoning"),
                       (axs[1], "L_max_best_estimate", "Best estimate (inferred context mode, see README §4)")]:
    for k in ORDER:
        x = np.sort(np.clip(t[t.config == k][col].values, 0, XMAX))
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COL[k], lw=2, label=LABEL[k])
    ax.axvline(WINDOW, color=INK2, lw=1, ls="--")
    ax.text(WINDOW + 3000, 0.97, "window 262 144", color=INK2, fontsize=8.5, rotation=90, va="top")
    ax.set_xlim(0, XMAX); ax.xaxis.set_major_formatter(kfmt)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("L_max = max over calls of input tokens of ONE call (Qwen3.8 template)")
    ax.legend(loc="lower right", fontsize=8.5)
axs[0].set_ylabel("share of trajectories ≤ x")
axs[0].text(XMAX - 3000, 0.40, "values > 400K clipped\n(max: SWE-agent·rebench 1.37M,\nCC 847K, OpenCode 824K)",
            ha="right", va="bottom", fontsize=8, color=INK2)
fig.suptitle("Open-SWE-Traces v1.2 · Qwen3.8-27B · all 235 391 trajectories", x=0.01, ha="left", fontsize=11.5)
fig.tight_layout(); fig.savefig(f"{out}/lmax_ecdf.png", dpi=150); plt.close(fig)

# 2. Histograms per config: best estimate (filled) and as-stored (outline)
fig, axs = plt.subplots(len(ORDER), 1, figsize=(9.5, 12), sharex=True)
bins = np.arange(0, XMAX + 1, 10_000)
for ax, k in zip(axs, ORDER):
    g = t[t.config == k]
    ax.hist(np.clip(g.L_max_best_estimate, 0, XMAX - 1), bins=bins, color=COL[k], edgecolor=SURF, linewidth=1.5)
    ax.hist(np.clip(g.L_max_visible_template, 0, XMAX - 1), bins=bins, histtype="step", color=INK2, linewidth=1)
    ax.axvline(WINDOW, color=INK2, lw=1, ls="--")
    h = k.split("/")[0]
    if h in THR:
        ax.axvline(THR[h], color=INK, lw=1, ls=":")
        ax.text(THR[h] + 2500, ax.get_ylim()[1] * 0.5, f"≈ compaction start {THR[h]//1000}K", fontsize=8, color=INK)
    ax.set_title(f"{LABEL[k]}  ·  n={len(g):,}  ·  median best {g.L_max_best_estimate.median()/1000:.0f}K  ·  "
                 f"as stored > window: {(g.L_max_visible_template >= WINDOW).sum():,}", fontsize=9.5, loc="left")
    ax.set_ylabel("trajectories")
axs[-1].xaxis.set_major_formatter(kfmt)
axs[-1].set_xlabel("L_max, 10K bins (last = ≥390K) · filled = best estimate · outline = as stored\n"
                   "dashed = window · dotted = compaction start")
fig.suptitle("L_max distribution by configuration (full v1.2)", x=0.01, ha="left", fontsize=11.5)
fig.tight_layout(); fig.savefig(f"{out}/lmax_hist_by_config.png", dpi=150); plt.close(fig)

# 3. Context growth for the 3 longest (as stored) trajectories per config
calls = pd.read_csv(f"{res}/calls.csv.gz", usecols=["trajectory_id", "call", "input_visible_template", "after_suspected_compaction"])
fig, axs = plt.subplots(1, len(ORDER), figsize=(18, 3.9), sharey=True)
for ax, k in zip(axs, ORDER):
    g = t[t.config == k].sort_values("L_max_visible_template", ascending=False)
    for i, tid in enumerate(g.trajectory_id.iloc[:3]):
        x = calls[calls.trajectory_id == tid]
        ax.plot(x.call, x.input_visible_template, color=COL[k], lw=2 if i == 0 else 1.2, alpha=1 if i == 0 else 0.55)
        fc = x[x.after_suspected_compaction == True].call.min()
        if not np.isnan(fc):
            ax.plot([fc], [x[x.call == fc].input_visible_template.iloc[0]], "o", ms=6, color=COL[k], mec=SURF, mew=2)
    ax.axhline(WINDOW, color=INK2, lw=1, ls="--")
    h = k.split("/")[0]
    if h in THR:
        ax.axhline(THR[h], color=INK, lw=1, ls=":")
    ax.set_title(LABEL[k], fontsize=9.5, loc="left"); ax.set_xlabel("LLM call #")
    ax.yaxis.set_major_formatter(kfmt)
axs[0].set_ylabel("input tokens of the call (as stored)")
fig.suptitle("Context growth as stored, 3 longest trajectories per configuration · dot = first call after suspected "
             "compaction (summary not stored) · dashed = window", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{out}/context_over_session.png", dpi=150); plt.close(fig)

# 4. Session sum vs L_max (small multiples, log-log, hexbin for 235K points)
fig, axs = plt.subplots(1, len(ORDER), figsize=(18, 3.9), sharex=True, sharey=True)
for ax, k in zip(axs, ORDER):
    g = t[t.config == k]
    ax.scatter(g.L_max_visible_template, g.session_prompt_tokens_sum, s=3, color=COL[k], alpha=0.15, linewidths=0)
    r = (g.session_prompt_tokens_sum / g.L_max_visible_template).median()
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{LABEL[k]}\nmedian ratio {r:.0f}×", fontsize=9.5, loc="left")
    ax.set_xlabel("L_max (one call, as stored)")
axs[0].set_ylabel("Σ input over session (as stored)")
fig.suptitle("Session sum of input tokens vs L_max — do not confuse the two", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{out}/session_sum_vs_lmax.png", dpi=150); plt.close(fig)
print("ok")
