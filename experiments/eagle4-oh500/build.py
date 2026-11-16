"""EAGLE-4 / Qwen3.8-27B: 500 full OpenHands trajectories from Open-SWE-Traces v1.2, stratified by L_max.

Subcommands (run with `python -I build.py <cmd> ...`; all paths are explicit arguments, nothing is read from cwd):

  select   audit tables -> selection.csv (500 rows: bin, rank, key, all audit metrics) + pool_counts.json
  export   selection + shards -> trajectories.parquet (original rows + provenance/selection/audit/render columns),
           calls.parquet (per-call token markup + per-call token-id check), tool_responses.parquet,
           render_ids.parquet (token ids of the full render); export_checks.json into the experiment dir
  report   manifest.json + distributions.md/.json (needs the outputs of select and export)
  verify   independent re-check of the outputs: counts/filters, re-selection with the same seed, re-render of every
           trajectory (ids == stored ids), markers at every offset, agreement with the audit -> verify.json

Selection (agreed, see README): config == openhands, resolved == 1, finish_kind == finish,
L_max_best_estimate in [32768, 262144), 5 bins x 100. Inside a bin the trajectories are ranked by
sha256(f"{seed}:{trajectory_id}") and the first 100 are taken: a seeded uniform sample without replacement that does
not depend on the row order of the input table or on the numpy version; any top-k of a bin is itself a uniform sample.

Render (as in the audit): Qwen3.8-27B@1d4bf0f2ff chat template, tools converted to OpenAI format, tool-call arguments
parsed from JSON strings into dicts (as vLLM does), reasoning_effort=xhigh, enable_thinking=True,
preserve_thinking=True (past reasoning stays in the context = template default).

Markup (token indices into the full render `ids`, add_generation_prompt=False):
  call_start        <|im_start|> of assistant turn q; the model input of call q is ids[:call_start] + GEN_IDS
                    (GEN_IDS = tokens of "<|im_start|>assistant\\n<think>\\n"); input_len = len of that input
  think_end         the </think> token of the turn (after it: "\\n\\n" + content + tool calls)
  tool_call_starts  every <tool_call> token of the turn;  tool_call_ends: every </tool_call>
  turn_end          the <|im_end|> closing the turn; output_len = turn_end + 1 - input_len
  tool responses    <tool_response> / </tool_response> tokens of every tool message, bound to the call by position
All marker positions are derived from the template structure (prefix renders), not by searching the text, and every
one of them is asserted to land on the expected special token id.
"""
import argparse, bisect, datetime, hashlib, json, os, platform, socket, sys, time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "open-swe-traces-audit"))
from audit import conv_msgs, conv_tools  # noqa: E402  (same conversion as the audit)

BINS = [32768, 65536, 100000, 131072, 200000, 262144]
BIN_LABELS = ["32K-64K", "64K-100K", "100K-128K", "128K-200K", "200K-256K"]
N_PER_BIN = 100
DEFAULT_SEED = 20261116
EXPECTED_POOL = [5087, 3467, 1428, 917, 149]  # from the audit hand-off; checked, not enforced

DATASET = "nvidia/Open-SWE-Traces"
DATASET_CONFIG = "v1.2"
DATASET_REV = "4fa4c2ae9afc6b397bc50391d43450c79993c138"
TOKENIZER_REPO = "Qwen/Qwen3.8-27B"
TOKENIZER_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOKENIZER_FILES = ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "config.json",
                   "generation_config.json"]
RENDER_KW = dict(reasoning_effort="xhigh", enable_thinking=True, preserve_thinking=True)
CONTEXT_MODE = "preserve"
HARNESS = "openhands"

GEN_PROMPT = "<|im_start|>assistant\n<think>\n"
IM_END_NL = "<|im_end|>\n"
THINK_CLOSE_SEP = "</think>\n\n"
USER_OPEN = "<|im_start|>user"
SPECIAL = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
           "</tool_response>"]

ORIGINAL_COLUMNS = ["instance_id", "repo", "license", "language", "trajectory_id", "messages", "tools", "resolved",
                    "metadata", "hf_dataset_name"]


# ----------------------------------------------------------------------------------------------------------- utils
def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                return h.hexdigest()
            h.update(b)


def rel(p):
    """Path relative to the project data/experiments layout (for the manifest; independent of the host)."""
    p = os.path.abspath(p)
    for anchor in ("data/open-swe-traces/", "experiments/"):
        i = p.find(anchor)
        if i != -1:
            return p[i:]
    return os.path.basename(p)


def sample_key(seed, tid):
    return hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest()


def load_tok(tokdir):
    from transformers import AutoTokenizer
    T = AutoTokenizer.from_pretrained(tokdir)
    T.chat_template = open(os.path.join(tokdir, "chat_template.jinja")).read()
    return T


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        f.write("\n")


# ---------------------------------------------------------------------------------------------------------- select
def do_select(trajectories_csv, termination_csv, seed):
    """Returns (selection DataFrame sorted by bin, rank; counts dict). Pure function of the inputs and the seed."""
    import numpy as np, pandas as pd
    t = pd.read_csv(trajectories_csv, low_memory=False)
    term = pd.read_csv(termination_csv)
    assert t.trajectory_id.is_unique and term.trajectory_id.is_unique
    t = t.merge(term, on="trajectory_id", how="left", validate="one_to_one")
    L = t.L_max_best_estimate
    m_cfg = t.config == HARNESS
    m_res = m_cfg & (t.resolved == 1)
    m_fin = m_res & (t.finish_kind == "finish")
    m_len = m_fin & (L >= BINS[0]) & (L < BINS[-1])
    pool = t[m_len].copy()
    # properties the selection relies on (from the audit); fail loudly if the inputs differ
    assert (t.loc[m_cfg, "harness"] == HARNESS).all()
    assert (pool.context_mode == CONTEXT_MODE).all()
    assert (pool.L_max_best_estimate == pool.L_max_visible_template).all()
    assert (pool.n_calls_over_window == 0).all() and pool.template_ok.all() and (pool.n_bad_tool_args == 0).all()
    pool["bin"] = np.searchsorted(BINS, pool.L_max_best_estimate.to_numpy(), side="right") - 1
    pool["sample_key"] = [sample_key(seed, x) for x in pool.trajectory_id]
    pool = pool.sort_values(["bin", "sample_key"])
    pool["rank_in_bin"] = pool.groupby("bin").cumcount()
    sel = pool[pool.rank_in_bin < N_PER_BIN].copy()
    sel["bin_lo"] = [BINS[b] for b in sel.bin]
    sel["bin_hi"] = [BINS[b + 1] for b in sel.bin]
    sel["bin_label"] = [BIN_LABELS[b] for b in sel.bin]
    sel["seed"] = seed
    sel["L_max"] = sel.L_max_best_estimate
    front = ["bin", "bin_label", "bin_lo", "bin_hi", "rank_in_bin", "sample_key", "seed", "L_max", "trajectory_id"]
    sel = sel[front + [c for c in sel.columns if c not in front]].reset_index(drop=True)
    counts = {
        "funnel": {"all_v1.2": int(len(t)), "config==openhands": int(m_cfg.sum()), "+resolved==1": int(m_res.sum()),
                   "+finish_kind==finish": int(m_fin.sum()), "+L_max in [32768,262144)": int(m_len.sum())},
        "openhands_finish_kind": {k: int(v) for k, v in t.loc[m_cfg, "finish_kind"].value_counts().items()},
        "openhands_resolved": {str(k): int(v) for k, v in t.loc[m_cfg, "resolved"].value_counts().items()},
        "pool_per_bin": {BIN_LABELS[b]: int((pool.bin == b).sum()) for b in range(len(BIN_LABELS))},
        "pool_per_bin_source": {BIN_LABELS[b]: {s: int(((pool.bin == b) & (pool.source == s)).sum())
                                                for s in sorted(pool.source.unique())} for b in range(len(BIN_LABELS))},
        "sample_per_bin": {BIN_LABELS[b]: int((sel.bin == b).sum()) for b in range(len(BIN_LABELS))},
        "expected_pool_per_bin": dict(zip(BIN_LABELS, EXPECTED_POOL)),
    }
    counts["pool_matches_expected"] = list(counts["pool_per_bin"].values()) == EXPECTED_POOL
    return sel, counts


def cmd_select(a):
    sel, counts = do_select(a.trajectories, a.termination, a.seed)
    os.makedirs(a.exp_out, exist_ok=True)
    sel.to_csv(os.path.join(a.exp_out, "selection.csv"), index=False)
    counts["inputs"] = {rel(p): sha256_file(p) for p in (a.trajectories, a.termination)}
    counts["seed"] = a.seed
    dump_json(counts, os.path.join(a.exp_out, "pool_counts.json"))
    print(json.dumps({k: counts[k] for k in ("funnel", "pool_per_bin", "sample_per_bin", "pool_matches_expected")},
                     indent=1))


# ---------------------------------------------------------------------------------------------------------- export
_T = None
_TOKDIR = None


def _init(tokdir):
    global _T, _TOKDIR
    _TOKDIR = tokdir
    _T = load_tok(tokdir)


def special_ids(T):
    return {s: T.convert_tokens_to_ids(s) for s in SPECIAL}


def markup_one(job):
    """Render one trajectory, locate all markers by template structure, check every call prefix at the token level."""
    tid, messages, tools_raw, teacher = job
    try:
        return _markup_one(tid, messages, tools_raw, teacher)
    except Exception as e:  # keep the trajectory id in the error
        raise RuntimeError(f"{tid}: {type(e).__name__}: {e}") from e


def _markup_one(tid, messages, tools_raw, teacher):
    import numpy as np
    T = _T
    S = special_ids(T)
    assert teacher["reasoning_effort"] == RENDER_KW["reasoning_effort"] and teacher["enable_thinking"] is True
    msgs, bad_args = conv_msgs(messages)
    assert bad_args == 0
    tools = conv_tools(tools_raw)

    def R(ms, gen):
        return T.apply_chat_template(ms, tools=tools, tokenize=False, add_generation_prompt=gen, **RENDER_KW)

    def enc(s):
        return T(s, add_special_tokens=False)["input_ids"]

    gen_ids = enc(GEN_PROMPT)
    text = R(msgs, False)
    e = T(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = e["input_ids"]
    tok_at = {}
    for i, (s, _) in enumerate(e["offset_mapping"]):
        tok_at.setdefault(s, i)

    def at(c, sym):
        i = tok_at[c]  # KeyError -> marker not on a token boundary
        assert ids[i] == S[sym], (sym, c, ids[i])
        assert text.startswith(sym, c), (sym, c)
        return i

    calls, tresp = [], []
    a_idx = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
    q_of = {j: q for q, j in enumerate(a_idx)}
    n_bind_mismatch = 0
    last_q = None
    group_start = None
    for j, m in enumerate(msgs):
        if m["role"] == "assistant":
            q = q_of[j]
            last_q = q
            P = R(msgs[:j], True)
            prefix_str_eq = P.endswith(GEN_PROMPT) and text.startswith(P)
            cs_char = len(P) - len(GEN_PROMPT)
            call_start = at(cs_char, "<|im_start|>")
            pids = enc(P)
            input_len = len(pids)
            eq_slice = pids == ids[:input_len]
            eq_hist_gen = pids == ids[:call_start] + gen_ids
            E = R(msgs[:j + 1], False)
            assert text.startswith(E)
            turn_end = at(len(E) - len(IM_END_NL), "<|im_end|>")
            r = (m.get("reasoning_content") or "").strip()
            c = (m["content"] or "").strip()
            tk_char = cs_char + len(GEN_PROMPT) + len(r) + 1
            think_end = at(tk_char, "</think>")
            tcs = m.get("tool_calls") or []
            tc_starts, tc_ends = [], []
            prev_end_char = None
            for k in range(len(tcs)):
                if k == 0:
                    s_char = tk_char + len(THINK_CLOSE_SEP) + len(c) + (2 if c else 0)
                else:
                    s_char = prev_end_char + len("</tool_call>") + 1
                Ek = E if k == len(tcs) - 1 else R(msgs[:j] + [dict(m, tool_calls=tcs[:k + 1])], False)
                e_char = len(Ek) - len(IM_END_NL) - len("</tool_call>")
                assert text.startswith(Ek[:e_char + len("</tool_call>")])
                tc_starts.append(at(s_char, "<tool_call>"))
                tc_ends.append(at(e_char, "</tool_call>"))
                prev_end_char = e_char
            raw = messages[j]
            calls.append(dict(
                trajectory_id=tid, call=q, msg_index=j, call_start=call_start, input_len=input_len,
                think_end=think_end, tool_call_starts=tc_starts, tool_call_ends=tc_ends, turn_end=turn_end,
                output_len=turn_end + 1 - input_len, n_reasoning_tokens=think_end - input_len,
                n_tool_calls=len(tcs), tool_names=[tc["function"]["name"] for tc in raw["tool_calls"] or []],
                tool_call_ids=[tc["id"] for tc in raw["tool_calls"] or []],
                reasoning_chars=len(raw["reasoning_content"] or ""), content_chars=len(raw["content"] or ""),
                reasoning_empty=not r,
                prefix_str_eq=prefix_str_eq, prefix_ids_eq_slice=eq_slice, prefix_ids_eq_hist_gen=eq_hist_gen))
        elif m["role"] == "tool":
            Ei = R(msgs[:j + 1], False)
            close_char = len(Ei) - len(IM_END_NL) - len("</tool_response>")
            assert text.startswith(Ei[:close_char + len("</tool_response>")])
            ct = (m["content"] or "").strip()
            open_char = close_char - 1 - len(ct) - 1 - len("<tool_response>")
            if msgs[j - 1]["role"] != "tool":  # first response of the group opens the user turn
                group_start = at(open_char - 1 - len(USER_OPEN), "<|im_start|>")
            k = 0
            while msgs[j - 1 - k]["role"] == "tool":
                k += 1
            aj = j - 1 - k  # the assistant turn this response belongs to (by position)
            assert msgs[aj]["role"] == "assistant"
            araw = messages[aj]["tool_calls"] or []
            tresp.append(dict(
                trajectory_id=tid, msg_index=j, call=q_of[aj], k=k,
                tool_name=araw[k]["function"]["name"] if k < len(araw) else None,
                tool_call_id=araw[k]["id"] if k < len(araw) else None,
                resp_start=at(open_char, "<tool_response>"), resp_end=at(close_char, "</tool_response>"),
                group_start=group_start, content_chars=len(messages[j]["content"] or "")))
            tresp[-1]["n_tokens"] = tresp[-1]["resp_end"] - tresp[-1]["resp_start"] - 1
    # binding by position: every assistant turn with N tool calls is followed by exactly N tool messages
    for q, j in enumerate(a_idx):
        n_tc = len(msgs[j].get("tool_calls") or [])
        n_tm = 0
        while j + 1 + n_tm < len(msgs) and msgs[j + 1 + n_tm]["role"] == "tool":
            n_tm += 1
        expected = 0 if q == len(a_idx) - 1 else n_tc  # the last call (finish) gets no response
        n_bind_mismatch += n_tm != expected
    first_user_char = text.find(USER_OPEN + "\n")
    ins = [x["input_len"] for x in calls]
    arr = np.asarray(ids, dtype=np.int32)
    traj = dict(
        trajectory_id=tid, render_n_tokens=len(ids), render_n_calls=len(calls),
        render_L_max=max(ins), render_L_max_call=ins.index(max(ins)), render_session_prompt_tokens_sum=sum(ins),
        render_output_tokens_max=max(x["output_len"] for x in calls),
        render_prefix_system_tools=at(first_user_char, "<|im_start|>"),
        render_n_tool_msgs=len(tresp), render_n_binding_mismatch=n_bind_mismatch,
        render_last_tool=",".join(calls[-1]["tool_names"]) if last_q is not None else "",
        render_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        render_ids_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
        render_tools_openai=[json.dumps(x, ensure_ascii=False) for x in tools],
        render_calls_prefix_ok=all(x["prefix_str_eq"] and x["prefix_ids_eq_hist_gen"] for x in calls),
        render_calls_slice_ok=sum(x["prefix_ids_eq_slice"] for x in calls),
    )
    return traj, calls, tresp, arr


def read_rows(job):
    """Take the selected rows of one shard file, in the given order; returns (positions, Arrow IPC bytes)."""
    import pyarrow as pa, pyarrow.parquet as pq
    path, rows, tids, positions = job
    tb = pq.read_table(path, columns=ORIGINAL_COLUMNS).take(rows)
    assert tb.column("trajectory_id").to_pylist() == tids, path
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, tb.schema) as w:
        w.write_table(tb)
    return positions, sink.getvalue().to_pybytes()


def load_audit_calls(calls_csv, tids):
    import pandas as pd
    keep = set(tids)
    parts = []
    for ch in pd.read_csv(calls_csv, chunksize=2_000_000,
                          usecols=["trajectory_id", "call", "msg_index", "input_visible_template", "output_template"]):
        parts.append(ch[ch.trajectory_id.isin(keep)])
    return pd.concat(parts, ignore_index=True)


def cmd_export(a):
    import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    t0 = time.time()
    sel = pd.read_csv(os.path.join(a.exp_out, "selection.csv"), low_memory=False)
    assert len(sel) == len(BIN_LABELS) * N_PER_BIN and sel.trajectory_id.is_unique
    os.makedirs(a.out, exist_ok=True)
    sel["src_path"] = [f"data/{h}/qwen38_27b/{s}/{f}" for h, s, f in zip(sel.harness, sel.source, sel.file)]
    # 1. original rows, in selection order (bin, rank)
    jobs = []
    for p, g in sel.groupby("src_path", sort=True):
        jobs.append((os.path.join(a.shards, p), g.row.astype(int).tolist(), g.trajectory_id.tolist(),
                     g.index.tolist()))
    parts = [None] * len(sel)
    with Pool(min(len(jobs), a.workers)) as pool:
        for positions, buf in pool.imap_unordered(read_rows, jobs):
            tb = pa.ipc.open_stream(buf).read_all()
            for i, pos in enumerate(positions):
                parts[pos] = tb.slice(i, 1)
    orig = pa.concat_tables(parts).combine_chunks()
    assert orig.column("trajectory_id").to_pylist() == sel.trajectory_id.tolist()
    print(f"read {orig.num_rows} rows from {len(jobs)} files in {time.time() - t0:.0f}s", flush=True)

    # 2. render + markup + exhaustive per-call token check (heaviest trajectories first for load balance)
    order = sel.sort_values("session_prompt_tokens_sum", ascending=False).index.tolist()
    msgs_col, tools_col, meta_col = orig.column("messages"), orig.column("tools"), orig.column("metadata")
    mjobs = [(sel.trajectory_id[i], msgs_col[i].as_py(), tools_col[i].as_py(), meta_col[i].as_py()["teacher_model"])
             for i in order]
    res = {}
    with Pool(a.workers, initializer=_init, initargs=(a.tok,)) as pool:
        for n, out in enumerate(pool.imap_unordered(markup_one, mjobs)):
            res[out[0]["trajectory_id"]] = out
            if n % 50 == 0:
                print(f"markup {n + 1}/{len(mjobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"markup done in {time.time() - t0:.0f}s", flush=True)
    tids = sel.trajectory_id.tolist()
    traj = pd.DataFrame([res[x][0] for x in tids])
    calls = pd.DataFrame([c for x in tids for c in res[x][1]])
    tresp = pd.DataFrame([c for x in tids for c in res[x][2]])
    bin_of = dict(zip(sel.trajectory_id, sel.bin))
    for df in (calls, tresp):
        df.insert(1, "bin", df.trajectory_id.map(bin_of).astype("int8"))
    calls.insert(3, "n_calls", calls.trajectory_id.map(traj.set_index("trajectory_id").render_n_calls))
    calls.insert(4, "is_last", calls.call == calls.n_calls - 1)

    # 3. agreement with the audit (per call and per trajectory)
    ac = load_audit_calls(a.audit_calls, tids)
    m = calls.merge(ac, on=["trajectory_id", "call"], how="left", validate="one_to_one", suffixes=("", "_audit"))
    calls["input_len_audit"] = m.input_visible_template.astype("int64").to_numpy()
    calls["output_len_audit"] = m.output_template.astype("int64").to_numpy()
    calls["msg_index_audit_eq"] = (m.msg_index == m.msg_index_audit).to_numpy()
    s = sel.set_index("trajectory_id").loc[tids]
    checks = {
        "n_trajectories": len(traj), "n_calls": len(calls), "n_tool_responses": len(tresp),
        "audit_calls_rows_matched": int(len(ac)),
        "calls_input_len_eq_audit": int((calls.input_len == calls.input_len_audit).sum()),
        "calls_output_len_eq_audit": int((calls.output_len == calls.output_len_audit).sum()),
        "calls_msg_index_eq_audit": int(calls.msg_index_audit_eq.sum()),
        "calls_prefix_str_eq": int(calls.prefix_str_eq.sum()),
        "calls_prefix_ids_eq_hist_gen": int(calls.prefix_ids_eq_hist_gen.sum()),
        "calls_prefix_ids_eq_slice": int(calls.prefix_ids_eq_slice.sum()),
        "calls_slice_mismatch_with_empty_reasoning": int((~calls.prefix_ids_eq_slice & calls.reasoning_empty).sum()),
        "calls_reasoning_empty": int(calls.reasoning_empty.sum()),
        "traj_L_max_eq_selection": int((traj.render_L_max.to_numpy() == s.L_max.to_numpy()).sum()),
        "traj_n_calls_eq_audit": int((traj.render_n_calls.to_numpy() == s.n_llm_calls.to_numpy()).sum()),
        "traj_n_tokens_eq_audit": int((traj.render_n_tokens.to_numpy() == s.total_rendered_tokens.to_numpy()).sum()),
        "traj_session_sum_eq_audit": int((traj.render_session_prompt_tokens_sum.to_numpy()
                                          == s.session_prompt_tokens_sum.to_numpy()).sum()),
        "traj_output_max_eq_audit": int((traj.render_output_tokens_max.to_numpy() == s.output_tokens_max.to_numpy()).sum()),
        "traj_prefix_system_tools_eq_audit": int((traj.render_prefix_system_tools.to_numpy()
                                                  == s.prefix_tokens_system_tools.to_numpy()).sum()),
        "traj_L_max_call_eq_audit": int((traj.render_L_max_call.to_numpy() == s.L_max_call_index.to_numpy()).sum()),
        "traj_n_tool_msgs_eq_audit": int((traj.render_n_tool_msgs.to_numpy() == s.n_tool_msgs.to_numpy()).sum()),
        "traj_binding_mismatch_turns": int(traj.render_n_binding_mismatch.sum()),
        "traj_last_tool_finish": int((traj.render_last_tool == "finish").sum()),
        "tool_responses_without_call": int(tresp.tool_name.isna().sum()),
        "seconds": round(time.time() - t0),
    }
    print(json.dumps(checks, indent=1), flush=True)

    # 4. trajectories.parquet: original columns unchanged + provenance / selection / audit / render columns
    orig_check = {"instance_id": "instance_id", "repo": "repo", "language": "language", "resolved": "resolved"}
    for oc, sc in orig_check.items():
        assert orig.column(oc).to_pylist() == s[sc].tolist(), oc
    cat = [x["category"] for x in meta_col.to_pylist()]
    assert cat == s.category.tolist()
    extra = pd.DataFrame({
        "src_dataset": DATASET, "src_config": DATASET_CONFIG, "src_revision": DATASET_REV,
        "src_harness": s.harness.to_numpy(), "src_source": s.source.to_numpy(), "src_path": s.src_path.to_numpy(),
        "src_row": s.row.astype("int32").to_numpy(),
        "bin": s.bin.astype("int8").to_numpy(), "bin_label": s.bin_label.to_numpy(),
        "bin_lo": s.bin_lo.astype("int32").to_numpy(), "bin_hi": s.bin_hi.astype("int32").to_numpy(),
        "L_max": s.L_max.astype("int32").to_numpy(), "seed": s.seed.astype("int64").to_numpy(),
        "rank_in_bin": s.rank_in_bin.astype("int16").to_numpy(), "sample_key": s.sample_key.to_numpy(),
        "render_tokenizer": TOKENIZER_REPO, "render_tokenizer_revision": TOKENIZER_REV,
        "render_chat_template_sha256": sha256_file(os.path.join(a.tok, "chat_template.jinja")),
        "render_reasoning_effort": RENDER_KW["reasoning_effort"], "render_enable_thinking": RENDER_KW["enable_thinking"],
        "render_preserve_thinking": RENDER_KW["preserve_thinking"], "render_context_mode": CONTEXT_MODE,
    })
    rcols = [c for c in traj.columns if c != "trajectory_id"]
    for c in rcols:
        extra[c] = traj[c].to_numpy() if c != "render_tools_openai" else traj[c].tolist()
    skip = {"trajectory_id", "instance_id", "repo", "language", "resolved", "harness", "source", "file", "row", "bin",
            "bin_label", "bin_lo", "bin_hi", "rank_in_bin", "sample_key", "seed", "L_max", "src_path"}
    for c in s.columns:
        if c not in skip:
            extra["audit_" + c] = s[c].to_numpy()
    ex = pa.Table.from_pandas(extra, preserve_index=False)
    full = orig
    for name in ex.column_names:
        full = full.append_column(name, ex.column(name))
    out_traj = os.path.join(a.out, "trajectories.parquet")
    pq.write_table(full, out_traj, row_group_size=20, compression="zstd")
    pq.write_table(pa.Table.from_pandas(calls, preserve_index=False), os.path.join(a.out, "calls.parquet"),
                   compression="zstd")
    pq.write_table(pa.Table.from_pandas(tresp, preserve_index=False), os.path.join(a.out, "tool_responses.parquet"),
                   compression="zstd")
    lens = np.array([len(res[x][3]) for x in tids], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    ids_tb = pa.table({"trajectory_id": pa.array(tids),
                       "bin": pa.array(sel.bin.astype("int8").tolist(), pa.int8()),
                       "n_tokens": pa.array(lens.astype(np.int32)),
                       "input_ids": pa.ListArray.from_arrays(pa.array(offs),
                                                             pa.array(np.concatenate([res[x][3] for x in tids])))})
    pq.write_table(ids_tb, os.path.join(a.out, "render_ids.parquet"), row_group_size=20, compression="zstd")
    checks["outputs"] = {rel(os.path.join(a.out, f)): {"rows": pq.ParquetFile(os.path.join(a.out, f)).metadata.num_rows,
                                                       "bytes": os.path.getsize(os.path.join(a.out, f)),
                                                       "sha256": sha256_file(os.path.join(a.out, f))}
                         for f in ("trajectories.parquet", "calls.parquet", "tool_responses.parquet",
                                   "render_ids.parquet")}
    checks["audit_calls_file"] = {rel(a.audit_calls): sha256_file(a.audit_calls)}
    checks["seconds"] = round(time.time() - t0)
    dump_json(checks, os.path.join(a.exp_out, "export_checks.json"))
    print(f"written in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------------------- report
def q(x, p):
    import numpy as np
    return float(np.quantile(x, p))


def md_table(df, index=False):
    """Minimal GitHub markdown table (no tabulate dependency)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    fmt = lambda v: f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v.item() if hasattr(v, "item") else v) for v in r) + " |")
    return "\n".join(lines)


def cmd_report(a):
    import numpy as np, pandas as pd, pyarrow.parquet as pq
    import transformers, tokenizers, pyarrow, jinja2
    sel = pd.read_csv(os.path.join(a.exp_out, "selection.csv"), low_memory=False)
    counts = json.load(open(os.path.join(a.exp_out, "pool_counts.json")))
    checks = json.load(open(os.path.join(a.exp_out, "export_checks.json")))
    calls = pq.read_table(os.path.join(a.out, "calls.parquet"),
                          columns=["trajectory_id", "bin", "input_len", "output_len", "n_tool_calls"]).to_pandas()
    # distributions
    D = {}
    rows = []
    for b, lab in enumerate(BIN_LABELS):
        g = sel[sel.bin == b]
        c = calls[calls.bin == b]
        rows.append(dict(bin=lab, pool=counts["pool_per_bin"][lab], sample=len(g),
                         sampling_fraction=round(len(g) / counts["pool_per_bin"][lab], 4),
                         scale_swe=int((g.source == "scale-swe").sum()),
                         swe_rebench_v2=int((g.source == "swe-rebench-v2").sum()),
                         L_max_min=int(g.L_max.min()), L_max_median=int(g.L_max.median()), L_max_max=int(g.L_max.max()),
                         calls_median=int(g.n_llm_calls.median()), calls_max=int(g.n_llm_calls.max()),
                         calls_total=int(len(c)), input_len_median=int(c.input_len.median()),
                         output_max_median=int(g.output_tokens_max.median()),
                         repos=int(g.repo.nunique()), languages=int(g.language.nunique())))
    by_bin = pd.DataFrame(rows)
    D["by_bin"] = rows
    D["sources"] = {k: int(v) for k, v in sel.source.value_counts().items()}
    D["languages"] = {k: int(v) for k, v in sel.language.value_counts().items()}
    D["categories"] = {k: int(v) for k, v in sel.category.value_counts().items()}
    rc = sel.repo.value_counts()
    D["repos"] = {"n_unique": int(len(rc)), "max_per_repo": int(rc.max()),
                  "n_repos_with_1": int((rc == 1).sum()), "top20": {k: int(v) for k, v in rc.head(20).items()},
                  "share_top10": round(float(rc.head(10).sum() / len(sel)), 4)}
    D["instances_unique"] = int(sel.instance_id.nunique())
    num = {"L_max": sel.L_max, "n_llm_calls": sel.n_llm_calls, "output_tokens_max": sel.output_tokens_max,
           "tool_output_tokens_max": sel.tool_output_tokens_max, "session_prompt_tokens_sum": sel.session_prompt_tokens_sum,
           "call_input_len (all calls)": calls.input_len, "call_output_len (all calls)": calls.output_len}
    D["quantiles"] = {k: {p: q(v, float(p)) for p in ("0", "0.1", "0.25", "0.5", "0.75", "0.9", "1")}
                      for k, v in num.items()}
    D["calls_total"] = int(len(calls))
    dump_json(D, os.path.join(a.exp_out, "distributions.json"))
    def counts_tbl(d, name):
        return md_table(pd.DataFrame({name: list(d.keys()), "n": [int(v) for v in d.values()],
                                      "share": [f"{v / len(sel):.1%}" for v in d.values()]}))

    qt = pd.DataFrame(D["quantiles"]).T.map(lambda v: f"{v:,.0f}")
    qt.columns = ["min", "p10", "p25", "p50", "p75", "p90", "max"]
    qt.index.name = "metric"
    md = ["# EAGLE-4 OH500: sample distributions", "",
          f"500 trajectories, {len(calls):,} model calls. Generated by `build.py report`.", "",
          "## By bin (L_max = max input tokens of one call)", "", md_table(by_bin), "",
          "## Sources", "", counts_tbl(D["sources"], "source"), "",
          "## Languages", "", counts_tbl(D["languages"], "language"), "",
          "## Categories", "", counts_tbl(D["categories"], "category"), "",
          f"## Repositories: {D['repos']['n_unique']} unique, max {D['repos']['max_per_repo']} per repo, "
          f"{D['repos']['n_repos_with_1']} with exactly one; top-10 hold {D['repos']['share_top10']:.1%}", "",
          "Top 20:", "", counts_tbl(D["repos"]["top20"], "repo"), "",
          "## Quantiles (tokens; calls for n_llm_calls)", "", md_table(qt, index=True), ""]
    md += ["## Bin x language", "",
           md_table(pd.crosstab(sel.bin_label, sel.language).loc[BIN_LABELS], index=True), ""]
    with open(os.path.join(a.exp_out, "distributions.md"), "w") as f:
        f.write("\n".join(md))
    # manifest
    tokfiles = {f: sha256_file(os.path.join(a.tok, f)) for f in TOKENIZER_FILES}
    man = {
        "name": "eagle4_oh500",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Full Qwen3.8-27B OpenHands trajectories whose per-call inputs can be rebuilt from the data, for "
                   "measuring speculative decoding (EAGLE-4) on Qwen3.8-27B. No models were run.",
        "dataset": {"repo": DATASET, "config": DATASET_CONFIG, "revision": DATASET_REV,
                    "files": f"data/{HARNESS}/qwen38_27b/{{scale-swe,swe-rebench-v2}}/train-*.parquet (12 files)"},
        "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REV, "sha256": tokfiles,
                      "note": "revision in effect when the trajectories were generated (Aug-Sep 2026); the next "
                              "template change (2026-11-07, 37561816e6) only adds {% generation %} tags"},
        "render": {"chat_template": "tokenizer chat_template.jinja", **RENDER_KW, "context_mode": CONTEXT_MODE,
                   "add_generation_prompt_for_call_inputs": True,
                   "tools": "OpenAI format ({type: function, function: {...}}), as stored for OpenHands",
                   "tool_call_arguments": "JSON strings parsed into dicts before rendering (as vLLM does)",
                   "generation_prompt": GEN_PROMPT, "add_special_tokens": False},
        "selection": {
            "filters": {"config": HARNESS, "resolved": 1, "finish_kind": "finish",
                        "L_max_best_estimate": "[32768, 262144)"},
            "L_max_definition": "max over calls of input tokens of one call (render of messages[:j] + generation "
                                "prompt); for OpenHands L_max_best_estimate == L_max_visible_template",
            "bins": BINS, "bin_labels": BIN_LABELS, "n_per_bin": N_PER_BIN, "seed": int(sel.seed.iloc[0]),
            "method": "inside each bin rank by sha256(f'{seed}:{trajectory_id}') ascending, take ranks 0..99 "
                      "(seeded uniform sampling without replacement; top-k of a bin is a uniform subsample)",
            "per_repo_cap": None, "per_task": "one trajectory per task in OpenHands, so automatically one per task",
            "funnel": counts["funnel"], "pool_per_bin": counts["pool_per_bin"],
            "pool_per_bin_source": counts["pool_per_bin_source"], "sample_per_bin": counts["sample_per_bin"],
            "pool_matches_audit_handoff": counts["pool_matches_expected"],
            "inputs_sha256": counts["inputs"],
        },
        "rationale": [
            "OpenHands is the cleanest configuration of the audit: with past reasoning in the context its L_max "
            "distribution stops exactly at the 262,144-token window and no trajectory exceeds it (both sources), so "
            "the template render with preserved reasoning is what the model saw",
            "one fixed system prompt (system+tools prefix = 3,938 tokens) and 4 tools (execute_bash, "
            "str_replace_editor, think, finish); no format-error turns; no incomplete runs; no compaction",
            "longest contexts among the clean configurations",
            "resolved == 1: the user's choice for the first run",
            "stratified by L_max: uniform sampling would give only ~7 trajectories in 200-256K",
            "full trajectories, no pre-cutting: test points (call start, after a tool response, after reasoning, ...) "
            "are chosen later from the markup",
            "no per-repo cap: natural repository distribution; repo is kept for repository-clustered CIs",
        ],
        "caveats": [
            "no sampling parameters, max_tokens or per-call usage in the data; generation_config: T=1.0, top_k=20, "
            "top_p=0.95; whether OpenHands runs were greedy is unknown",
            "OpenHands version v0.53.0 per the dataset author for v1.0, unconfirmed for v1.2",
            "tool responses are bound to calls by position (tool messages have no tool_call_id)",
            "failed responses (no tool call) were removed during dataset conversion in mini-SWE-agent; OpenHands "
            "shows no trace of this, but it cannot be ruled out",
            "no generated token ids, logprobs or usage in the dataset (checked: one schema across all 100 v1.2 shards, "
            "no such files in the repo tree, not mentioned in the card); tool_calls[].id is a vLLM call id, not token "
            "ids; render_ids.parquet holds our render of the text",
            "call starts are exact despite that: the harness sends the history as text and the server re-tokenizes it "
            "deterministically (canonically), so the model saw exactly ids[:call_start] + GEN_IDS",
            "points inside a turn (after </think>, before/after <tool_call>) are canonicalized: during the rollout the "
            "KV cache held the sampled ids, which may be a non-canonical split of the same text; the text and the "
            "special-token boundaries are exact, the ids of the current turn's reasoning/arguments may differ and "
            "cannot be recovered",
            "only call starts are real model inputs; points inside a turn are continuations of generated text",
            "lengths cannot be checked against the server (no usage); they are template renders",
        ],
        "outputs": checks["outputs"],
        "checks": {k: v for k, v in checks.items() if k not in ("outputs",)},
        "scripts_sha256": {rel(os.path.join(HERE, "build.py")): sha256_file(os.path.join(HERE, "build.py")),
                           rel(os.path.join(HERE, "..", "open-swe-traces-audit", "audit.py")):
                               sha256_file(os.path.join(HERE, "..", "open-swe-traces-audit", "audit.py"))},
        "software": {"python": platform.python_version(), "transformers": transformers.__version__,
                     "tokenizers": tokenizers.__version__, "pyarrow": pyarrow.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "jinja2": jinja2.__version__},
        "host": {"platform": platform.platform(), "cpus": os.cpu_count()},
    }
    vpath = os.path.join(a.exp_out, "verify.json")
    if os.path.exists(vpath):
        man["verify"] = json.load(open(vpath))
    dump_json(man, os.path.join(a.exp_out, "manifest.json"))
    print(by_bin.to_string(index=False))


# ---------------------------------------------------------------------------------------------------------- verify
def verify_one(job):
    """Fresh render of one exported trajectory: ids must equal the stored ids; markers must sit on their tokens."""
    tid, messages, tools_raw, stored_ids, calls, tresp = job
    T = _T
    S = special_ids(T)
    msgs, _ = conv_msgs(messages)
    tools = conv_tools(tools_raw)
    text = T.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False, **RENDER_KW)
    ids = T(text, add_special_tokens=False)["input_ids"]
    gen_ids = T(GEN_PROMPT, add_special_tokens=False)["input_ids"]
    bad = []
    if ids != stored_ids:
        bad.append("ids")
    for c in calls:
        if ids[c["call_start"]] != S["<|im_start|>"] or ids[c["think_end"]] != S["</think>"] \
                or ids[c["turn_end"]] != S["<|im_end|>"] or c["input_len"] != c["call_start"] + len(gen_ids) \
                or any(ids[x] != S["<tool_call>"] for x in c["tool_call_starts"]) \
                or any(ids[x] != S["</tool_call>"] for x in c["tool_call_ends"]) \
                or len(c["tool_call_starts"]) != c["n_tool_calls"] \
                or not (c["call_start"] < c["think_end"] < c["turn_end"]):
            bad.append(f"call{c['call']}")
    for r in tresp:
        if ids[r["resp_start"]] != S["<tool_response>"] or ids[r["resp_end"]] != S["</tool_response>"]:
            bad.append(f"resp{r['msg_index']}")
    # ordering: calls increase, each turn ends before the next call starts, responses sit between turns
    cs = [c["call_start"] for c in calls]
    if cs != sorted(cs) or any(calls[i]["turn_end"] >= calls[i + 1]["call_start"] for i in range(len(calls) - 1)):
        bad.append("order")
    if calls[-1]["turn_end"] != len(ids) - 2:  # last turn ends with "<|im_end|>\n"
        bad.append("last_turn_end")
    return tid, bad


def cmd_verify(a):
    import pandas as pd, pyarrow.parquet as pq
    t0 = time.time()
    V = {}
    tra = pq.read_table(os.path.join(a.out, "trajectories.parquet"))
    tr = tra.drop_columns(["messages", "tools", "metadata", "render_tools_openai"]).to_pandas()
    tr.insert(5, "messages", None)  # placeholders keep the original column order for the check below
    tr.insert(6, "tools", None)
    tr.insert(8, "metadata", tra.column("metadata").to_pylist())
    calls = pq.read_table(os.path.join(a.out, "calls.parquet")).to_pandas()
    tresp = pq.read_table(os.path.join(a.out, "tool_responses.parquet")).to_pandas()
    ids = pq.read_table(os.path.join(a.out, "render_ids.parquet"))
    # 1. counts and filters
    V["rows"] = len(tr)
    V["unique_trajectory_id"] = int(tr.trajectory_id.nunique())
    V["per_bin"] = {BIN_LABELS[b]: int(n) for b, n in tr.bin.value_counts().sort_index().items()}
    V["all_resolved_1"] = bool((tr.resolved == 1).all())
    V["all_finish_kind_finish"] = bool((tr.audit_finish_kind == "finish").all())
    V["all_config_openhands"] = bool((tr.audit_config == HARNESS).all() and (tr.src_harness == HARNESS).all())
    V["all_L_max_in_bin"] = bool(((tr.L_max >= tr.bin_lo) & (tr.L_max < tr.bin_hi)).all())
    V["all_L_max_eq_render"] = bool((tr.L_max == tr.render_L_max).all())
    V["all_teacher_xhigh_thinking"] = bool(all(m["teacher_model"]["reasoning_effort"] == "xhigh"
                                               and m["teacher_model"]["enable_thinking"] for m in tr.metadata))
    V["original_columns_first"] = list(tr.columns[:len(ORIGINAL_COLUMNS)]) == ORIGINAL_COLUMNS
    if a.shards:  # original columns are value-identical (incl. schema) to the source rows
        same = 0
        for p, g in tr.groupby("src_path", sort=True):
            src = pq.read_table(os.path.join(a.shards, p), columns=ORIGINAL_COLUMNS).take(g.src_row.tolist())
            mine = tra.select(ORIGINAL_COLUMNS).take(g.index.tolist())
            same += len(g) if src.equals(mine) else 0
        V["original_rows_identical_to_shards"] = same
    # 2. re-selection with the stored seed reproduces the sample (ids, bins, ranks, order)
    seed = int(tr.seed.iloc[0])
    sel2, _ = do_select(a.trajectories, a.termination, seed)
    V["reselect_same_seed_identical"] = (sel2.trajectory_id.tolist() == tr.trajectory_id.tolist()
                                         and sel2.bin.tolist() == tr.bin.tolist()
                                         and sel2.rank_in_bin.tolist() == tr.rank_in_bin.tolist())
    # 3. calls table vs audit and per-call token checks (computed in export)
    V["calls"] = {"n": len(calls),
                  "input_len_eq_audit": int((calls.input_len == calls.input_len_audit).sum()),
                  "prefix_str_eq": int(calls.prefix_str_eq.sum()),
                  "prefix_ids_eq_hist_gen": int(calls.prefix_ids_eq_hist_gen.sum()),
                  "prefix_ids_eq_slice": int(calls.prefix_ids_eq_slice.sum()),
                  "n_calls_eq_audit": bool((calls.groupby("trajectory_id").size().sort_index()
                                            == tr.set_index("trajectory_id").audit_n_llm_calls.sort_index()).all()),
                  "L_max_eq": bool((calls.groupby("trajectory_id").input_len.max().sort_index()
                                    == tr.set_index("trajectory_id").L_max.sort_index()).all())}
    # 4. fresh render of every trajectory with a fresh tokenizer, ids and markers
    cg = {k: g.sort_values("call").to_dict("records") for k, g in calls.groupby("trajectory_id")}
    rg = {k: g.to_dict("records") for k, g in tresp.groupby("trajectory_id")}
    id_by = dict(zip(ids.column("trajectory_id").to_pylist(), range(ids.num_rows)))
    order = tr.sort_values("render_n_tokens", ascending=False).index.tolist()
    mcol, tcol = tra.column("messages"), tra.column("tools")
    jobs = [(tr.trajectory_id[i], mcol[i].as_py(), tcol[i].as_py(),
             ids.column("input_ids")[id_by[tr.trajectory_id[i]]].as_py(),
             cg[tr.trajectory_id[i]], rg.get(tr.trajectory_id[i], [])) for i in order]
    bad = {}
    with Pool(a.workers, initializer=_init, initargs=(a.tok,)) as pool:
        for tid, b in pool.imap_unordered(verify_one, jobs):
            if b:
                bad[tid] = b
    V["rerender_trajectories_ok"] = len(jobs) - len(bad)
    V["rerender_failures"] = bad
    V["seconds"] = round(time.time() - t0)
    ok = (V["rows"] == 500 and V["unique_trajectory_id"] == 500 and set(V["per_bin"].values()) == {N_PER_BIN}
          and V["all_resolved_1"] and V["all_finish_kind_finish"] and V["all_config_openhands"] and V["all_L_max_in_bin"]
          and V["all_L_max_eq_render"] and V["reselect_same_seed_identical"] and not bad
          and V["calls"]["input_len_eq_audit"] == V["calls"]["n"] and V["calls"]["prefix_str_eq"] == V["calls"]["n"]
          and V["calls"]["prefix_ids_eq_hist_gen"] == V["calls"]["n"]
          and V.get("original_rows_identical_to_shards", 500) == 500)
    V["ALL_OK"] = bool(ok)
    dump_json(V, os.path.join(a.exp_out, "verify.json"))
    print(json.dumps(V, indent=1, default=str))


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "export", "report", "verify"])
    ap.add_argument("--trajectories", help="audit/final/trajectories.csv")
    ap.add_argument("--termination", help="audit/final/termination.csv")
    ap.add_argument("--audit-calls", help="audit calls.csv.gz (input_visible_template per call)")
    ap.add_argument("--shards", help="root that contains data/<harness>/qwen38_27b/<source>/*.parquet")
    ap.add_argument("--tok", help="tokenizer dir (Qwen3.8-27B@1d4bf0f2ff)")
    ap.add_argument("--out", help="data output dir (…/data/open-swe-traces/eagle4_oh500)")
    ap.add_argument("--exp-out", default=HERE, help="small results (selection.csv, manifest.json, …)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"select": cmd_select, "export": cmd_export, "report": cmd_report, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
