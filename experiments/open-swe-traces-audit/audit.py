"""Open-SWE-Traces v1.2 (Qwen3.8-27B) audit: per-call input lengths via the chat template.

The dataset has NO per-call usage, so every length here is a *reconstruction*: the visible
messages (system prompt + tools are stored!) rendered with Qwen3.8-27B@1d4bf0f2ff's template.

For trajectory i and assistant turn j:
  input_j  = tokens(render(messages[:j]) + generation prompt)   -- one call, not the session sum
  output_j = tokens of assistant turn j up to and including <|im_end|>
L_i^max = max_j input_j.

Usage: python -I audit.py <parquet files...> --tok <tokenizer_dir> --out <dir> [--workers N]
       [--exact-check K]   (token-level check of K calls per trajectory on the first 300 traj/shard)
"""
import argparse, bisect, gzip, hashlib, json, os, re, sys
from multiprocessing import Pool

ASSIST_START = "<|im_start|>assistant\n"
USER_START = "<|im_start|>user\n"
GEN_PROMPT = "<|im_start|>assistant\n<think>\n"
WINDOW = 262144
REDACT_RX = re.compile(r"<(?:REDACTED_[A-Z_]+|HOME|TMP)>")

_TOK = None
TOK_DIR = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOK_DIR)
        _TOK.chat_template = open(os.path.join(TOK_DIR, "chat_template.jinja")).read()
    return _TOK


def conv_tools(tools):
    out = []
    for s in tools or []:
        t = json.loads(s)
        if "input_schema" in t and "function" not in t:  # Anthropic format (Claude Code) -> OpenAI
            t = {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                   "parameters": t["input_schema"]}}
        out.append(t)
    return out


def conv_msgs(msgs):
    out, bad = [], 0
    for m in msgs:
        d = {"role": m["role"], "content": m["content"] or ""}
        if m["role"] == "assistant":
            d["reasoning_content"] = m["reasoning_content"] or ""
            if m["tool_calls"]:
                tcs = []
                for tc in m["tool_calls"]:
                    try:
                        a = json.loads(tc["function"]["arguments"])
                        if not isinstance(a, dict):
                            raise ValueError
                    except Exception:
                        bad += 1
                        a = {"raw": tc["function"]["arguments"]}
                    tcs.append({"type": "function", "function": {"name": tc["function"]["name"], "arguments": a}})
                d["tool_calls"] = tcs
        out.append(d)
    return out, bad


def process(job):
    path, row0, rows, harness, source, exact_k = job
    import pyarrow.parquet as pq
    T = tok()
    n_gen = len(T(GEN_PROMPT, add_special_tokens=False)["input_ids"])
    pf = pq.ParquetFile(path)
    tbl = pf.read(columns=["trajectory_id", "instance_id", "repo", "language", "resolved", "messages", "tools",
                           "metadata.teacher_model", "metadata.category"]).slice(row0, rows).to_pylist()
    traj_rows, call_rows = [], []
    for k, r in enumerate(tbl):
        msgs_raw = r["messages"]
        msgs, bad_args = conv_msgs(msgs_raw)
        tools = conv_tools(r["tools"])
        tm = r["metadata"]["teacher_model"]
        kw = {"reasoning_effort": tm["reasoning_effort"]} if tm.get("reasoning_effort") else {}
        text = T.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False, **kw)
        enc = T(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, starts = enc["input_ids"], [o[0] for o in enc["offset_mapping"]]
        apos, i = [], text.find(ASSIST_START)
        while i != -1:
            apos.append(i); i = text.find(ASSIST_START, i + 1)
        a_idx = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
        ok = len(apos) == len(a_idx)
        tpos = [bisect.bisect_left(starts, p) for p in apos]
        im_end_id = T.convert_tokens_to_ids("<|im_end|>")
        first_user_tok = bisect.bisect_left(starts, text.find(USER_START))
        # token-level exactness check on a few calls
        exact = []
        if exact_k and row0 + k < 300:
            picks = sorted(set([0, len(a_idx) // 2, len(a_idx) - 1][:exact_k]))
            for q in picks:
                j = a_idx[q]
                pref = T.apply_chat_template(msgs[:j], tools=tools, tokenize=False, add_generation_prompt=True, **kw)
                pids = T(pref, add_special_tokens=False)["input_ids"]
                exact.append(pids == ids[:tpos[q]] + ids[tpos[q]:tpos[q] + n_gen])
        # tool outputs
        tool_txt = [m["content"] for m in msgs if m["role"] == "tool"]
        tool_tok = [len(x) for x in T(tool_txt, add_special_tokens=False)["input_ids"]] if tool_txt else []
        ins, outs = [], []
        for q, j in enumerate(a_idx):
            inp = tpos[q] + n_gen
            # output: from after gen prompt to the next <|im_end|> inclusive
            e = tpos[q] + n_gen
            while e < len(ids) and ids[e] != im_end_id:
                e += 1
            out = e + 1 - (tpos[q] + n_gen)
            m = msgs_raw[j]
            ins.append(inp); outs.append(out)
            call_rows.append(dict(trajectory_id=r["trajectory_id"], harness=harness, call=q, msg_index=j,
                                  input_visible_template=inp, output_template=out,
                                  reasoning_chars=len(m["reasoning_content"] or ""), content_chars=len(m["content"] or ""),
                                  n_tool_calls=len(m["tool_calls"] or []),
                                  tool_names="|".join(tc["function"]["name"] for tc in (m["tool_calls"] or []))))
        sysc = msgs_raw[0]["content"] if msgs_raw and msgs_raw[0]["role"] == "system" else ""
        raw = json.dumps(msgs_raw, ensure_ascii=False)
        tool_names = {}
        for m in msgs_raw:
            for tc in m["tool_calls"] or []:
                tool_names[tc["function"]["name"]] = tool_names.get(tc["function"]["name"], 0) + 1
        sorted_tt = sorted(tool_tok)
        traj_rows.append(dict(
            trajectory_id=r["trajectory_id"], instance_id=r["instance_id"], repo=r["repo"], language=r["language"],
            harness=harness, source=source, file=os.path.basename(path), row=row0 + k, resolved=r["resolved"],
            category=r["metadata"]["category"], reasoning_effort=tm.get("reasoning_effort"),
            enable_thinking=tm.get("enable_thinking"), n_msgs=len(msgs), n_llm_calls=len(a_idx),
            template_ok=ok, n_bad_tool_args=bad_args,
            exact_checked=len(exact), exact_ok=sum(exact),
            prefix_tokens_system_tools=first_user_tok, system_chars=len(sysc),
            system_sha=hashlib.sha1(sysc.encode()).hexdigest()[:12],
            L_max_visible_template=max(ins) if ins else 0, L_max_call_index=ins.index(max(ins)) if ins else -1,
            L_first_call=ins[0] if ins else 0, total_rendered_tokens=len(ids),
            session_prompt_tokens_sum=sum(ins), output_tokens_sum=sum(outs), output_tokens_max=max(outs) if outs else 0,
            n_calls_over_window=sum(1 for x in ins if x >= WINDOW),
            n_calls_in_plus_out_over_window=sum(1 for a, b in zip(ins, outs) if a + b > WINDOW),
            n_tool_calls=sum(tool_names.values()), tool_calls_by_type=json.dumps(tool_names),
            n_tool_msgs=len(tool_tok), tool_output_tokens_sum=sum(tool_tok),
            tool_output_tokens_median=sorted_tt[len(sorted_tt) // 2] if tool_tok else 0,
            tool_output_tokens_max=max(tool_tok) if tool_tok else 0,
            n_redaction_placeholders=len(REDACT_RX.findall(raw)),
            n_user_mid=sum(1 for j, m in enumerate(msgs_raw) if m["role"] == "user" and j > 1),
        ))
    return traj_rows, call_rows


def main():
    global TOK_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--exact-check", type=int, default=3)
    a = ap.parse_args()
    TOK_DIR = a.tok
    import pyarrow.parquet as pq
    import pandas as pd
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for f in a.files:
        parts = f.split("/")
        harness, source = parts[parts.index("qwen38_27b") - 1], parts[parts.index("qwen38_27b") + 1]
        n = pq.ParquetFile(f).metadata.num_rows
        jobs += [(f, s, min(a.chunk, n - s), harness, source, a.exact_check) for s in range(0, n, a.chunk)]
    T, C = [], []
    with Pool(a.workers, initializer=_init, initargs=(a.tok,)) as p:
        for i, (t, c) in enumerate(p.imap_unordered(process, jobs)):
            T += t; C += c
            if i % 20 == 0:
                print(f"{i+1}/{len(jobs)} jobs, {len(T)} traj", file=sys.stderr, flush=True)
    pd.DataFrame(T).sort_values(["harness", "file", "row"]).to_csv(f"{a.out}/trajectories.csv", index=False)
    pd.DataFrame(C).to_csv(f"{a.out}/calls.csv.gz", index=False, compression="gzip")
    print(len(T), "trajectories,", len(C), "calls")


def _init(d):
    global TOK_DIR
    TOK_DIR = d


if __name__ == "__main__":
    main()
