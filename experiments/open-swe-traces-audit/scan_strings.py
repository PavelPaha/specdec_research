"""Structural + string scan of downloaded shards (one per harness).

usage: scan_strings.py <shards_root> <out.json> [workers]   (parallel over parquet files, merged per harness)
"""
import collections, glob, json, re, sys
import pyarrow.parquet as pq

root, out = sys.argv[1], sys.argv[2]
PATTERNS = {
    "redacted": re.compile(r"<REDACTED[_A-Z]*>|\[REDACTED[^\]]*\]|REDACTED_"),
    "home_tmp_ph": re.compile(r"<HOME>|<TMP>|<USER>|<PATH>|<EMAIL>|<IP>"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "system_reminder": re.compile(r"<system-reminder>"),
    "session_continued": re.compile(r"This session is being continued|conversation that ran out of context|<summary>|Summary of the conversation|compact", re.I),
    "clipped": re.compile(r"<response clipped>|Observations should not exceed|[Oo]utput (was |has been )?truncated|Output too long|too large|exceeds maximum|\[\.\.\. \d+|characters truncated|lines truncated|elided", re.I),
    "tool_call_text": re.compile(r"<tool_call>|<function="),
    "think_text": re.compile(r"</?think>"),
    "interrupted": re.compile(r"interrupted|Request was aborted|context length|maximum context|exceeded", re.I),
}

def scan(path):
    h = path.split("/shards/data/")[1].split("/")[0]
    t = pq.read_table(path, columns=["trajectory_id", "messages"]).to_pylist()
    c = collections.Counter(); ex = collections.defaultdict(list)
    rlen, clen = [], []
    for r in t:
        msgs = r["messages"]; seen = collections.Counter()
        c["traj"] += 1
        for i, m in enumerate(msgs):
            role, txt, rc = m["role"], m["content"] or "", m["reasoning_content"]
            for k, p in PATTERNS.items():
                for src, s in (("content", txt), ("reasoning", rc or "")):
                    n = len(p.findall(s))
                    if n:
                        c[f"{k}|{role}|{src}|hits"] += n
                        if not seen[(k, role, src)]:
                            c[f"{k}|{role}|{src}|traj"] += 1; seen[(k, role, src)] = 1
                            if len(ex[f"{k}|{role}|{src}"]) < 3:
                                mm = p.search(s); ex[f"{k}|{role}|{src}"].append(
                                    (r["trajectory_id"], i, s[max(0, mm.start() - 150): mm.end() + 150]))
            if role == "assistant":
                tcs = m["tool_calls"] or []
                rlen.append(len(rc or "")); clen.append(len(txt))
                if rc is None: c["asst_reasoning_none"] += 1
                elif not rc.strip(): c["asst_reasoning_empty"] += 1
                if not tcs:
                    c["asst_no_toolcalls_final" if i == len(msgs) - 1 else "asst_no_toolcalls_mid"] += 1
                    if i != len(msgs) - 1 and len(ex["asst_no_tc_mid"]) < 3:
                        ex["asst_no_tc_mid"].append((r["trajectory_id"], i, txt[:300], (rc or "")[-300:]))
                c["asst_calls"] += 1; c["tool_calls"] += len(tcs)
                if len(tcs) > 1: c["asst_parallel"] += 1
                for tc in tcs:
                    try:
                        a = json.loads(tc["function"]["arguments"])
                        if not isinstance(a, dict): c["args_not_dict"] += 1
                    except Exception:
                        c["args_bad_json"] += 1
                        if len(ex["args_bad_json"]) < 3: ex["args_bad_json"].append((r["trajectory_id"], i, tc["function"]["arguments"][:300]))
                # count following tool msgs
                k = i + 1; nt = 0
                while k < len(msgs) and msgs[k]["role"] == "tool": nt += 1; k += 1
                if nt != len(tcs):
                    c["toolresp_mismatch"] += 1
                    if len(ex["toolresp_mismatch"]) < 5: ex["toolresp_mismatch"].append((r["trajectory_id"], i, len(tcs), nt, k < len(msgs) and msgs[k]["role"]))
            elif role == "user" and i > 1:
                c["user_mid"] += 1
                if len(ex["user_mid"]) < 5: ex["user_mid"].append((r["trajectory_id"], i, txt[:400]))
            elif role == "tool":
                if not txt.strip(): c["tool_empty"] += 1
    return h, c, rlen, clen, ex


def main():
    import numpy as np
    from multiprocessing import Pool
    files = sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
    agg = {}
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as p:
        for h, c, rlen, clen, ex in p.imap_unordered(scan, files):
            A = agg.setdefault(h, [collections.Counter(), [], [], collections.defaultdict(list)])
            A[0].update(c); A[1] += rlen; A[2] += clen
            for k, v in ex.items():
                A[3][k] = (A[3][k] + v)[:5]
    q = lambda a: [int(x) for x in np.percentile(a, [5, 25, 50, 75, 95, 99])] + [int(max(a))]
    res = {}
    for h, (c, rlen, clen, ex) in sorted(agg.items()):
        res[h] = dict(counts=dict(sorted(c.items())), reasoning_chars_q=q(rlen), content_chars_q=q(clen), examples=ex)
        print(h, c["traj"], "traj")
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
