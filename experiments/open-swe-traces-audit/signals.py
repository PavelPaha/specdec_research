"""Indirect evidence of hidden context management + greedy-check hashes, per trajectory.

The dataset stores one flat message list per trajectory. If the harness compacted or rewrote
history, the stored list is NOT what the model saw. We look for:
  * summary_ref  -- the model refers to a summary it was given ("from the summary", ...)
  * omitted      -- harness markers of elided old outputs (SWE-agent / OpenHands / opencode)
For the greedy check we hash the prefix before the first assistant turn and that first turn:
identical prefix + different first output across runs of one task => not greedy.

usage: signals.py <shards_root> <out.csv> [workers]
"""
import glob, hashlib, re, sys
from multiprocessing import Pool
import pyarrow.parquet as pq, pandas as pd

SUMMARY_REF = re.compile(
    r"(?:from|per|in|according to|recall(?:ed)? from|based on) (?:the|my) (?:previous |earlier |prior |conversation |context )?summary"
    r"|the (?:previous |earlier |conversation )?summary (?:said|says|mentioned|mentions|noted|notes|states|stated|indicates|established|recorded)"
    r"|(?:conversation|context|session) (?:was|has been|got) (?:compacted|summari[sz]ed|truncated|condensed|reset)"
    r"|previous (?:session|context window)", re.I)
# stricter variant used in the report (fewer false positives such as "I'll note it in the summary")
STRICT_REF = re.compile(
    r"pre-compaction|(?:from|per|according to|recall(?:ed)? from|in) (?:the|my) (?:previous |earlier |prior |conversation )?summary[:,]? "
    r"|the (?:previous |earlier |conversation )?summary (?:said|says|mentioned|noted|states|stated|established|recorded)"
    r"|context (?:was|has been) (?:compacted|summari[sz]ed)|previous (?:session|context window)|continued from a previous", re.I)
OMIT = re.compile(r"Old environment output|lines omitted|<response clipped>|CONDENSATION|condensed|Previous output (?:was )?truncated"
                  r"|\[Old tool result content cleared\]|tool result (?:was )?(?:cleared|pruned)", re.I)
sha = lambda s: hashlib.sha1(s.encode()).hexdigest()[:16]


def one(path):
    h = path.split("/shards/data/")[1].split("/")[0]
    rows = []
    for r in pq.read_table(path, columns=["trajectory_id", "instance_id", "messages"]).to_pylist():
        first_ref = None; n_ref = 0; first_strict = None; omit_role = {}; first_omit = None; ex = ""
        a = -1; pre_sha = out_sha = rea_sha = None
        msgs = r["messages"]
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                a += 1
                if a == 0:
                    pre_sha = sha(repr([(x["role"], x["content"]) for x in msgs[:i]]))
                    rea_sha = sha(m["reasoning_content"] or "")
                    out_sha = sha((m["reasoning_content"] or "") + "\x00" + (m["content"] or "") + "\x00"
                                  + repr([(tc["function"]["name"], tc["function"]["arguments"]) for tc in m["tool_calls"] or []]))
                s = (m["reasoning_content"] or "") + "\n" + (m["content"] or "")
                mm = SUMMARY_REF.search(s)
                if mm:
                    n_ref += 1
                    if first_ref is None:
                        first_ref = a; ex = s[max(0, mm.start() - 100): mm.end() + 100].replace("\n", " ")
                if first_strict is None and STRICT_REF.search(s):
                    first_strict = a
            mm = OMIT.search(m["content"] or "")
            if mm:
                omit_role[m["role"]] = omit_role.get(m["role"], 0) + 1
                if first_omit is None:
                    first_omit = (i, mm.group(0))
        rows.append(dict(harness=h, trajectory_id=r["trajectory_id"], instance_id=r["instance_id"],
                         first_summary_ref_call=first_ref, first_strict_summary_ref_call=first_strict,
                         n_summary_ref=n_ref, summary_ref_example=ex,
                         omit_tool=omit_role.get("tool", 0), omit_user=omit_role.get("user", 0),
                         omit_assistant=omit_role.get("assistant", 0), first_omit=str(first_omit),
                         first_prefix_sha=pre_sha, first_output_sha=out_sha, first_reasoning_sha=rea_sha))
    return rows


def main():
    files = sorted(glob.glob(f"{sys.argv[1]}/**/*.parquet", recursive=True))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as p:
        rows = [x for rs in p.imap_unordered(one, files) for x in rs]
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
