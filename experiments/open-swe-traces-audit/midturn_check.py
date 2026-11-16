"""Where do mid-trajectory user messages sit, and do harnesses' error/format notices appear elsewhere?

usage: midturn_check.py <shards_root> <out.csv> [workers]
For every user message after the first: previous role, first line of text.
Also counts tool messages whose text looks like a harness format-error notice.
"""
import collections, glob, re, sys
from multiprocessing import Pool
import pyarrow.parquet as pq, pandas as pd

ERR = re.compile(r"Tool call error|did not (?:include|contain|call) a (?:tool|function)|Your (?:last )?(?:output|response) "
                 r"(?:did not|was not)|format error|No function call|Please use the tools|You must use|Invalid tool call"
                 r"|not a valid tool|parse.*tool call", re.I)


def one(path):
    h = path.split("/shards/data/")[1].split("/")[0]
    src = path.split("/qwen38_27b/")[1].split("/")[0]
    rows = []
    for r in pq.read_table(path, columns=["trajectory_id", "messages"]).to_pylist():
        ms = r["messages"]
        for i, m in enumerate(ms):
            if i <= 1:
                continue
            txt = m["content"] or ""
            if m["role"] == "user" or (m["role"] == "tool" and ERR.search(txt[:600])):
                rows.append(dict(harness=h, source=src, trajectory_id=r["trajectory_id"], msg=i, role=m["role"],
                                 prev_role=ms[i - 1]["role"],
                                 prev_asst_has_tool_calls=bool(ms[i - 1].get("tool_calls")) if ms[i - 1]["role"] == "assistant" else None,
                                 first_line=txt.strip().split("\n")[0][:120]))
    return rows


def main():
    files = sorted(glob.glob(f"{sys.argv[1]}/**/*.parquet", recursive=True))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as p:
        rows = [x for rs in p.imap_unordered(one, files) for x in rs]
    d = pd.DataFrame(rows)
    d.to_csv(sys.argv[2], index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(d.groupby(["harness", "role", "prev_role"]).size().to_string())
    print(d.groupby(["harness", "role", "first_line"]).size().sort_values(ascending=False).head(25).to_string())


if __name__ == "__main__":
    main()
