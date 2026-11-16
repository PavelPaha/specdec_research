"""How does each trajectory end? (completed by the agent vs cut off)

usage: termination.py <shards_root> <out.csv> [workers]
Per trajectory: last assistant turn shape, finish action, and a repetition score of the last reasoning.
  finish_kind:
    submit / finish / complete_task  -- harness's explicit finish action (SWE-agent / OpenHands / mini)
    final_text                       -- last assistant turn has text and no tool calls (CC / OpenCode final answer)
    empty_final                      -- last turn has no text and no tool calls (e.g. cut by max_tokens)
    other_tool                       -- ended on an ordinary tool call (step/cost/time limit or crash)
  rep_last: share of repeated 20-char shingles in the tail (last 20K chars) of the last reasoning+text
"""
import glob, sys
from multiprocessing import Pool
import pyarrow.parquet as pq, pandas as pd


def rep(s, k=20):
    s = s[-20000:]
    if len(s) < 2000:
        return 0.0
    sh = [s[i:i + k] for i in range(0, len(s) - k, 7)]
    return 1 - len(set(sh)) / len(sh)


def one(path):
    h = path.split("/shards/data/")[1].split("/")[0]
    rows = []
    for r in pq.read_table(path, columns=["trajectory_id", "messages"]).to_pylist():
        ms = r["messages"]
        last = [m for m in ms if m["role"] == "assistant"][-1]
        tcs = last["tool_calls"] or []
        names = [tc["function"]["name"] for tc in tcs]
        args = " ".join(tc["function"]["arguments"] for tc in tcs)
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in args:
            kind = "complete_task"
        elif "submit" in names:
            kind = "submit"
        elif "finish" in names:
            kind = "finish"
        elif not tcs and (last["content"] or "").strip():
            kind = "final_text"
        elif not tcs:
            kind = "empty_final"
        else:
            kind = "other_tool"
        rows.append(dict(trajectory_id=r["trajectory_id"], finish_kind=kind, last_tool=",".join(names),
                         rep_last=round(rep((last["reasoning_content"] or "") + (last["content"] or "")), 3)))
    return rows


def main():
    files = sorted(glob.glob(f"{sys.argv[1]}/**/*.parquet", recursive=True))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as p:
        rows = [x for rs in p.imap_unordered(one, files) for x in rs]
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
    print(len(rows))


if __name__ == "__main__":
    main()
