"""Per call: tokens of past reasoning inside the visible input (to test 'harness dropped reasoning').

usage: reasoning_share.py <calls.csv.gz> <tokenizer_dir> <out.csv> <parquet files...>
Writes per-trajectory: L_max_visible_template, L_max_no_past_reasoning (input minus past reasoning tokens).
"""
import sys
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow.parquet as pq

calls, tokdir, out, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
T = None


def work(path):
    global T
    if T is None:
        from transformers import AutoTokenizer
        T = AutoTokenizer.from_pretrained(tokdir)
    h = path.split("/shards/data/")[1].split("/")[0]
    rows = []
    for r in pq.read_table(path, columns=["trajectory_id", "messages"]).to_pylist():
        rs = [(m["reasoning_content"] or "").strip() for m in r["messages"] if m["role"] == "assistant"]
        rt = [len(x) for x in T(rs, add_special_tokens=False)["input_ids"]]
        rows.append((h, r["trajectory_id"], np.concatenate([[0], np.cumsum(rt)[:-1]]).tolist()))
    return rows


def main():
    with Pool(len(files)) as p:
        res = [x for rows in p.map(work, files) for x in rows]
    past = {tid: v for _, tid, v in res}
    c = pd.read_csv(calls, usecols=["trajectory_id", "harness", "call", "input_visible_template"])
    c["past_reasoning"] = [past[t][k] for t, k in zip(c.trajectory_id, c.call)]
    c["input_no_past_reasoning"] = c.input_visible_template - c.past_reasoning
    g = c.groupby(["harness", "trajectory_id"]).agg(L_max_visible_template=("input_visible_template", "max"),
                                                    L_max_no_past_reasoning=("input_no_past_reasoning", "max"),
                                                    past_reasoning_at_end=("past_reasoning", "max")).reset_index()
    g.to_csv(out, index=False)
    print(g.groupby("harness")[["L_max_visible_template", "L_max_no_past_reasoning"]].describe(percentiles=[.5, .9, .99]).T.round(-2))


if __name__ == "__main__":
    main()
