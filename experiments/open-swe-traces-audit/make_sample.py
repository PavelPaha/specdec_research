"""Pick 8 trajectories per harness (min, p10, p25, p50, p75, p90, p99, max of L_max_visible_template)
from the downloaded shards and save them (all columns) to one parquet.

usage: make_sample.py <trajectories.csv> <shards_root> <out.parquet>
"""
import glob, sys
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

t = pd.read_csv(sys.argv[1])
tabs, picks = [], []
for h, g in t.groupby("harness"):
    g = g.sort_values("L_max_visible_template").reset_index(drop=True)
    idx = sorted(set(int(round(q * (len(g) - 1))) for q in (0, .1, .25, .5, .75, .9, .99, 1)))
    sel = g.iloc[idx]
    path = glob.glob(f"{sys.argv[2]}/data/{h}/qwen38_27b/*/{sel.file.iloc[0]}")[0]
    tb = pq.read_table(path).take(sel.row.tolist())
    tb = tb.append_column("harness", pa.array([h] * len(sel)))
    tabs.append(tb)
    picks.append(sel[["harness", "trajectory_id", "file", "row", "n_llm_calls", "L_max_visible_template"]])
pq.write_table(pa.concat_tables(tabs), sys.argv[3])
print(pd.concat(picks).to_string(index=False))
