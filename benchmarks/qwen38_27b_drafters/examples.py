"""Per-request example files for the report's example viewer.

Writes per panel:
  <out>/index.js                     EXINDEX({...}): requests with dataset, prompt length, per-method summary
  <out>/<panel>/<id>.js              EXDATA(key, base64 gzip JSON): decoded text of every method's trajectory with per-token lengths,
                                     verification rounds, prompt head/tail, detected cycles with exact spans, first token differing
                                     from AR; AR with logged logprobs also EOS-top1 positions

Usage: python examples.py --results results --prompts data/prompts --tables tables --out report/examples
"""

import argparse
import base64
import codecs
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import CYCLE_STEP, periodic

PROMPT_HEAD, PROMPT_TAIL = 512, 2048
G = {}  # shared with forked workers: token bytes table, rounds, prompts


def bytes_to_unicode():
    """GPT-2 byte-level BPE byte <-> printable unicode map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_bytes(model, revision):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    dec = {c: b for b, c in bytes_to_unicode().items()}
    special = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    n = max(len(tok), max(special, default=0) + 1)
    table = []
    for i, t in enumerate(tok.convert_ids_to_tokens(list(range(n)))):
        if i in special:
            table.append(special[i].encode())
        elif t is None:
            table.append(b"")
        else:
            table.append(bytes(dec[c] for c in t) if all(c in dec for c in t) else t.encode())
    return tok, table


def decode(ids):
    """Text and per-token lengths (UTF-16 units, as JS slices strings); multi-byte chars go to the token that completes them."""
    d = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pieces = [d.decode(G["bytes"][i] if 0 <= i < len(G["bytes"]) else b"") for i in ids]
    tail = d.decode(b"", final=True)
    if tail and pieces:
        pieces[-1] += tail
    return "".join(pieces), [len(p.encode("utf-16-le")) // 2 for p in pieces]


def cycles(ids):
    x = np.asarray(ids, dtype=np.int64)
    out = []
    for t in range(2048, len(x) + 1, CYCLE_STEP):
        p = periodic(x[t - 2048:t], range(1, 129)) or periodic(x[max(0, t - 8192):t], range(129, 2049))
        if p:
            out.append([t, p])
    return out


BOUNDARY = 8  # a cycle starting this close before </think> repeats the end of reasoning together with the answer
SEG_GAP = 3  # mismatching tokens tolerated inside a cycle (counters, numbering)


def segments(ids, cyc):
    """Exact cycle spans [start, end, period] around each detector hit.

    Inside a span every token equals the token one period earlier, except gaps of <= SEG_GAP tokens;
    start is the first token of the first repetition, end is exclusive.
    """
    x = np.asarray(ids, dtype=np.int64)
    runs, out = {}, []
    for t, p in cyc:
        if out and out[-1][2] == p and t - p <= out[-1][1]:
            continue  # this checkpoint lies in the span already found
        if p not in runs:
            m = np.concatenate([[False], x[p:] == x[:-p], [False]]).astype(np.int8)
            st, en = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)  # match runs [st, en)
            keep = np.concatenate([[True], st[1:] - en[:-1] > SEG_GAP])
            runs[p] = (st[keep], np.concatenate([en[:-1][keep[1:]], en[-1:]]))
        st, en = runs[p]
        w0 = max(0, t - (2048 if p <= 128 else 8192))
        ov = np.minimum(en, t - p) - np.maximum(st, w0)  # overlap with the detector window
        if not len(ov) or ov.max() <= 0:
            continue
        k = int(ov.argmax())
        a, b = int(st[k]), int(en[k]) + p
        if out and a < out[-1][1] and min(b, out[-1][1]) - a >= (b - a) / 2:
            continue  # mostly inside the previous span (e.g. a multiple of its period)
        out.append([a, b, p])
    return out


def build(args):
    panel, rid, files, out = args
    prompt = G["prompts"][rid]
    pids = prompt["prompt_token_ids"]
    head, _ = decode(pids[:PROMPT_HEAD])
    tail, _ = decode(pids[-PROMPT_TAIL:]) if len(pids) > PROMPT_HEAD else ("", [])
    doc = dict(id=rid, dataset=prompt["dataset"], domain=prompt["domain"], prompt_tokens=len(pids),
               prompt_head=head, prompt_tail=tail, prompt_skipped=max(0, len(pids) - PROMPT_HEAD - PROMPT_TAIL),
               methods={})
    ids_by = {}
    for m, f in files.items():
        with gzip.open(f, "rt") as h:
            ids_by[m] = (json.load(h))
    ar_ids = ids_by["ar"]["token_ids"] if "ar" in ids_by else None
    if ar_ids is not None and prompt.get("force_tokens"):
        ar_ids = ar_ids[: prompt["force_tokens"]]
    row = dict(id=rid, dataset=prompt["dataset"], domain=prompt["domain"], p=len(pids),
               max_new=prompt.get("max_new_tokens"), m={})
    for m, d in ids_by.items():
        ids = d["token_ids"]
        if prompt.get("force_tokens"):  # forced panel: the common span shared by every method
            ids = ids[: prompt["force_tokens"]]
            if d.get("logprobs"):
                d["logprobs"] = {k: v[: prompt["force_tokens"]] for k, v in d["logprobs"].items()}
        text, lens = decode(ids)
        e = dict(t=text, l=lens, cyc=cycles(ids))
        e["seg"] = segments(ids, e["cyc"])
        e["think"] = ids.index(G["think"]) if G["think"] in ids else None  # first </think>
        r = G["rounds"].get((m, rid))
        if r is not None:
            e["g"], e["a"] = r
        lp = d.get("logprobs")
        if lp and lp.get("eos_rank"):
            e["eos"] = [i for i, k in enumerate(lp["eos_rank"]) if k == 1]
            row["eos"] = e["eos"][0] if e["eos"] else None
        div = None
        if ar_ids is not None and m != "ar":
            n = min(len(ids), len(ar_ids))
            neq = np.flatnonzero(np.asarray(ids[:n]) != np.asarray(ar_ids[:n]))
            div = int(neq[0]) if len(neq) else (None if len(ids) == len(ar_ids) else n)
            e["div"] = div
        doc["methods"][m] = e
        acc = round(sum(r[1]) / (7 * len(r[1])), 4) if r and m != "ar" else None  # every round proposes 7
        c0 = e["seg"][0] if e["seg"] else None
        # where the first cycle starts relative to the end of reasoning: none / think / boundary / answer
        # ("boundary": a few tokens before </think>, i.e. the repeated unit is "end of reasoning + </think> + answer")
        loc = ("none" if c0 is None else "think" if e["think"] is None or c0[0] < e["think"] - BOUNDARY
               else "boundary" if c0[0] <= e["think"] else "answer")
        # [answer length, acceptance, first cycle start (generated tokens), its period, first token differing from AR,
        #  first </think>, cycle location]
        row["m"][m] = [len(ids), acc, c0[0] if c0 else None, c0[2] if c0 else None, div, e["think"], loc]
    data = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode()
    # script-tag (JSONP) files so the viewer also works from file:// where fetch is blocked
    blob = base64.b64encode(gzip.compress(data, compresslevel=6)).decode()
    (out / panel / f"{rid}.js").write_text(f"EXDATA({json.dumps(panel + '/' + rid)},\"{blob}\");\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--prompts", default="data/prompts")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--config", default="configs/natural.json")
    ap.add_argument("--out", default="report/examples")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok, G["bytes"] = token_bytes(cfg["target"], cfg.get("target_revision"))
    G["think"] = tok.convert_tokens_to_ids("</think>")
    probe = list(range(1000, 1300))
    assert decode(probe)[0] == tok.decode(probe), "byte table does not reproduce tokenizer.decode"
    out = Path(a.out)
    index = {}
    for panel in ["natural", "noeos", "forced"]:
        root = Path(a.results) / panel
        if not root.exists():
            continue
        G["prompts"] = {}
        with open(Path(a.prompts) / f"{panel}.jsonl") as h:
            for line in h:
                p = json.loads(line)
                G["prompts"][p["id"]] = p
        rounds = pd.read_parquet(Path(a.tables) / panel / "rounds.parquet",
                                 columns=["method", "id", "gen", "proposed", "accepted"])
        rounds = rounds[rounds.proposed > 0].sort_values(["method", "id", "gen"])
        G["rounds"] = {k: (g.gen.tolist(), g.accepted.tolist()) for k, g in rounds.groupby(["method", "id"])}
        files = {}
        for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in (mdir / "outputs").glob("*.json.gz"):
                files.setdefault(f.name.removesuffix(".json.gz"), {})[mdir.name] = f
        (out / panel).mkdir(parents=True, exist_ok=True)
        jobs = [(panel, rid, fs, out) for rid, fs in sorted(files.items()) if rid in G["prompts"]]
        with ProcessPoolExecutor(a.workers) as pool:
            index[panel] = list(pool.map(build, jobs, chunksize=4))
        print(panel, len(index[panel]), "requests", flush=True)
    (out / "index.js").write_text("EXINDEX(" + json.dumps(index, ensure_ascii=False, separators=(",", ":")) + ");\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
