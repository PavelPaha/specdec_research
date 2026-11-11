"""Build tokenized prompt files for the Qwen3.8-27B drafter benchmark.

Run from this directory with network access and a Hugging Face cache:
    python prepare_data.py --out data/prompts

Writes one JSONL per panel. Every record keeps chat messages, exact prompt token IDs
(chat template, thinking enabled), source and revision, and the output budget.
"""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

TARGET = "Qwen/Qwen3.8-27B"
TARGET_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MAX_MODEL_LEN = 262144
HERE = Path(__file__).parent
API = HfApi()
BINS = [(4_000, 8_000), (8_000, 16_000), (16_000, 32_000), (32_000, 64_000),
        (64_000, 128_000), (128_000, 250_000)]
MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

tok = AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REVISION)


def rank(key):
    return hashlib.sha256(str(key).encode()).hexdigest()


def length_bin(n):
    for lo, hi in BINS:
        if lo <= n < hi:
            return f"{lo // 1000}k_{hi // 1000}k"
    return None


def record(dataset, domain, source_id, messages, max_new_tokens, source, **extra):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  enable_thinking=True, tokenize=True)
    if hasattr(ids, "input_ids") or isinstance(ids, dict):
        ids = ids["input_ids"]
    return dict(
        id=f"{dataset}-{rank(source_id)[:16]}", dataset=dataset, domain=domain,
        source_id=str(source_id), source=source, messages=messages,
        prompt_token_ids=ids, prompt_tokens=len(ids),
        length_bin=length_bin(len(ids)),
        max_new_tokens=min(max_new_tokens, MAX_MODEL_LEN - len(ids)),
        tokenizer=dict(model=TARGET, revision=TARGET_REVISION, enable_thinking=True),
        **extra,
    )


def user(text):
    return [{"role": "user", "content": text}]


def pick(rows, n, key=lambda r: r["source_id"]):
    return sorted(rows, key=lambda r: rank(key(r)))[:n]


# ---------- short panel ----------

def issue11(n=20):
    """Same messages as PavelPaha/specdec_research issue11 branch (first n per domain)."""
    names = dict(math="math500", code="humaneval", chat="ultrachat", reasoning="bbh")
    domains = dict(math="math", code="code", chat="chat", reasoning="reasoning")
    rows, seen = [], {}
    for line in (HERE / "data/sources/issue11_prompts.jsonl").read_text().splitlines():
        r = json.loads(line)
        if seen.get(r["domain"], 0) >= n:
            continue
        seen[r["domain"]] = seen.get(r["domain"], 0) + 1
        rows.append(record(names[r["domain"]], domains[r["domain"]], r["source_id"],
                           r["messages"], 32768,
                           dict(dataset=r["source"], revision=r["revision"]),
                           upstream_id=r["id"]))
    return rows


def matharena(name, dataset):
    rev = API.dataset_info(name).sha
    ds = load_dataset(name, split="train", revision=rev)
    return [record(dataset, "math", r["problem_idx"], user(r["problem"] + MATH_SUFFIX), 32768,
                   dict(dataset=name, revision=rev), answer=str(r["answer"])) for r in ds]


def livecodebench(n=20):
    name = "livecodebench/code_generation_lite"
    rev = API.dataset_info(name).sha
    rows = []
    for filename in ["test.jsonl"] + [f"test{i}.jsonl" for i in range(2, 7)]:
        path = hf_hub_download(name, filename, repo_type="dataset", revision=rev)
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r["difficulty"].lower() == "hard":
                rows.append(r)
    rows = sorted({r["question_id"]: r for r in rows}.values(),
                  key=lambda r: r["contest_date"], reverse=True)[:n]
    out = []
    for r in rows:
        text = ("Write a Python 3 solution to this programming problem. "
                "Return the complete program in a Python code block.\n\n" + r["question_content"])
        if r.get("starter_code"):
            text += "\n\nStarter code:\n" + r["starter_code"]
        out.append(record("livecodebench", "code", r["question_id"], user(text), 32768,
                          dict(dataset=name, revision=rev, file="test*.jsonl"),
                          contest_date=r["contest_date"]))
    return out


def wildchat(n=20):
    rows = [json.loads(l) for l in (HERE / "data/sources/wildchat_everyday.jsonl").read_text().splitlines()]
    return [record("wildchat", "chat", r["source_id"], r["messages"], 32768,
                   dict(dataset=r["source"], revision=r["revision"],
                        selection="manually reviewed everyday conversations"))
            for r in rows[:n]]


# ---------- long-input panel ----------

def issue10(n=10):
    """Prompts from PavelPaha/specdec_research issue #10 (mikcoff), without code."""
    names = dict(academic="arxiv", government="govreport", legal="multilexsum",
                 literature="narrativeqa", conversation="mrcr")
    out = []
    for domain, dataset in names.items():
        for path in sorted((HERE / "data/issue10" / domain).glob("*.json")):
            rows = json.loads(path.read_text())
            for r in sorted(rows, key=lambda r: rank(r["id"]))[:n]:
                msgs = r["request"].get("messages") or user(r["request"]["prompt"])
                out.append(record(dataset, domain, r["id"], msgs, 8192,
                                  dict(dataset=r["source"]["dataset"],
                                       revision=r["source"].get("revision"),
                                       issue10_file=f"{domain}/{path.name}"),
                                  issue10_length_bin=r["length_bin"]))
    return out


def swe_template():
    for path in (HERE / "data/issue10/code").glob("*.json"):
        for r in json.loads(path.read_text()):
            if r["id"].startswith("swe-bench"):
                text = r["request"]["messages"][-1]["content"]
                return text[: text.index("<issue>")], text[text.index("</code>"):]
    raise RuntimeError("SWE-bench oracle template not found in issue10 data")


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def swe_live(per_bin=10):
    """SWE-bench-Live with oracle files (files touched by the gold patch), SWE-bench oracle format."""
    name = "SWE-bench-Live/SWE-bench-Live"
    rev = API.dataset_info(name).sha
    head, tail = swe_template()
    ds = load_dataset(name, split="verified", revision=rev)
    counts, out = {}, []
    for r in sorted(ds, key=lambda r: rank(r["instance_id"])):
        files = sorted(set(re.findall(r"^diff --git a/(\S+) b/", r["patch"], re.M)))
        try:
            code = ""
            for f in files:
                body = fetch(f"https://raw.githubusercontent.com/{r['repo']}/{r['base_commit']}/{f}")
                numbered = "\n".join(f"{i} {line}" for i, line in enumerate(body.split("\n"), 1))
                code += f"[start of {f}]\n{numbered}\n[end of {f}]\n"
        except Exception as error:  # deleted/renamed/binary files: skip the instance
            print("skip", r["instance_id"], repr(error), flush=True)
            continue
        text = f"{head}<issue>\n{r['problem_statement']}\n</issue>\n<code>\n{code}{tail}"
        rec = record("swebench_live", "code", r["instance_id"], user(text), 16384,
                     dict(dataset=name, revision=rev, split="verified",
                          oracle_files=files, base_commit=r["base_commit"]),
                     created_at=r["created_at"])
        b = rec["length_bin"]
        if b is None or b == "128k_250k" or counts.get(b, 0) >= per_bin:
            continue
        counts[b] = counts.get(b, 0) + 1
        out.append(rec)
        if len(counts) == 5 and all(v >= per_bin for v in counts.values()):
            break
    print("swe_live bins", counts, flush=True)
    return out


def mrcr_long(n=10):
    name = "openai/mrcr"
    rev = API.dataset_info(name).sha
    ds = load_dataset(name, split="train", revision=rev)
    rows = [(i, r) for i, r in enumerate(ds) if 450_000 <= r["n_chars"] <= 1_100_000]
    out = []
    for i, r in sorted(rows, key=lambda x: rank(x[0])):
        rec = record("mrcr", "conversation", f"row{i}", json.loads(r["prompt"]), 8192,
                     dict(dataset=name, revision=rev, row=i), n_needles=r["n_needles"])
        if rec["length_bin"] == "128k_250k":
            out.append(rec)
        if len(out) >= n:
            break
    return out


def longbench_v2_long(n=20):
    name = "THUDM/LongBench-v2"
    rev = API.dataset_info(name).sha
    ds = load_dataset(name, split="train", revision=rev)
    template = ("Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\n"
                "What is the correct answer to this question: {question}\nChoices:\n"
                "(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
                "Format your response as follows: \"The correct answer is (insert answer here)\".")
    by_domain, out = {}, []
    for r in sorted(ds, key=lambda r: rank(r["_id"])):
        if r["length"] == "short":
            continue
        text = template.format(context=r["context"], question=r["question"], A=r["choice_A"],
                               B=r["choice_B"], C=r["choice_C"], D=r["choice_D"])
        if not 120_000 * 3 <= len(text) <= 260_000 * 5:
            continue
        rec = record("longbench_v2", "longbench_v2", r["_id"], user(text), 8192,
                     dict(dataset=name, revision=rev), lb_domain=r["domain"], answer=r["answer"])
        if rec["length_bin"] != "128k_250k" or by_domain.get(r["domain"], 0) >= 4:
            continue
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
        out.append(rec)
        if len(out) >= n:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/prompts")
    parser.add_argument("--parts", nargs="+", default=["short", "long"])
    args = parser.parse_args()
    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    builders = dict(
        short=lambda: (issue11() + matharena("MathArena/aime_2026", "aime2026")
                       + matharena("MathArena/hmmt_feb_2026", "hmmt2026")
                       + livecodebench() + wildchat()),
        long=lambda: issue10() + swe_live() + mrcr_long() + longbench_v2_long(),
    )
    for part in args.parts:
        rows = builders[part]()
        assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
        path = out / f"{part}.jsonl"
        with path.open("w") as handle:
            for r in rows:
                handle.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        summary = {}
        for r in rows:
            key = f"{r['dataset']}/{r['length_bin']}"
            summary[key] = summary.get(key, 0) + 1
        print(part, len(rows), json.dumps(summary, indent=1), flush=True)
    # Panel files used by the runs: natural = everything; noeos = issue11 short sets.
    if (out / "short.jsonl").exists() and (out / "long.jsonl").exists():
        short = (out / "short.jsonl").read_text()
        (out / "natural.jsonl").write_text(short + (out / "long.jsonl").read_text())
        issue11_sets = {"math500", "humaneval", "ultrachat", "bbh"}
        (out / "noeos.jsonl").write_text("".join(
            l + "\n" for l in short.splitlines() if json.loads(l)["dataset"] in issue11_sets))


if __name__ == "__main__":
    main()
