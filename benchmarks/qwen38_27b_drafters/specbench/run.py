"""Run one target+drafter over a prompt shard in vLLM and record per-round acceptance.

Based on the issue11 long-context runner (PavelPaha/specdec_research,
benchmarks/long_context): same scheduler trace, continuous batching.
Additions: natural or masked EOS, sharding with resume, target top-k logprobs,
teacher forcing on a reference trajectory (--ref, see force.py).

Outputs in --output:
  metadata.json                 config, engine args, package versions
  rounds-<attempt>.jsonl.gz     one line per verification round (trace.py)
  requests-s<k>of<n>.jsonl      one line per finished request (+ attempt)
  outputs/<id>.json.gz          generated token ids (+ compact logprobs stats if enabled)
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path


def load_prompts(path, shard, num_shards):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: -r["prompt_tokens"] - r["max_new_tokens"])  # long first
    return rows[shard::num_shards]


def logprob_stats(entries, chosen, eos_ids):
    """Compact per-token stats from vLLM top-k logprobs (raw target distribution)."""
    out = dict(chosen=[], top1=[], top1_id=[], top2=[], eos=[], eos_rank=[], entropy_topk=[])
    for token, entry in zip(chosen, entries):
        items = sorted(entry.items(), key=lambda kv: -kv[1].logprob)
        lps = [lp.logprob for _, lp in items]
        out["chosen"].append(entry[token].logprob if token in entry else None)
        out["top1"].append(lps[0])
        out["top1_id"].append(items[0][0])
        out["top2"].append(lps[1] if len(lps) > 1 else None)
        eos = [(lp.logprob, lp.rank) for t, lp in entry.items() if t in eos_ids]
        best = max(eos) if eos else None
        out["eos"].append(best[0] if best else None)
        out["eos_rank"].append(best[1] if best else None)
        probs = [math.exp(x) for x in lps]
        z = sum(probs)
        out["entropy_topk"].append(-sum(p / z * math.log(p / z) for p in probs if p > 0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True, help="'ar' or a key of config['drafters']")
    parser.add_argument("--budget", type=int, default=7, help="draft tokens per round")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, help="cap for smoke tests")
    parser.add_argument("--max-num-seqs", type=int, help="override config max_num_seqs")
    parser.add_argument("--ref", help="teacher forcing: run dir whose outputs/<id>.json.gz is the reference text")
    parser.add_argument("--kv-tokens", type=int, help="KV cache capacity in tokens for --kv-fill (vLLM startup log)")
    parser.add_argument("--engine-json", help="extra vLLM engine args as JSON, e.g. a larger cudagraph capture size")
    parser.add_argument("--kv-fill", type=float, default=0.0,
                        help="forced runs: admit requests longest-first while their final lengths fit into this "
                             "fraction of the KV cache (no preemption, largest batch); 0 = plain max_num_seqs refill")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    output = Path(args.output).resolve()
    (output / "outputs").mkdir(parents=True, exist_ok=True)
    attempt = f"s{args.shard}of{args.num_shards}-{int(time.time())}"
    requests_path = output / f"requests-s{args.shard}of{args.num_shards}.jsonl"
    done = set()
    for f in output.glob("requests-*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if not args.ref or rec.get("forced_ok"):  # a forced request counts only if it reproduced the reference
                    done.add(rec["id"])
    prompts = [p for p in load_prompts(args.prompts, args.shard, args.num_shards)
               if p["id"] not in done]
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        print("nothing to do", flush=True)
        return

    refs = {}
    if args.ref:  # teacher forcing: the target emits exactly the reference tokens
        force_dir = output / "force"
        force_dir.mkdir(exist_ok=True)
        import numpy as np
        for p in prompts:
            with gzip.open(Path(args.ref) / "outputs" / f"{p['id']}.json.gz", "rt") as handle:
                ids = json.load(handle)["token_ids"]
            if args.max_new_tokens:
                ids = ids[: args.max_new_tokens]
            if p.get("force_tokens"):  # per-prompt forcing length (prepare_forced.py)
                ids = ids[: p["force_tokens"]]
            refs[p["id"]] = ids
            np.save(force_dir / f"{p['id']}.npy", np.array([p["prompt_tokens"]] + ids, dtype=np.int64))
        plugin = str(Path(__file__).resolve().parent / "vllm_plugin")
        os.environ["PYTHONPATH"] = plugin + os.pathsep + os.environ.get("PYTHONPATH", "")
        import sys
        sys.path.insert(0, plugin)
        os.environ["SPECDEC_FORCE_DIR"] = str(force_dir)
    os.environ["SPECDEC_TRACE_PATH"] = str(output / f"rounds-{attempt}.jsonl.gz")
    os.environ.setdefault("VLLM_COMPUTE_NANS_IN_LOGITS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer, GenerationConfig
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    target, revision = config["target"], config["target_revision"]
    tokenizer = AutoTokenizer.from_pretrained(target, revision=revision)
    eos_ids = GenerationConfig.from_pretrained(target, revision=revision).eos_token_id
    eos_ids = sorted(set(([eos_ids] if isinstance(eos_ids, int) else list(eos_ids or []))
                         + [tokenizer.eos_token_id]))
    os.environ["SPECDEC_EOS_IDS"] = json.dumps(eos_ids)
    eos_policy = config["eos"]  # "natural" or "mask" (ignore_eos + EOS banned in target)
    os.environ["SPECDEC_EOS_POLICY"] = "bad_words" if eos_policy == "mask" else "natural"

    spec = None
    if args.method != "ar":
        spec = dict(config["drafters"][args.method])
        spec.setdefault("method", args.method)
        if spec["method"] is None:
            spec.pop("method")  # let vLLM detect the checkpoint format (speculators)
        spec.update(num_speculative_tokens=args.budget, max_model_len=config["max_model_len"])
    engine_args = dict(
        model=target, revision=revision, tokenizer_revision=revision, dtype=config["dtype"],
        tensor_parallel_size=config["tensor_parallel_size"],
        max_model_len=config["max_model_len"], max_num_seqs=args.max_num_seqs or config["max_num_seqs"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enable_prefix_caching=False, async_scheduling=False, language_model_only=True,
        speculative_config=spec,
        scheduler_cls="specbench.trace.TracingScheduler", seed=config["seed"],
        disable_log_stats=False, generation_config="vllm",
    )
    engine_args.update(config.get("engine_overrides", {}).get(args.method, {}))
    if args.engine_json:
        engine_args.update(json.loads(args.engine_json))
    logprobs = config.get("logprobs", {}).get(args.method)
    metadata = dict(
        args=vars(args), attempt=attempt, config=config, engine_args=engine_args,
        eos_token_ids=eos_ids, eos_policy=eos_policy, logprobs=logprobs, forced_from=args.ref,
        prompt_file_sha256=hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        packages={p: importlib.metadata.version(p)
                  for p in ["vllm", "torch", "transformers"]},
        started_at=time.time(),
    )
    (output / f"metadata-{attempt}.json").write_text(json.dumps(metadata, indent=2))
    llm = LLM(**engine_args)
    engine = llm.llm_engine

    states, alias = {}, {}
    eos_set = set(eos_ids)
    pending = iter(prompts)
    bad_words = [tokenizer.convert_ids_to_tokens(i) for i in eos_ids] if eos_policy == "mask" else None

    def submit(prompt):
        per_prompt = eos_policy == "natural" or config.get("budget_from_prompt")
        max_tokens = prompt["max_new_tokens"] if per_prompt else config["max_new_tokens"]
        max_tokens = min(max_tokens, config["max_model_len"] - prompt["prompt_tokens"])
        if args.max_new_tokens:
            max_tokens = min(max_tokens, args.max_new_tokens)
        if args.ref:
            max_tokens = len(refs[prompt["id"]])
        sampling = SamplingParams(
            temperature=config["temperature"], top_p=1.0, top_k=-1, seed=config["seed"],
            ignore_eos=eos_policy == "mask", bad_words=bad_words, max_tokens=max_tokens,
            logprobs=logprobs, detokenize=False, output_kind=RequestOutputKind.DELTA)
        internal = engine.add_request(prompt["id"], {"prompt_token_ids": prompt["prompt_token_ids"]}, sampling)
        states[internal] = dict(prompt=prompt, ids=[], lps=None, started=time.time(),
                                max_tokens=max_tokens, engine_id=internal)
        alias[internal] = alias[prompt["id"]] = internal
        print(f"START {prompt['id']} input={prompt['prompt_tokens']} max_new={max_tokens}", flush=True)

    def finish(state, reason):
        prompt, ids = state["prompt"], state["ids"]
        payload = dict(id=prompt["id"], token_ids=ids)
        if logprobs:
            payload["logprobs"] = state["lps"]
        summary = dict(
            id=prompt["id"], engine_request_id=state["engine_id"], attempt=attempt,
            dataset=prompt["dataset"], domain=prompt["domain"], prompt_tokens=prompt["prompt_tokens"],
            generated_tokens=len(ids), max_new_tokens=state["max_tokens"], finish_reason=reason,
            first_eos=next((i for i, t in enumerate(ids) if t in eos_ids), None),
            seconds=time.time() - state["started"],
            output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
        if args.ref:
            ref = refs[prompt["id"]]
            summary.update(forced=True, ref_tokens=len(ref), forced_ok=ids == ref,
                           first_mismatch=next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), None))
        # a failed forced attempt must not take the canonical outputs/<id> name used by the analysis
        ok = not args.ref or summary["forced_ok"]
        path = output / "outputs" / f"{prompt['id']}.json.gz" if ok else output / "outputs-failed" / f"{prompt['id']}.{attempt}.json.gz"
        path.parent.mkdir(exist_ok=True)
        with gzip.open(path, "wt") as handle:
            json.dump(payload, handle)
        with requests_path.open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        if args.ref and not summary["forced_ok"]:
            # forcing must reproduce the reference exactly; anything else means the patch is not active
            raise RuntimeError(f"forced output differs from reference: {prompt['id']} at {summary['first_mismatch']}")

    max_seqs = args.max_num_seqs or config["max_num_seqs"]
    kv_budget = None
    if args.kv_fill and args.ref:
        cache = engine.vllm_config.cache_config
        # num_gpu_blocks * block_size overstates capacity when drafter layers keep their own KV groups;
        # prefer the "GPU KV cache size: N tokens" figure vLLM logs at startup, passed as --kv-tokens
        kv_tokens = args.kv_tokens or cache.num_gpu_blocks * cache.block_size
        kv_budget = int(args.kv_fill * kv_tokens)
        # exact final length of every request is known: prompt + forced reference (+ one draft block)
        need = {p["id"]: p["prompt_tokens"] + len(refs[p["id"]]) + args.budget + 1 for p in prompts}
        queue = sorted(prompts, key=lambda p: -need[p["id"]])
        print(f"KV packing: {kv_tokens} cache tokens, budget {kv_budget}, max_seqs {max_seqs}, "
              f"{len(queue)} requests, {sum(need.values())} tokens", flush=True)

    def refill():
        if kv_budget is None:
            while len(states) < max_seqs:
                prompt = next(pending, None)
                if prompt is None:
                    return
                submit(prompt)
            return
        used = sum(need[s["prompt"]["id"]] for s in states.values())
        # strictly longest first: letting short requests jump ahead leaves long ones to run alone at the end
        while queue and len(states) < max_seqs and (used + need[queue[0]["id"]] <= kv_budget or not states):
            used += need[queue[0]["id"]]
            submit(queue.pop(0))

    refill()
    while states:
        for result in engine.step():
            key = alias[result.request_id]
            state = states[key]
            reason = None
            for completion in result.outputs:
                new = list(completion.token_ids)
                state["ids"].extend(new)
                if logprobs and new:
                    stats = logprob_stats(completion.logprobs[-len(new):], new, eos_set)
                    if state["lps"] is None:
                        state["lps"] = stats
                    else:
                        for k, v in stats.items():
                            state["lps"][k].extend(v)
                reason = completion.finish_reason or reason
            if reason is not None:
                finish(states.pop(key), reason)
        refill()
    engine.engine_core.shutdown()
    (output / f"metadata-{attempt}.json").write_text(
        json.dumps({**metadata, "finished_at": time.time(), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
