"""Keep request slots occupied; persist each completion without a batch barrier."""

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import time


def drive(engine, prompts, concurrency, submit, consume):
    pending = iter(prompts)
    active = set()

    def refill():
        while len(active) < concurrency:
            prompt = next(pending, None)
            if prompt is None:
                break
            request_id = submit(prompt)
            assert request_id not in active
            active.add(request_id)

    refill()
    while active:
        if not engine.has_unfinished_requests():
            raise RuntimeError("Engine lost active requests")
        for result in engine.step():
            finished_id = consume(result)
            if finished_id is not None:
                active.remove(finished_id)
        refill()


def run_continuous(llm, tokenizer, prompts, config, args, eos_ids, output):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    engine = llm.llm_engine
    states = {}
    lookup = {}
    total_tokens = 0
    completed = 0
    started = time.perf_counter()
    last_sample = started
    last_tokens = 0
    text_jobs = []

    def save_text(request_id, ids):
        with gzip.open(output / f"{request_id}.text.txt.gz", "wt") as handle:
            handle.write(tokenizer.decode(ids, skip_special_tokens=False))

    with ThreadPoolExecutor(max_workers=1) as writer:
        def submit(prompt):
            rid = prompt["id"]
            prompt_ids = prompt["prompt_token_ids"]
            max_tokens = config["max_model_len"] - len(prompt_ids)
            if args.max_new_tokens:
                max_tokens = min(max_tokens, args.max_new_tokens)
            sampling = SamplingParams(
                temperature=config["temperature"], top_p=1.0, top_k=-1,
                seed=config["seed"], ignore_eos=True, max_tokens=max_tokens,
                detokenize=False, output_kind=RequestOutputKind.DELTA,
            )
            state = dict(prompt=prompt, generated_ids=[], started=time.perf_counter(),
                         started_at=time.time(),
                         first_token_time=None, max_tokens=max_tokens,
                         stream=gzip.open(output / f"{rid}.tokens.jsonl.gz", "wt", compresslevel=1))
            internal = engine.add_request(rid, {"prompt_token_ids": prompt_ids}, sampling)
            state["engine_request_id"] = internal
            states[rid] = state
            lookup[rid] = lookup[internal] = rid
            with (output / "requests.started.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(id=rid, engine_request_id=internal,
                    domain=prompt["domain"], max_new_tokens=max_tokens,
                    started_at=state["started_at"], active_at_submission=len(states),
                    concurrency_limit=config["max_num_seqs"], batching="continuous")) + "\n")
            print(f"START {rid} input={len(prompt_ids)} max_new={max_tokens}", flush=True)
            return rid

        def consume(result):
            nonlocal total_tokens, completed, last_sample, last_tokens
            rid = lookup[result.request_id]
            state = states[rid]
            finish_reason = None
            for completion in result.outputs:
                tokens = list(completion.token_ids)
                if tokens:
                    now = time.perf_counter()
                    state["first_token_time"] = state["first_token_time"] or now
                    state["generated_ids"].extend(tokens)
                    total_tokens += len(tokens)
                    state["stream"].write(json.dumps(dict(
                        elapsed=now-state["started"], token_ids=tokens)) + "\n")
                finish_reason = completion.finish_reason or finish_reason
            now = time.perf_counter()
            if now - last_sample >= 10:
                with (output / "throughput.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(unix_time=time.time(), elapsed=now-started,
                        interval_seconds=now-last_sample, tokens=total_tokens-last_tokens,
                        aggregate_tokens_per_second=(total_tokens-last_tokens)/(now-last_sample),
                        active_requests=len(states), completed_requests=completed)) + "\n")
                last_sample, last_tokens = now, total_tokens
            if finish_reason is None:
                return None
            ids = state["generated_ids"]
            prompt = state["prompt"]
            elapsed = now-state["started"]
            state["stream"].close()
            summary = dict(
                id=rid, engine_request_id=state["engine_request_id"], domain=prompt["domain"],
                prompt_tokens=prompt["prompt_tokens"], generated_tokens=len(ids),
                final_context=prompt["prompt_tokens"]+len(ids), max_new_tokens=state["max_tokens"],
                seconds=elapsed, tokens_per_second=len(ids)/elapsed,
                started_at=state["started_at"], completed_at=time.time(),
                concurrency_limit=config["max_num_seqs"], batching="continuous",
                ttft_seconds=state["first_token_time"]-state["started"] if state["first_token_time"] else None,
                first_eos_output_index=next((i for i,t in enumerate(ids) if t in eos_ids),None),
                finish_reason=finish_reason,
                output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                completed_requested_length=len(ids)==state["max_tokens"],
            )
            with (output / "requests.jsonl").open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
            if not summary["completed_requested_length"]:
                raise RuntimeError(f"{rid}: stopped before requested length")
            text_jobs.append(writer.submit(save_text, rid, ids))
            del lookup[state["engine_request_id"]]
            lookup.pop(rid, None)
            del states[rid]
            completed += 1
            return rid

        try:
            drive(engine, prompts, config["max_num_seqs"], submit, consume)
            seconds = time.perf_counter()-started
        finally:
            for state in states.values():
                state["stream"].close()
        for job in text_jobs:
            job.result()
    (output / "workload.json").write_text(json.dumps(dict(
        batching="continuous", requests=completed, generated_tokens=total_tokens,
        seconds=seconds, aggregate_tokens_per_second=total_tokens/seconds,
    ), indent=2))
