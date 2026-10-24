"""Read-only scheduler instrumentation; accepted counts come from vLLM itself."""

import gzip
import json
import os
from pathlib import Path
import time

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class TraceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_file = gzip.open(os.environ["SPECDEC_TRACE_PATH"], "wt", compresslevel=1)
        self.eos_ids = set(json.loads(os.environ["SPECDEC_EOS_IDS"]))
        self.first_eos = {}
        self.step_index = 0
        self.last_flush = time.monotonic()
        self.schedule_times = {}
        self.last_emission = {}
        self.counts = {}
        self.progress = {}
        self.preemptions = 0
        self.peak_running_requests = 0
        self.peak_running_computed_tokens = 0
        self.progress_path = Path(os.environ["SPECDEC_TRACE_PATH"]).with_name("progress.json")
        from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
        tokens, capacity = get_kv_cache_capacity(self.vllm_config, self.kv_cache_config)
        required = float(os.environ.get("SPECDEC_MIN_KV_CAPACITY", "0"))
        self.progress_path.with_name("capacity.json").write_text(json.dumps(dict(
            kv_tokens=tokens, max_context_concurrency=capacity, required_concurrency=required), indent=2))
        if capacity < required:
            raise RuntimeError(f"KV cache fits {capacity:.2f} full-context requests; need {required:.2f}")

    def schedule(self, *args, **kwargs):
        started = time.perf_counter()
        result = super().schedule(*args, **kwargs)
        self.schedule_times[id(result)] = started
        return result

    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens,
                                num_accepted_tokens, num_invalid_spec_tokens, request_id):
        effective = num_draft_tokens - (num_invalid_spec_tokens or {}).get(request_id, 0)
        self.counts[request_id] = (effective, num_accepted_tokens)
        return super().make_spec_decoding_stats(
            spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
            num_invalid_spec_tokens, request_id,
        )

    def update_from_output(self, scheduler_output, model_runner_output):
        before = {}
        for request_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(request_id)
            if request is not None and not request.is_finished():
                before[request_id] = (request.num_tokens, request.num_output_tokens)
        self.counts = {}
        result = super().update_from_output(scheduler_output, model_runner_output)
        self.preemptions += len(scheduler_output.preempted_req_ids or ())
        running = [r for r in self.running if not r.is_finished()]
        self.peak_running_requests = max(self.peak_running_requests, len(running))
        self.peak_running_computed_tokens = max(self.peak_running_computed_tokens,
                                               sum(r.num_computed_tokens for r in running))
        arrived = time.perf_counter()
        elapsed = arrived - self.schedule_times.pop(id(scheduler_output))
        self.step_index += 1
        for batch in result.values():
            for output in batch.outputs:
                request_id = output.request_id
                if request_id not in before or not output.new_token_ids:
                    continue
                prefix, generated = before[request_id]
                tokens = output.new_token_ids
                first_eos = self.first_eos.get(request_id)
                phase = "post_eos" if first_eos is not None else "pre_eos"
                if first_eos is None:
                    for index, token in enumerate(tokens):
                        if token in self.eos_ids:
                            first_eos = generated + index
                            self.first_eos[request_id] = first_eos
                            phase = "crosses_eos"
                            break
                proposed, accepted = self.counts.get(request_id, (0, 0))
                scheduled_ids = scheduler_output.scheduled_spec_decode_tokens.get(request_id, [])
                draft_ids_available = all(token >= 0 for token in scheduled_ids)
                interarrival = arrived - self.last_emission.get(request_id, arrived)
                self.last_emission[request_id] = arrived
                record = {
                    "request_id": request_id, "step": self.step_index,
                    "prefix_tokens": prefix, "generated_before": generated,
                    "proposed": proposed, "accepted": accepted,
                    "emitted": len(tokens), "token_ids": tokens,
                    "draft_token_ids": scheduled_ids if draft_ids_available else None,
                    "draft_token_ids_available": draft_ids_available,
                    "scheduled_draft_token_ids": scheduled_ids,
                    "phase": phase, "first_eos_output_index": first_eos,
                    "step_seconds": interarrival if self.scheduler_config.async_scheduling else elapsed,
                    "scheduler_latency_seconds": elapsed,
                    "timing": "output_interarrival" if self.scheduler_config.async_scheduling else "scheduler_wall",
                    "unix_time": time.time(),
                    "is_prefill": generated == 0,
                    "num_nans_in_logits": (model_runner_output.num_nans_in_logits or {}).get(request_id),
                    "scheduled_requests": len(scheduler_output.num_scheduled_tokens),
                }
                self.trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                previous = self.progress.get(request_id, {})
                self.progress[request_id] = {
                    "context_tokens": prefix + len(tokens),
                    "generated_tokens": generated + len(tokens),
                    "first_eos_output_index": first_eos,
                    "proposed": previous.get("proposed", 0) + proposed,
                    "accepted": previous.get("accepted", 0) + accepted,
                    "nan_logits": previous.get("nan_logits", 0) + (record["num_nans_in_logits"] or 0),
                }
        if time.monotonic() - self.last_flush >= 10:
            self.trace_file.flush()
            self.write_progress()
            self.last_flush = time.monotonic()
        return result

    def write_progress(self):
        temporary = self.progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "updated_at": time.time(), "step": self.step_index,
            "preemptions": self.preemptions, "requests": self.progress,
            "peak_running_requests": self.peak_running_requests,
            "peak_running_computed_tokens": self.peak_running_computed_tokens,
        }, indent=2))
        temporary.replace(self.progress_path)

    def shutdown(self):
        try:
            self.trace_file.close()
            self.write_progress()
        finally:
            super().shutdown()


class TracingScheduler(TraceMixin, Scheduler):
    """Legacy synchronous control experiment."""


class TracingAsyncScheduler(TraceMixin, AsyncScheduler):
    """Observe official async scheduler accounting without changing its decisions."""
