"""Read-only scheduler instrumentation; accepted counts come from vLLM itself."""

import gzip
import json
import os
import time

from vllm.v1.core.sched.scheduler import Scheduler


class TracingScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scheduler_config.async_scheduling:
            raise ValueError("Tracing requires synchronous scheduling")
        self.trace_file = gzip.open(os.environ["SPECDEC_TRACE_PATH"], "wt", compresslevel=1)
        self.eos_ids = set(json.loads(os.environ["SPECDEC_EOS_IDS"]))
        self.first_eos = {}
        self.step_index = 0
        self.last_flush = time.monotonic()
        self.step_started = self.last_flush
        self.counts = {}

    def schedule(self):
        self.step_started = time.perf_counter()
        return super().schedule()

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
        elapsed = time.perf_counter() - self.step_started
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
                record = {
                    "request_id": request_id, "step": self.step_index,
                    "prefix_tokens": prefix, "generated_before": generated,
                    "proposed": proposed, "accepted": accepted,
                    "emitted": len(tokens), "token_ids": tokens,
                    "draft_token_ids": scheduler_output.scheduled_spec_decode_tokens.get(request_id, []),
                    "phase": phase, "first_eos_output_index": first_eos,
                    "step_seconds": elapsed, "unix_time": time.time(),
                    "is_prefill": generated == 0,
                    "num_nans_in_logits": (model_runner_output.num_nans_in_logits or {}).get(request_id),
                }
                self.trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        if time.monotonic() - self.last_flush >= 10:
            self.trace_file.flush()
            self.last_flush = time.monotonic()
        return result

    def shutdown(self):
        try:
            self.trace_file.close()
        finally:
            super().shutdown()
