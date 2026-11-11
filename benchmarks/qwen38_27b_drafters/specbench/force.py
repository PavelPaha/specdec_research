"""Teacher forcing for speculative decoding in vLLM: the target "chooses" a fixed reference token.

Loaded in every vLLM process as a general plugin (see specbench/vllm_plugin/) when
SPECDEC_FORCE_DIR is set. For each request <key> the directory holds <key>.npy:
[prompt_len, ref_0, ref_1, ...] where ref_i is the token the target must emit at output
position i. Right before sampling (plain and rejection sampler, V1 and V2 model runners)
every logit row of a forced request is replaced by a one-hot at the reference token.
Greedy verification then accepts a draft token iff it equals the reference, so every
drafter is evaluated on byte-identical text, at every position.

Before masking, the target's own argmax (EOS excluded) is compared with the reference;
per-request agreement counts are written to SPECDEC_FORCE_DIR/agree-<pid>.json by TP rank 0
(a direct measure of how often the target alone would leave the reference text).
"""

import atexit
import json
import os
import time

import numpy as np

_DIR = os.environ.get("SPECDEC_FORCE_DIR")
_EOS = json.loads(os.environ.get("SPECDEC_EOS_IDS", "[]"))
_cache, _stats = {}, {}
_state = dict(last_flush=time.time(), errors=0, rank=None, installed=False)


def _ref(req_id):
    """(prompt_len, ref ids, key) for an engine request id '<key>' or '<key>-<suffix>'."""
    for key in (req_id, req_id.rsplit("-", 1)[0]):
        if key in _cache:
            return _cache[key]
        path = os.path.join(_DIR, key + ".npy")
        if os.path.exists(path):
            arr = np.load(path)
            _cache[key] = (int(arr[0]), arr[1:], key)
            _cache[req_id] = _cache[key]
            return _cache[key]
    _cache[req_id] = None
    return None


def _rank0():
    if _state["rank"] is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            _state["rank"] = get_tensor_model_parallel_rank()
        except Exception:
            _state["rank"] = 0
    return _state["rank"] == 0


def _flush(force=False):
    if not _rank0() or not (force or time.time() - _state["last_flush"] > 30):
        return
    _state["last_flush"] = time.time()
    path = os.path.join(_DIR, f"agree-{os.getpid()}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as h:
        json.dump(dict(errors=_state["errors"], last_error=_state.get("last_error"), requests=_stats), h)
    os.replace(tmp, path)


def _error(msg):
    """Forcing could not be applied in this step: count it and make it visible on disk."""
    _state["errors"] += 1
    _state["last_error"] = msg[:500]
    if _state["errors"] == 1:
        import sys
        print("specbench.force: forcing failed:", msg[:500], file=sys.stderr, flush=True)
    _flush(force=_state["errors"] == 1)


def force_rows(logits, req_ids, out_idx, first=None):
    """req_ids/out_idx: per logit row. Rows of unknown requests or past the reference are left as is.
    first: per row, True for the first row of its request in this step - the only row whose
    conditioning prefix is guaranteed to be the reference (later rows follow draft tokens)."""
    import torch
    rows, toks, keys, check = [], [], [], []
    for r, (rid, i) in enumerate(zip(req_ids, out_idx)):
        ref = _ref(rid)
        if ref is None or i < 0 or i >= len(ref[1]):
            continue
        rows.append(r)
        toks.append(int(ref[1][i]))
        keys.append(ref[2])
        check.append(first is None or first[r])
    if not rows:
        return logits
    rows_t = torch.tensor(rows, device=logits.device)
    toks_t = torch.tensor(toks, device=logits.device)
    if _rank0():
        sub = logits.index_select(0, rows_t).float()
        if _EOS:
            sub[:, _EOS] = float("-inf")
        own = sub.argmax(-1).eq(toks_t).cpu().tolist()
        for key, i, ok, c in zip(keys, (out_idx[r] for r in rows), own, check):
            if not c:
                continue
            s = _stats.setdefault(key, [0, 0, []])  # round starts checked, target argmax != reference, positions
            s[0] += 1
            if not ok:
                s[1] += 1
                if len(s[2]) < 64:
                    s[2].append(int(i))
    logits.index_fill_(0, rows_t, float("-inf"))
    logits[rows_t, toks_t] = 0.0
    _flush()
    return logits


# ---------- V1 model runner ----------
def _patch_v1():
    from vllm.v1.worker import gpu_model_runner as m
    orig = m.GPUModelRunner._sample
    if getattr(orig, "_specdec_forced", False):
        return

    def _sample(self, logits, spec_decode_metadata):
        if logits is not None:
            ib = self.input_batch
            n = ib.num_reqs
            nd = spec_decode_metadata.num_draft_tokens if spec_decode_metadata is not None else [0] * n
            req_ids, out_idx, first = [], [], []
            for rid, k in zip(ib.req_ids[:n], nd):
                base = len(self.requests[rid].output_token_ids)
                req_ids += [rid] * (k + 1)
                out_idx += range(base, base + k + 1)
                first += [True] + [False] * k
            if len(req_ids) == logits.shape[0]:
                force_rows(logits, req_ids, out_idx, first)
            else:
                _error("rows %d != logits %d" % (len(req_ids), logits.shape[0]))
        return orig(self, logits, spec_decode_metadata)

    _sample._specdec_forced = True
    m.GPUModelRunner._sample = _sample


# ---------- V2 model runner ----------
class _Forcing:
    """Wraps a V2 sampler / rejection sampler: __call__(logits, input_batch, ...)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, logits, input_batch, *args, **kwargs):
        try:
            n = input_batch.num_reqs
            cu = np.asarray(input_batch.cu_num_logits_np)
            comp = np.asarray(input_batch.num_computed_tokens_np)
            sched = np.asarray(input_batch.num_scheduled_tokens)
            req_ids, out_idx, first = [], [], []
            for b in range(n):
                rid = input_batch.req_ids[b]
                ref = _ref(rid)
                nrows = int(cu[b + 1] - cu[b])
                first += [True] + [False] * (nrows - 1)
                if ref is None:
                    req_ids += [rid] * nrows
                    out_idx += [-1] * nrows
                    continue
                # row j predicts sequence position comp + sched - nrows + j + 1
                start = int(comp[b] + sched[b] - nrows + 1 - ref[0])
                req_ids += [rid] * nrows
                out_idx += range(start, start + nrows)
            if len(req_ids) == logits.shape[0]:
                force_rows(logits, req_ids, out_idx, first)
            else:
                _error("rows %d != logits %d" % (len(req_ids), logits.shape[0]))
        except Exception as e:
            _error(repr(e))
        return self._inner(logits, input_batch, *args, **kwargs)


def _patch_v2():
    try:
        from vllm.v1.worker.gpu import model_runner as m
    except Exception:
        return
    cls = getattr(m, "GPUModelRunner", None) or getattr(m, "ModelRunner", None)
    if cls is None or getattr(cls.sample, "_specdec_forced", False):
        return
    orig = cls.sample

    def sample(self, *args, **kwargs):
        for name in ("sampler", "rejection_sampler"):
            inner = getattr(self, name, None)
            if inner is not None and not isinstance(inner, _Forcing):
                setattr(self, name, _Forcing(inner))
        return orig(self, *args, **kwargs)

    sample._specdec_forced = True
    cls.sample = sample


def install():
    """vLLM general plugin entry point (called in every vLLM process)."""
    if not _DIR or _state["installed"]:
        return
    _state["installed"] = True
    _patch_v1()
    _patch_v2()
    atexit.register(lambda: _flush(force=True))
