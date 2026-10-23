"""Extend storage for ordinary RoPE without changing its frequencies or scaling."""

import os


def extend_position_cache(config):
    limit = int(os.environ["SPECDEC_POSITION_CACHE_LEN"])
    text = getattr(config, "text_config", config)
    if isinstance(text, dict):
        rope = text.get("rope_parameters", text.get("rope_scaling")) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("This experiment requires unscaled default RoPE")
        text["max_position_embeddings"] = limit
    else:
        rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("This experiment requires unscaled default RoPE")
        text.max_position_embeddings = limit
    return config
