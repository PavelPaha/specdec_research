"""Explicit, lossless key adaptation for the pinned SpecDrift checkpoint."""

import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def adapt_eagle_checkpoint(spec):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    import torch

    source = Path(snapshot_download(spec["model"], revision=spec["revision"],
                                    allow_patterns=["config.json", "model.safetensors"],
                                    local_files_only=True))
    original = source / "model.safetensors"
    config = json.loads((source / "config.json").read_text())
    destination = Path.home() / ".cache/specdec_research" / f"eagle-fc-norm-v1-{spec['revision']}"
    manifest_path = destination / "adaptation.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert sha256(original) == manifest["source_sha256"]
        assert sha256(destination / "model.safetensors") == manifest["adapted_sha256"]
        return str(destination), manifest
    weights = load_file(original)
    layers = config["eagle_config"]["eagle_aux_hidden_state_layer_ids"]
    mapping = {f"fcs.{i}.weight": f"fc_norm.{i}.weight" for i in range(len(layers))}
    assert config["fc_norm"] is True
    assert {key for key in weights if key.startswith("fcs.")} == set(mapping)
    assert not any(key.startswith("fc_norm.") for key in weights)
    for key in mapping:
        assert weights[key].shape == (config["hidden_size"],)
    adapted = {mapping.get(key, key): tensor for key, tensor in weights.items()}
    destination.mkdir(parents=True, exist_ok=True)
    save_file(adapted, destination / "model.safetensors", metadata={"format": "pt"})
    shutil.copyfile(source / "config.json", destination / "config.json")
    reloaded = load_file(destination / "model.safetensors")
    assert set(reloaded) == set(adapted)
    for key, tensor in weights.items():
        assert torch.equal(tensor, reloaded[mapping.get(key, key)]), key
    manifest = dict(source_model=spec["model"], source_revision=spec["revision"],
                    source_sha256=sha256(original),
                    adapted_sha256=sha256(destination / "model.safetensors"),
                    key_mapping=mapping, all_tensors_bitwise_equal=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return str(destination), manifest
