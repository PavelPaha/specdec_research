"""Cache the exact target and draft revisions before offline inference."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    config = json.loads(Path("benchmarks/long_context/config.json").read_text())
    models = [(config["target"], config["target_revision"])]
    models.extend((draft["model"], draft["revision"])
                  for draft in config["drafters"].values() if "model" in draft)
    for model, revision in models:
        print(model, revision, flush=True)
        print(snapshot_download(
            model, revision=revision, max_workers=4,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt",
                            "*.model", "README.md", "LICENSE*"],
        ), flush=True)


if __name__ == "__main__":
    main()
