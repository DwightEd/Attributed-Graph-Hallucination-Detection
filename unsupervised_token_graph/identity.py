"""Stable provenance signatures for local or Hub model/tokenizer sources."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


_MODEL_FILE_SUFFIXES = {
    ".bin",
    ".json",
    ".model",
    ".pt",
    ".safetensors",
    ".tiktoken",
    ".txt",
}


def _local_file_inventory(directory: Path) -> list[dict[str, object]]:
    inventory = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in _MODEL_FILE_SUFFIXES:
            continue
        stat = path.stat()
        inventory.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return inventory


def model_source_signature(source: str | Path, model=None, tokenizer=None) -> str:
    """Fingerprint checkpoint/tokenizer provenance without hashing multi-GB weights."""

    source_path = Path(source).expanduser()
    package_versions = {}
    for package_name in ("torch", "transformers", "tokenizers", "safetensors"):
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = None
    payload: dict[str, object] = {
        "source": str(source),
        "runtime": {
            "python": platform.python_version(),
            "packages": package_versions,
        },
    }
    if source_path.exists():
        resolved = source_path.resolve()
        payload["resolved_source"] = str(resolved)
        payload["files"] = (
            _local_file_inventory(resolved)
            if resolved.is_dir()
            else [
                {
                    "path": resolved.name,
                    "size": resolved.stat().st_size,
                    "mtime_ns": resolved.stat().st_mtime_ns,
                }
            ]
        )
    if model is not None:
        payload["model_class"] = type(model).__qualname__
        config = getattr(model, "config", None)
        if config is not None:
            payload["model_commit"] = getattr(config, "_commit_hash", None)
            payload["attention_implementation"] = getattr(
                config, "_attn_implementation", None
            )
            if hasattr(config, "to_dict"):
                payload["model_config"] = config.to_dict()
        try:
            parameter = next(model.parameters())
            payload["effective_parameter_dtype"] = str(parameter.dtype)
            payload["effective_parameter_device_type"] = parameter.device.type
        except (AttributeError, StopIteration):
            pass
    if tokenizer is not None:
        payload["tokenizer_class"] = type(tokenizer).__qualname__
        init_kwargs = getattr(tokenizer, "init_kwargs", {})
        payload["tokenizer_commit"] = init_kwargs.get("_commit_hash")
        payload["tokenizer_name"] = getattr(tokenizer, "name_or_path", None)
        payload["tokenizer_vocab_size"] = getattr(tokenizer, "vocab_size", None)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
