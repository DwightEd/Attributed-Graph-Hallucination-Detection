"""Generate target-model BoolQ answers without exposing gold booleans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .data import (
    BOOLQ_ANSWER_HEADER,
    compose_example,
    parse_bool_answer,
    read_json_records,
)
from .identity import model_source_signature
from .trace import assign_segment_ids


_PROMPT_VERSION = "boolq_yes_no_v3_exact_token_replay"


def _build_boolq_generation_example(passage: str, question: str):
    return compose_example(
        passage,
        question,
        "",
        example_id="generation-prompt",
        answer_header=BOOLQ_ANSWER_HEADER,
    )


def build_boolq_generation_prompt(passage: str, question: str) -> str:
    """Create the same passage/question context without a gold-answer field."""

    return _build_boolq_generation_example(passage, question).text


def _generation_fingerprint(
    record,
    *,
    model_signature: str,
    max_input_tokens: int | None,
    max_new_tokens: int,
) -> str:
    payload = {
        "schema": "boolq_prediction_v2",
        "prompt_version": _PROMPT_VERSION,
        "prompt": build_boolq_generation_prompt(
            str(record["passage"]), str(record["question"])
        ),
        "model_signature": model_signature,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _atomic_json_write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_path(directory: Path, example_id: str) -> Path:
    digest = hashlib.sha256(str(example_id).encode("utf-8")).hexdigest()[:32]
    return directory / f"{digest}.json"


def generate_boolq_predictions(
    model,
    tokenizer,
    records,
    *,
    output_path: str | Path,
    model_id: str,
    max_input_tokens: int | None,
    max_new_tokens: int,
    model_signature: str | None = None,
) -> int:
    """Generate answers from passage/question fields and write a label-free JSONL."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    signature = model_signature or model_id
    existing: dict[str, dict[str, object]] = {}
    if output.exists():
        existing = {
            str(row["id"]): row for row in read_json_records(output)
        }
    checkpoint_directory = output.parent / f"{output.name}.parts"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    invalid_predictions = []
    for checkpoint in checkpoint_directory.glob("*.json"):
        row = json.loads(checkpoint.read_text(encoding="utf-8"))
        existing[str(row["id"])] = row
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")
    selected: dict[str, dict[str, object]] = {}
    for index, record in enumerate(records):
        example_id = str(record.get("id", index))
        if example_id in selected:
            raise ValueError(f"Duplicate BoolQ id {example_id!r}")
        fingerprint = _generation_fingerprint(
            record,
            model_signature=signature,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )
        if example_id in existing:
            resumed = existing[example_id]
            if resumed.get("generation_fingerprint") != fingerprint:
                raise RuntimeError(
                    f"stale BoolQ prediction for {example_id!r}; use a new output "
                    "path or remove the old prediction checkpoints"
                )
            selected[example_id] = resumed
            continue
        prompt_example = _build_boolq_generation_example(
            str(record["passage"]), str(record["question"])
        )
        prompt = prompt_example.text
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        input_ids = encoded["input_ids"].to(device)
        if max_input_tokens is not None and input_ids.shape[1] > max_input_tokens:
            raise ValueError(
                f"BoolQ example {example_id!r} exceeds max_input_tokens="
                f"{max_input_tokens}; generation refuses truncation"
            )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer_tokens = generated[0, input_ids.shape[1] :].detach().cpu()
        special_ids = set(getattr(tokenizer, "all_special_ids", ()))
        special_ids.add(tokenizer.eos_token_id)
        while len(answer_tokens) and int(answer_tokens[-1]) in special_ids:
            answer_tokens = answer_tokens[:-1]
        model_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        if not model_answer.strip():
            invalid_predictions.append(
                {
                    "id": example_id,
                    "model_answer": model_answer,
                    "model_id": model_id,
                    "generation_fingerprint": fingerprint,
                    "reason": "empty_generation",
                }
            )
            print(f"Skipping BoolQ {example_id!r}: empty generation", flush=True)
            continue
        try:
            parse_bool_answer(model_answer)
        except ValueError as error:
            invalid_predictions.append(
                {
                    "id": example_id,
                    "model_answer": model_answer,
                    "model_id": model_id,
                    "generation_fingerprint": fingerprint,
                    "reason": str(error),
                }
            )
            print(f"Skipping BoolQ {example_id!r}: {error}", flush=True)
            continue
        prompt_offsets = torch.as_tensor(encoded["offset_mapping"])[0].tolist()
        prompt_special = (
            torch.as_tensor(encoded["special_tokens_mask"])[0].bool().tolist()
        )
        prompt_segment_ids = assign_segment_ids(
            prompt_offsets,
            prompt_example.segment_char_spans,
            prompt_special,
        )
        prompt_attention_mask = encoded.get("attention_mask")
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(encoded["input_ids"])
        answer_length = int(len(answer_tokens))
        prediction = {
            "id": example_id,
            "model_answer": model_answer,
            "model_id": model_id,
            "generation_fingerprint": fingerprint,
            "replay_input_ids": input_ids[0].detach().cpu().tolist()
            + answer_tokens.tolist(),
            "replay_attention_mask": (
                torch.as_tensor(prompt_attention_mask)[0].detach().cpu().tolist()
                + [1] * answer_length
            ),
            "replay_offset_mapping": prompt_offsets
            + [[len(prompt), len(prompt)]] * answer_length,
            "replay_special_tokens_mask": prompt_special + [False] * answer_length,
            "replay_segment_ids": prompt_segment_ids + [3] * answer_length,
        }
        selected[example_id] = prediction
        _atomic_json_write(
            _checkpoint_path(checkpoint_directory, example_id), prediction
        )
        if len(selected) == 1 or len(selected) % 25 == 0:
            print(f"Generated/resumed {len(selected)} BoolQ answers")

    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in selected.values()
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output)
    invalid_output = output.parent / f"{output.name}.invalid.jsonl"
    temporary_invalid_output = invalid_output.with_suffix(invalid_output.suffix + ".tmp")
    temporary_invalid_output.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in invalid_predictions
        )
        + ("\n" if invalid_predictions else ""),
        encoding="utf-8",
    )
    temporary_invalid_output.replace(invalid_output)
    return len(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate label-blind BoolQ answers.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype)
    ).to(args.device)
    model.eval()
    source_signature = model_source_signature(args.model, model=model, tokenizer=tokenizer)
    records = read_json_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    count = generate_boolq_predictions(
        model,
        tokenizer,
        records,
        output_path=args.output,
        model_id=args.model,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        model_signature=source_signature,
    )
    print(json.dumps({"predictions": count, "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
